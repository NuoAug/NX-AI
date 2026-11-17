"""Cancel only this debug client's HTTP sockets, including header waits.

The local model uses HTTP. No shared/global urllib3 pools are modified.
"""
import socket
import sys
import threading
from concurrent.futures import CancelledError
import requests
from requests.adapters import HTTPAdapter
from urllib3.connection import HTTPConnection
from urllib3.connectionpool import HTTPConnectionPool

# Windows: 这里的 socket 都带 timeout（requests 会 settimeout），在途读实际阻塞在
# select 上；shutdown() 不会唤醒 select，而 socket.close() 在 http.client 调过
# makefile()（io_refs>0）时只是延迟关闭，fd 依然有效。因此必须让句柄真正作废。
_CLOSESOCKET = None
if sys.platform == 'win32':
    try:
        import ctypes

        _ws2 = ctypes.WinDLL('ws2_32', use_last_error=True)
        _ws2.closesocket.argtypes = [ctypes.c_size_t]
        _ws2.closesocket.restype = ctypes.c_int
        _CLOSESOCKET = _ws2.closesocket
    except Exception:  # noqa: BLE001
        _CLOSESOCKET = None


def _wake_blocked_reader(sock):
    """让在途的读立刻失败返回；重复调用幂等，不抛异常。"""
    if _CLOSESOCKET is not None:
        try:
            handle = sock.detach()   # 交出句柄所有权，避免后续被二次关闭
        except OSError:
            handle = None
        if handle is not None and handle >= 0:
            try:
                _CLOSESOCKET(handle)
            except OSError:
                pass
            return
    try:
        sock.close()
    except OSError:
        pass


class LocalModelSession(requests.Session):
    def __init__(self, cancel_event):
        super().__init__()
        self._cancel_event = cancel_event
        self._closed = threading.Event()
        self._socket_lock = threading.Lock()
        self._sockets = set()
        owner = self

        class Connection(HTTPConnection):
            def connect(self):
                super().connect()
                owner._remember(self.sock)

            def close(self):
                sock = self.sock
                try:
                    super().close()
                finally:
                    with owner._socket_lock:
                        owner._sockets.discard(sock)

        class Pool(HTTPConnectionPool):
            ConnectionCls = Connection

        class Adapter(HTTPAdapter):
            def init_poolmanager(self, *args, **kwargs):
                super().init_poolmanager(*args, **kwargs)
                self.poolmanager.pool_classes_by_scheme = dict(self.poolmanager.pool_classes_by_scheme)
                self.poolmanager.pool_classes_by_scheme['http'] = Pool

        self.mount('http://', Adapter())
        self._watcher = threading.Thread(target=self._watch, name='debug-model-cancel', daemon=True)
        self._watcher.start()

    def _remember(self, sock):
        with self._socket_lock:
            self._sockets.add(sock)
        if self._cancel_event.is_set():
            self._abort()

    def _abort(self):
        with self._socket_lock:
            sockets = tuple(self._sockets)
            self._sockets.clear()
        for sock in sockets:
            # 只 shutdown 不够：对端暂时不写数据时，阻塞在 recv 的线程不会被唤醒
            # （实测 Windows 上 shutdown 后仍阻塞 >1.5s）。_wake_blocked_reader 在
            # Windows 上用 detach()+closesocket() 让句柄真正作废，其它平台用
            # sock.close()；重复调用均为幂等。
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            _wake_blocked_reader(sock)

    def _watch(self):
        while not self._closed.wait(.02):
            if self._cancel_event.is_set():
                self._abort()
                return

    def send(self, request, **kwargs):
        if self._cancel_event.is_set():
            raise CancelledError('Model request cancelled')
        try:
            response = super().send(request, **kwargs)
        except (requests.RequestException, OSError):
            # 中断在途读会让 urllib3 直接抛 OSError（WinError 10038）：已取消就归一为
            # CancelledError，未取消仍原样抛出。
            if self._cancel_event.is_set():
                raise CancelledError('Model request cancelled') from None
            raise
        if self._cancel_event.is_set():
            response.close()
            raise CancelledError('Model request cancelled')
        return response

    def close(self):
        self._closed.set()
        self._abort()
        super().close()
        if self._watcher is not threading.current_thread():
            self._watcher.join(timeout=1)
