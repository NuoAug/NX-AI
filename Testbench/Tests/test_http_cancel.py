"""Real loopback sockets: cancellation before headers and during body reads."""
import json, threading, time
from concurrent.futures import CancelledError
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import pytest, requests
from cancel_http import LocalModelSession


@pytest.mark.parametrize('phase',['headers','body'])
def test_cancel_owns_only_its_request_and_does_not_wait_for_server(phase):
    entered=threading.Event();release=threading.Event();errors=[]
    class Handler(BaseHTTPRequestHandler):
        protocol_version='HTTP/1.1'
        def do_GET(self):
            self.send_response(200);self.send_header('Content-Length','2');self.end_headers();self.wfile.write(b'OK')
        def do_POST(self):
            self.rfile.read(int(self.headers['Content-Length']))
            body=b'{"ok":true}'
            if phase=='headers':entered.set();release.wait(3)
            try:
                self.send_response(200);self.send_header('Content-Length',str(len(body)));self.end_headers()
                if phase=='body':entered.set();release.wait(3)
                self.wfile.write(body)
            except (BrokenPipeError,ConnectionResetError):pass
        def log_message(self,*args):pass
    server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
    serving=threading.Thread(target=server.serve_forever,daemon=True);serving.start()
    url=f'http://127.0.0.1:{server.server_port}'
    cancel=threading.Event();session=LocalModelSession(cancel)
    def request():
        try:session.post(url,json={'q':'test'},timeout=5)
        except BaseException as exc:errors.append(exc)
    work=threading.Thread(target=request,daemon=True);work.start()
    try:
        assert entered.wait(2)
        began=time.monotonic();cancel.set();work.join(1)
        assert not work.is_alive() and time.monotonic()-began<1
        assert len(errors)==1 and isinstance(errors[0],CancelledError)
        assert requests.get(url,timeout=1).text=='OK'
        with pytest.raises(CancelledError):session.post(url,json={})
    finally:
        release.set();session.close();work.join(1)
        server.shutdown();server.server_close();serving.join(1)
    assert not session._watcher.is_alive()
