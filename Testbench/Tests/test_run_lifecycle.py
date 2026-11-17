import asyncio,tempfile,threading
from pathlib import Path
from types import SimpleNamespace
from unittest import IsolatedAsyncioTestCase
from unittest.mock import Mock,AsyncMock,patch
from contextlib import ExitStack
from aieyes.config import AppConfig
from worker import RunSession

class RunLifecycleTests(IsolatedAsyncioTestCase):
    async def test_started_run_waits_for_explicit_stop(self):
        cfg=AppConfig();worker=SimpleNamespace(cfg=cfg,bench={},audio_in={},audio_out={},emit_state=Mock(),devices_patch=lambda:{},log=Mock())
        with tempfile.TemporaryDirectory() as tmp:
            session=RunSession(worker,'test',Path(tmp))
            sink=Mock();frontend=Mock();engine=SimpleNamespace(start=AsyncMock())
            session._assemble=lambda:(setattr(session,'engine',engine),setattr(session,'deps',SimpleNamespace(sink=sink,frontend=frontend,contract=None)))
            session._cleanup=AsyncMock()
            task=asyncio.create_task(session._main())
            for _ in range(10):
                await asyncio.sleep(0)
                if session._ready.is_set():break
            self.assertTrue(session._ready.is_set())
            self.assertFalse(task.done())
            session._cleanup.assert_not_awaited()
            session._stop_event.set();await task
            session._cleanup.assert_awaited_once()

    def _module_or_stub(self, path):
        """导入模块；本机缺少 POSIX 依赖（fcntl）时用桩模块占位。

        worker._assemble 里的 import 是局部 import，补丁必须打在这些真实导入点上，
        所以这里按 _assemble 的导入表逐条准备；Windows 上 chat_adapter /
        aieyes.scene.providers / scene_guard / ocr_adapter / vlm_adapter 依赖 fcntl
        无法导入，用 ModuleType 桩替代，验证的仍是 worker 的装配接线。
        """
        import importlib
        import sys
        import types
        try:
            return importlib.import_module(path)
        except ImportError:
            parts = path.split('.')
            for i in range(1, len(parts) + 1):
                name = '.'.join(parts[:i])
                if name in sys.modules:
                    continue
                mod = types.ModuleType(name)
                mod.__path__ = []
                sys.modules[name] = mod
                if i > 1:
                    setattr(sys.modules['.'.join(parts[:i - 1])], parts[i - 1], mod)
            return sys.modules[path]

    async def test_build_engine_dictionary_is_adapted_for_runtime(self):
        # 补丁必须打在 worker._assemble 真正 import 的位置：那里是
        # `from speech_vad import SpeechVad as EnergyVad`（局部 import），
        # patch('aieyes.audio.vad.EnergyVad') 打不到，会构造真实 VAD。
        cfg=AppConfig();worker=SimpleNamespace(cfg=cfg,bench={},ensure_hub=lambda:object(),emit_event=Mock())
        targets=['intent_adapter.install','devices.require_audio','aieyes.app._make_source','aieyes.audio.output.AudioSink','speech_vad.SpeechVad','chat_adapter.make_chat','aieyes.events.EventBus','aieyes.logging_setup.JsonlEvidenceLogger','aieyes.scene.providers.CameraFrameProvider','scene_guard.SceneService','asr_adapter.StreamingAsr','aieyes.speech.frontend.SpeechFrontend','kws_adapter.SherpaKws','aieyes.speech.tts.PiperTts','ocr_adapter.OcrEngine','vlm_adapter.VlmClient','bridge.Bridge','media.CameraHubAdapter','media.FrameStore']
        for name in targets:
            self._module_or_stub(name.rsplit('.',1)[0])
        with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
            session=RunSession(worker,'test',Path(tmp))
            patches={name:stack.enter_context(patch(name,create=True)) for name in targets}
            deps={'sink':object(),'frontend':object(),'tts':object(),'contract':None}
            engine=object();stack.enter_context(patch('aieyes.app.build_engine',return_value=(engine,deps),create=True))
            session._assemble()
            self.assertIs(session.engine,engine)
            self.assertIs(session.deps.sink,deps['sink'])
            self.assertIs(session.deps.frontend,deps['frontend'])
            # 生效性验证：真正构造的是被打补丁的 VAD，参数取自 cfg.vad
            vad=patches['speech_vad.SpeechVad']
            vad.assert_called_once()
            self.assertIs(vad.call_args.args[0],cfg.vad)
            self.assertEqual(vad.call_args.kwargs['sample_rate'],cfg.audio.sample_rate)
            self.assertIs(session.vlm,patches['vlm_adapter.VlmClient'].return_value)
            # 事件归属：交给 Bridge 的 emit 必须绑定本轮 run_id，
            # 否则收尾期间仍在飞的旧线程事件会被 server 记到下一轮（bridge.py EmitFn 契约）。
            emit_arg=patches['bridge.Bridge'].call_args.args[0]
            self.assertIs(getattr(emit_arg,'__self__',None),session)
            self.assertIs(getattr(emit_arg,'__func__',None),session.__class__._emit_bridge_event)
