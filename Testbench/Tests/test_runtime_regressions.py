import os, threading
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch,Mock
from .test_server import ServerTestBase
import server as server_mod
from worker import Worker

class RuntimeServerTests(ServerTestBase):
    def test_python_symlink_is_not_resolved_out_of_venv(self):
        with patch('supervisor.WorkerProcess') as factory:
            server_mod.BenchServer._spawn_worker(self.ctx)
            executable = factory.call_args.kwargs['python_path']
            self.assertEqual(executable,os.path.abspath(server_mod.BENCH_DIR/'../.venv/bin/python'))
            # 语义：python 路径必须仍指向 venv 内的解释器（符号链接未被解析到 venv 之外）。
            # POSIX 用字面 '/.venv/bin/python'；Windows 分隔符是反斜杠，用等价路径段断言，
            # 不放宽 Linux 断言。
            if os.name=='nt':
                self.assertIn(os.path.join('.venv','bin','python'),executable)
            else:
                self.assertIn('/.venv/bin/python',executable)

    def test_idle_recycle_cannot_stop_new_active_task(self):
        owned=Mock(alive=True)
        self.ctx._worker=owned;self.ctx._debug_active=True
        self.ctx.stop_worker('idle_recycle')
        owned.stop.assert_not_called()
        self.assertIs(self.ctx._worker,owned)
        self.ctx._debug_active=False;self.ctx._worker=None

class RuntimeWorkerTests(TestCase):
    def test_finished_context_clears_owning_debug_run(self):
        worker=Worker.__new__(Worker);worker.lock=threading.RLock()
        context=SimpleNamespace(run_id='current',module='chat')
        worker.debug=SimpleNamespace(ctx=context)
        worker.devices_patch=lambda:{};worker.emit_state=Mock();worker.log=Mock()
        worker.on_debug_finished(context,'completed')
        self.assertIsNone(worker.debug)
        self.assertEqual(worker.emit_state.call_args.args[0]['debug']['status'],'completed')

    def test_old_context_cannot_complete_new_debug_run(self):
        worker=Worker.__new__(Worker);worker.lock=threading.RLock()
        worker.debug=SimpleNamespace(ctx=object())
        worker.emit_state=Mock();worker.log=Mock()
        worker.on_debug_finished(SimpleNamespace(run_id='old',module='chat'),'completed')
        self.assertIsNotNone(worker.debug)
        worker.emit_state.assert_not_called()

class CancellationRegressionTests(TestCase):
    def test_final_status_is_not_overwritten_by_stopping(self):
        worker=Worker.__new__(Worker);worker.lock=threading.RLock();states=[]
        context=SimpleNamespace(run_id='current',module='chat')
        worker.debug=SimpleNamespace(run_id='current',module='chat',stop=lambda:worker.emit_state({'debug':{'status':'interrupted'}}))
        worker.emit_state=lambda state:states.append(state['debug']['status'])
        worker._cmd_debug_stop({})
        self.assertEqual(states,['stopping','interrupted'])

    def test_stream_cancellation_suppresses_late_final_answer(self):
        from debug import DebugRunner
        context=SimpleNamespace(stopped=False)
        runner=DebugRunner.__new__(DebugRunner);runner.ctx=context;runner._input={'text':'test'}
        runner._log=Mock();runner._timing=Mock();runner._emit=Mock();runner._track=lambda name,obj:obj;runner._make_vlm=lambda:object()
        class Chat:
            def ask_stream(self,text):
                yield 'first'
                context.stopped=True
                yield 'late'
        runner._make_chat=lambda vlm:Chat()
        runner._run_chat()
        self.assertEqual(runner._emit.call_count,1)
        self.assertEqual(runner._emit.call_args.args[0],'assistant.delta')
