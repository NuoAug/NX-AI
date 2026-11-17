import json
from .test_server import ServerTestBase

class ControlRegressionTests(ServerTestBase):
    def test_existing_waiter_acquires_released_control(self):
        self.assertTrue(self.ctx.heartbeat('old')['owner'])
        self.assertFalse(self.ctx.heartbeat('waiter')['owner'])
        self.ctx.controller = None
        self.assertTrue(self.ctx.heartbeat('waiter')['owner'])

    def claim(self, origin=None):
        return self.request('POST', '/api/client/claim', body=b'{}', headers={
            'Origin': origin or f'http://127.0.0.1:{self.ctx.port}',
            'Content-Type': 'application/json', 'X-Bench-Client': 'new'})

    def test_explicit_idle_takeover_keeps_new_owner(self):
        self.serve()
        self.ctx.heartbeat('old')
        status, body, _ = self.claim()
        self.assertEqual(status, 200)
        self.assertTrue(json.loads(body)['owner'])
        self.assertFalse(self.ctx.heartbeat('old')['owner'])
        self.assertEqual(self.ctx.controller, 'new')

    def test_active_work_cannot_be_taken_over(self):
        self.serve()
        for flag in ('_run_active', '_debug_active', '_preview_active'):
            self.ctx.controller = 'old'
            setattr(self.ctx, flag, True)
            try:
                status, _, _ = self.claim()
                self.assertEqual(status, 409)
                self.assertEqual(self.ctx.controller, 'old')
            finally:
                setattr(self.ctx, flag, False)

    def test_cross_origin_takeover_rejected(self):
        self.serve()
        self.ctx.heartbeat('old')
        status, _, _ = self.claim('https://example.invalid')
        self.assertEqual(status, 403)
        self.assertEqual(self.ctx.controller, 'old')
