"""Offline Socratic API regressions; never touches project databases or Gemini."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from fastapi.testclient import TestClient
import api
from backend.api_store import APIStore


class SocraticTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / 'test.db'
        self.store = APIStore(self.path)
        api.app.dependency_overrides[api.get_store] = lambda: self.store
        self.client = TestClient(api.app)
        self.patch = patch.object(api, 'get_engine')
        self.engine = self.patch.start().return_value
        self.engine.socratic_dialogue_turn.return_value = dict(
            status='continue', acknowledgment='순서에 대한 설명은 맞습니다.',
            question='단계별 결과물은 왜 필요할까요?', hint_level=1)
        self.attempt = self.store.create_attempt('EIP', dict(topic='S1', question='문제',
            options=['가', '나'], answer=1, explanation='서버 해설', code_block='code'))
        self.store.submit(self.attempt, 2)
        self.body = dict(attempt_id=self.attempt, session_id='session', request_id='r1',
                         revision=0, answer='모르겠습니다.')

    def tearDown(self):
        self.client.close()
        self.patch.stop()
        api.app.dependency_overrides.clear()
        self.temp.cleanup()

    def post(self, **changes):
        return self.client.post('/api/quiz/socratic', json={**self.body, **changes})

    def test_real_context_history_and_completion_are_persisted(self):
        self.assertEqual(self.post().status_code, 200)
        self.assertEqual(self.post(request_id='r2', revision=1).json()['status'], 'continue')
        args = self.engine.socratic_dialogue_turn.call_args.kwargs
        self.assertIn('서버 해설', args['quiz_question'])
        self.assertIn('code', args['quiz_question'])
        self.assertIn('모르겠습니다', args['history'])
        self.engine.socratic_dialogue_turn.return_value = dict(
            status='complete', acknowledgment='핵심 원리를 설명했습니다.', question='', hint_level=2)
        self.assertEqual(self.post(request_id='r3', revision=2, answer='원리 설명').json()['status'], 'complete')
        reloaded = APIStore(self.path)
        api.app.dependency_overrides[api.get_store] = lambda: reloaded
        data = self.client.get(f'/api/quiz/{self.attempt}/socratic/session').json()
        self.assertTrue(data['completed'])
        self.assertEqual(len(data['turns']), 3)
        with self.store.connect() as db:
            self.assertEqual(db.execute('SELECT COUNT(*) FROM quiz_logs').fetchone()[0], 1)
            self.assertEqual(db.execute('SELECT is_correct FROM quiz_logs').fetchone()[0], 0)
        self.assertEqual(self.post(request_id='r4', revision=3).status_code, 409)

    def test_duplicate_stale_and_reset(self):
        first = self.post()
        self.assertEqual(first.json(), self.post().json())
        self.assertEqual(self.engine.socratic_dialogue_turn.call_count, 1)
        self.assertEqual(self.post(answer='변경').status_code, 409)
        self.assertEqual(self.post(request_id='new').status_code, 409)
        self.assertEqual(self.post(session_id='restart').status_code, 200)

    def test_invalid_and_unsubmitted(self):
        self.assertEqual(self.post(answer=' ').status_code, 422)
        self.assertEqual(self.post(attempt_id='missing').status_code, 404)
        unsubmitted = self.store.create_attempt('EIP', {})
        self.assertEqual(self.post(attempt_id=unsubmitted).status_code, 409)
        self.engine.socratic_dialogue_turn.assert_not_called()

    def test_failure_and_malformed_output_do_not_save(self):
        self.engine.socratic_dialogue_turn.side_effect = RuntimeError('private detail')
        response = self.post()
        self.assertEqual(response.status_code, 502)
        self.assertNotIn('private detail', response.text)
        self.engine.socratic_dialogue_turn.side_effect = None
        self.engine.socratic_dialogue_turn.return_value = dict(status='complete')
        self.assertEqual(self.post().status_code, 502)
        data = self.client.get(f'/api/quiz/{self.attempt}/socratic/session').json()
        self.assertEqual(data['revision'], 0)

    def test_concurrent_turns_have_one_winner(self):
        barrier = Barrier(2)
        def answer(**kwargs):
            barrier.wait(timeout=5)
            return dict(status='continue', acknowledgment='피드백', question='질문', hint_level=1)
        self.engine.socratic_dialogue_turn.side_effect = answer
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(self.post, request_id=f'r{i}') for i in range(2)]
            self.assertEqual(sorted(f.result().status_code for f in futures), [200, 409])


if __name__ == '__main__':
    unittest.main()
