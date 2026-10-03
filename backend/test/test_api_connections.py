"""Offline API regressions. Uses a temporary database and a fake AI engine."""
import copy
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient
import api
from backend.api_store import APIStore, StateError
from backend.cert_config import CERT_CONFIG


class FakeEngine:
    def __init__(self):
        self.calls = []
        self.fail = False
        self.fallback = False

    def generate_response(self, **kwargs):
        self.calls.append(kwargs)
        if self.fail:
            raise RuntimeError('private error detail')
        return '설명입니다.'

    def generate_advanced_quiz(self, cert):
        config = CERT_CONFIG[cert]
        return dict(question='문제', topic=next(iter(config['topics'])),
                    options=[f'{i + 1}) 보기 {i + 1}' for i in range(config['option_count'])],
                    answer=config['option_count'], explanation='해설', is_fallback=self.fallback)


class ConnectionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / 'state.db'
        self.store = APIStore(self.path)
        self.engine = FakeEngine()
        api.app.dependency_overrides[api.get_store] = lambda: self.store
        self.engine_patch = patch.object(api, 'get_engine', return_value=self.engine)
        self.engine_patch.start()
        self.client = TestClient(api.app)

    def tearDown(self):
        self.client.close()
        self.engine_patch.stop()
        api.app.dependency_overrides.clear()
        self.temp.cleanup()

    def create(self, cert='EIP'):
        result = self.client.post('/api/quiz', json={'cert': cert})
        self.assertEqual(result.status_code, 200, result.text)
        return result.json()

    def count_logs(self):
        with self.store.connect() as db:
            return db.execute('SELECT COUNT(*) FROM quiz_logs').fetchone()[0]

    def test_certifications_and_selection(self):
        configs = self.client.get('/api/certifications').json()['certifications']
        self.assertEqual({c['id'] for c in configs}, set(CERT_CONFIG))
        for cert, config in CERT_CONFIG.items():
            quiz = self.create(cert)
            self.assertEqual(quiz['cert'], cert)
            self.assertEqual(len(quiz['options']), config['option_count'])
            self.assertNotIn('answer', quiz)
            self.assertNotIn('explanation', quiz)
            result = self.client.post('/api/quiz/submit', json={
                'attempt_id': quiz['attempt_id'], 'selected_answer': config['option_count']})
            self.assertTrue(result.json()['is_correct'])

    def test_invalid_cert_does_not_call_ai(self):
        for route, body in [('/api/quiz', {}), ('/api/chat', {'message': '질문'})]:
            response = self.client.post(route, json={**body, 'cert': 'INVALID'})
            self.assertEqual(response.status_code, 422)
            self.assertEqual(response.json()['detail']['code'], 'INVALID_CERT')
        self.assertEqual(self.engine.calls, [])

    def test_repeat_submit_and_conflicting_answer(self):
        quiz = self.create()
        body = {'attempt_id': quiz['attempt_id'], 'selected_answer': 1}
        first = self.client.post('/api/quiz/submit', json=body)
        second = self.client.post('/api/quiz/submit', json=body)
        self.assertEqual(first.json(), second.json())
        self.assertEqual(self.count_logs(), 1)
        conflict = self.client.post('/api/quiz/submit', json={**body, 'selected_answer': 2})
        self.assertEqual(conflict.status_code, 409)
        self.assertEqual(self.count_logs(), 1)

    def test_concurrent_submissions_log_once(self):
        quiz = self.create()
        def submit(_):
            return APIStore(self.path).submit(quiz['attempt_id'], 2)
        with ThreadPoolExecutor(max_workers=6) as pool:
            results = list(pool.map(submit, range(12)))
        self.assertTrue(all(result == results[0] for result in results))
        self.assertEqual(self.count_logs(), 1)

    def test_restart_and_old_client_alias(self):
        quiz = self.create()
        self.store = APIStore(self.path)
        result = self.client.post('/api/quiz/submit', json={'quiz_id': quiz['quiz_id'], 'selected_answer': 4})
        self.assertEqual(result.status_code, 200)
        self.assertTrue(result.json()['is_correct'])
        self.store = APIStore(self.path)
        self.assertEqual(self.store.submit(quiz['attempt_id'], 4), result.json())
        self.assertEqual(self.count_logs(), 1)

    def test_invalid_answers_do_not_log(self):
        quiz = self.create()
        for choice in [0, 5, -1, True, '1', 1.2]:
            response = self.client.post('/api/quiz/submit', json={
                'attempt_id': quiz['attempt_id'], 'selected_answer': choice})
            self.assertEqual(response.status_code, 422, response.text)
        self.assertEqual(self.count_logs(), 0)
        self.assertEqual(self.client.post('/api/quiz/submit', json={
            'attempt_id': 'missing', 'selected_answer': 1}).status_code, 404)

    def test_conversation_context_restart_and_cert_boundary(self):
        first = self.client.post('/api/chat', json={'cert': 'EIP', 'message': '첫 질문'}).json()
        self.store = APIStore(self.path)
        second = self.client.post('/api/chat', json={
            'cert': 'EIP', 'message': '방금 설명의 예시는?', 'conversation_id': first['conversation_id']})
        self.assertEqual(second.status_code, 200)
        history = self.engine.calls[-1]['chat_history']
        self.assertEqual([m.content for m in history], ['첫 질문', '설명입니다.'])
        wrong_cert = self.client.post('/api/chat', json={
            'cert': 'LREA_1', 'message': '질문', 'conversation_id': first['conversation_id']})
        self.assertEqual(wrong_cert.status_code, 409)
        fresh = self.client.post('/api/chat', json={'cert': 'EIP', 'message': '새 대화'}).json()
        self.assertNotEqual(fresh['conversation_id'], first['conversation_id'])
        self.assertEqual(self.engine.calls[-1]['chat_history'], [])

    def test_history_limit_and_concurrent_turn_conflict(self):
        cid, rev, _ = self.store.conversation(None, 'EIP')
        for i in range(10):
            self.store.append_turn(cid, i, f'q{i}', f'a{i}')
        _, revision, rows = self.store.conversation(cid, 'EIP')
        self.assertEqual(len(rows), 12)
        self.assertEqual(rows[0]['content'], 'q4')
        self.store.append_turn(cid, revision, 'new', 'answer')
        with self.assertRaises(StateError):
            self.store.append_turn(cid, revision, 'stale', 'answer')

    def test_generation_failure_is_safe_and_not_saved(self):
        self.engine.fail = True
        response = self.client.post('/api/chat', json={'message': '질문'})
        self.assertEqual(response.status_code, 502)
        self.assertNotIn('private error', response.text)
        self.assertTrue(response.json()['detail']['retryable'])
        with self.store.connect() as db:
            self.assertEqual(db.execute('SELECT COUNT(*) FROM api_messages').fetchone()[0], 0)
        self.engine.fallback = True
        response = self.client.post('/api/quiz', json={'cert': 'EIP'})
        self.assertEqual(response.status_code, 502)
        self.assertEqual(self.count_logs(), 0)

    def test_validation_and_cors(self):
        for payload in [{'message': '  '}, {'message': 'x', 'answer_length': 'invalid'}]:
            response = self.client.post('/api/chat', json=payload)
            self.assertEqual(response.status_code, 422)
            self.assertEqual(response.json()['detail']['code'], 'INVALID_REQUEST')
        response = self.client.options('/api/chat', headers={
            'Origin': 'http://localhost:5173', 'Access-Control-Request-Method': 'POST'})
        self.assertEqual(response.headers['access-control-allow-origin'], 'http://localhost:5173')


class EngineContractTests(unittest.TestCase):
    def test_four_and_five_options_without_truncation(self):
        from backend.chat_engine import AITutorEngine
        engine = object.__new__(AITutorEngine)
        for cert, count in [('EIP', 4), ('LREA_1', 5)]:
            quiz = dict(options=[f'{i+1}) 보기{i+1}' for i in range(count)], answer=count)
            normalized = engine._normalize_quiz_data(copy.deepcopy(quiz), 'topic', cert)
            self.assertEqual(len(normalized['options']), count)
            self.assertEqual(normalized['answer'], count)
        invalid = engine._normalize_quiz_data(quiz, 'topic', 'EIP')
        self.assertTrue(invalid['is_fallback'])


if __name__ == '__main__':
    unittest.main()
