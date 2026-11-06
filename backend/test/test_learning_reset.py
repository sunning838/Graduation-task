"""Scoped reset and late-response regressions using disposable databases only."""
import json
import sqlite3
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import MagicMock, patch
from uuid import uuid4

from fastapi.testclient import TestClient

import api
from backend.api_store import APIStore, StateError
from backend.cert_config import CERT_CONFIG
from backend.learning import Progress
from backend.learning_reset import reset_learning
from backend import learning_routes as routes, mock_exams as bank, mock_preparation as prep


class LearningResetTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.store = APIStore(root / 'user_data.db')
        self.progress = Progress(root / 'learning.db')
        self.lessons = [dict(id='eip-a', cert='EIP', version='v1'),
                        dict(id='eip-b', cert='EIP', version='v1'),
                        dict(id='lrea-a', cert='LREA_1', version='v1')]
        self.patches = [patch.object(routes, '_progress', return_value=self.progress),
                        patch.object(routes, '_catalog', return_value=(CERT_CONFIG, self.lessons)),
                        patch.object(api, 'get_engine', return_value=MagicMock()),
                        patch.object(prep, 'dispatch')]
        self.progress_mock, _, self.engine_factory, self.dispatch = [p.start() for p in self.patches]
        api.app.dependency_overrides[api.get_store] = lambda: self.store
        self.client = TestClient(api.app)

    def tearDown(self):
        self.client.close()
        api.app.dependency_overrides.clear()
        for p in reversed(self.patches):
            p.stop()
        self.tmp.cleanup()

    def quiz(self, cert='EIP'):
        return dict(topic=next(iter(CERT_CONFIG[cert]['topics'])), question=str(uuid4()),
                    options=[str(i) for i in range(CERT_CONFIG[cert]['option_count'])],
                    answer=1, explanation='설명')

    def seed(self, cert='EIP'):
        quiz = self.quiz(cert)
        attempt = self.store.create_attempt(cert, quiz)
        self.store.submit(attempt, 2)
        self.store.create_attempt(cert, quiz)
        session = str(uuid4())
        with self.store.connect() as db:
            db.execute('INSERT INTO socratic_sessions(id,attempt_id) VALUES(?,?)', (session, attempt))
            db.execute('INSERT INTO socratic_turns(session_id,revision,request_id,answer,evaluation) VALUES(?,1,?,?,?)',
                       (session, str(uuid4()), '답변', '{}'))
            db.execute('INSERT INTO summary_notes(id,cert,topic,markdown,metadata,created_at) VALUES(?,?,?,?,?,?)',
                       (str(uuid4()), cert, quiz['topic'], '요약', '{}', 'now'))
        self.store.begin_summary_generation(cert)
        bank.add_question(self.store, cert, quiz, {'kind': 'test'}, 'ready', {'kind': 'test'})
        parts = [dict(topic=quiz['topic'], count=1)]
        exam = bank.create_exam(self.store, cert, parts, str(uuid4()))
        bank.save_answer(self.store, exam['id'], exam['items'][0]['id'], 1)
        bank.submit_exam(self.store, exam['id'])
        prep_id = str(uuid4())
        prep.start(self.store, cert, parts, prep_id)
        conversation, revision, _ = self.store.conversation(None, cert)
        self.store.append_turn(conversation, revision, '질문', '답변')
        return attempt, exam['id'], prep_id

    def reset(self, include=False, key=None, **extra):
        return self.client.post('/api/learning/reset', json={
            'cert': 'EIP', 'learner': '내 학습', 'include_progress': include,
            'confirm': True, 'request_id': key or str(uuid4()), **extra})

    def run_reset(self, include=False, key=None):
        return reset_learning(self.store, 'EIP', '내 학습', include, key or str(uuid4()),
                              self.progress if include else None, self.lessons)

    def count(self, table, where='', params=()):
        with self.store.connect() as db:
            return db.execute(f'SELECT COUNT(*) FROM {table} {where}', params).fetchone()[0]

    def test_default_clears_related_records_but_preserves_progress_and_shared_data(self):
        self.seed()
        other = self.seed('LREA_1')
        self.progress.save('내 학습', self.lessons[0], True)
        response = self.reset()
        self.assertEqual(response.status_code, 200, response.text)
        self.progress_mock.assert_not_called()
        for table in ('quiz_logs', 'api_attempts', 'mock_exams', 'summary_notes', 'summary_generation'):
            self.assertEqual(self.count(table, 'WHERE cert=?', ('EIP',)), 0, table)
            self.assertGreater(self.count(table, 'WHERE cert=?', ('LREA_1',)), 0, table)
        self.assertEqual(self.count('socratic_sessions'), 1)
        self.assertEqual(self.count('socratic_turns'), 1)
        self.assertEqual(self.count('mock_exam_items'), 2)
        self.assertEqual(self.count('question_bank'), 2)
        self.assertEqual(self.count('api_conversations'), 2)
        self.assertEqual(self.count('api_messages'), 4)
        self.assertEqual(self.progress.states('내 학습', self.lessons)['eip-a'][0], 'complete')
        self.assertEqual(bank.exam_view(self.store, other[1])['status'], 'submitted')
        stats = self.client.get('/api/stats?cert=EIP').json()
        self.assertEqual((stats['total_solved'], stats['today_solved'], stats['correct_solved']), (0, 0, 0))
        self.assertIsNone(stats['accuracy'])
        self.assertEqual(stats['weakness']['status'], 'insufficient_data')
        self.assertEqual(self.client.get('/api/summary-notes?cert=EIP').json()['notes'], [])

    def test_optional_progress_is_scoped_to_cert_and_learner(self):
        for lesson in self.lessons:
            self.progress.save('내 학습', lesson, True)
        self.progress.save('다른 학습', self.lessons[0], True)
        response = self.reset(True)
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()['deleted']['progress'], 2)
        self.assertEqual(set(self.progress.states('내 학습', self.lessons)), {'lrea-a'})
        self.assertEqual(set(self.progress.states('다른 학습', self.lessons)), {'eip-a'})
        snapshot = self.client.get('/api/learning/progress?learner=내 학습&cert=EIP').json()
        self.assertEqual(snapshot, {'states': {}, 'reset_revision': 1})

    def test_reset_retry_does_not_delete_newly_created_learning(self):
        key = str(uuid4())
        self.seed()
        first = self.reset(True, key)
        attempt = self.store.create_attempt('EIP', self.quiz())
        self.store.submit(attempt, 1)
        self.progress.save('내 학습', self.lessons[0], True, reset_revision=1)
        second = self.reset(True, key)
        self.assertEqual(first.json(), second.json())
        self.assertEqual(self.count('quiz_logs'), 1)
        self.assertIn('eip-a', self.progress.states('내 학습', self.lessons))
        self.assertEqual(self.store.learning_generation('EIP'), 1)

    def test_conflicting_reset_id_and_invalid_confirmation_never_delete(self):
        key = str(uuid4())
        self.reset(key=key)
        attempt = self.store.create_attempt('EIP', self.quiz())
        self.store.submit(attempt, 1)
        self.assertEqual(self.reset(True, key).status_code, 409)
        for extra in ({'confirm': False}, {'confirm': 'true'}, {'cert': 'INVALID'}, {'include_progress': 'true'}):
            self.assertEqual(self.reset(**extra).status_code, 422)
        response = self.client.post('/api/learning/reset', json={'cert': 'EIP', 'confirm': True})
        self.assertEqual(response.status_code, 422)
        self.assertEqual(self.count('quiz_logs'), 1)

    def test_optional_progress_failure_rolls_back_both_databases(self):
        self.seed()
        self.progress.save('내 학습', self.lessons[0], True)
        with self.progress.connect() as db:
            db.execute("""CREATE TRIGGER prevent_delete BEFORE DELETE ON progress
                BEGIN SELECT RAISE(ABORT, 'test rollback'); END""")
        before = self.count('quiz_logs')
        with self.assertRaises(sqlite3.IntegrityError):
            self.run_reset(True)
        self.assertEqual(self.count('quiz_logs'), before)
        self.assertGreater(self.count('api_attempts'), 0)
        self.assertEqual(self.store.learning_generation('EIP'), 0)
        self.assertEqual(self.count('learning_reset_requests'), 0)
        self.assertEqual(self.progress.snapshot('내 학습', self.lessons, 'EIP')[1], 0)
        self.assertIn('eip-a', self.progress.states('내 학습', self.lessons))

    def test_old_attempts_and_exams_cannot_be_submitted_after_reset(self):
        attempt, exam, _ = self.seed()
        self.reset()
        self.assertEqual(self.client.post('/api/quiz/submit', json={'attempt_id': attempt, 'selected_answer': 1}).status_code, 404)
        self.assertEqual(self.client.post(f'/api/mock-exams/{exam}/submit', json={}).status_code, 404)
        self.assertEqual(self.count('quiz_logs'), 0)

    def test_generation_finishing_after_reset_does_not_store_an_attempt(self):
        quiz = self.quiz()
        def generate(**_kwargs):
            self.run_reset()
            return quiz
        self.engine_factory.return_value.generate_advanced_quiz.side_effect = generate
        response = self.client.post('/api/quiz', json={'cert': 'EIP'})
        self.assertEqual(response.status_code, 409, response.text)
        self.assertEqual(response.json()['detail']['code'], 'LEARNING_RESET')
        self.assertEqual(self.count('api_attempts'), 0)

    def test_socratic_evaluation_finishing_after_reset_cannot_recreate_sessions(self):
        attempt = self.store.create_attempt('EIP', self.quiz())
        self.store.submit(attempt, 1)
        def evaluate(**_kwargs):
            self.run_reset()
            return dict(status='complete', question='', acknowledgment='잘 설명했습니다.', hint_level=0)
        self.engine_factory.return_value.socratic_dialogue_turn.side_effect = evaluate
        response = self.client.post('/api/quiz/socratic', json=dict(
            attempt_id=attempt, session_id=str(uuid4()), request_id=str(uuid4()), revision=0, answer='내 설명'))
        self.assertEqual(response.status_code, 409, response.text)
        self.assertEqual(self.count('socratic_sessions'), 0)
        self.assertEqual(self.count('socratic_turns'), 0)

    def test_summary_token_is_invalidated_and_late_result_cannot_return(self):
        token = self.store.begin_summary_generation('EIP')
        self.run_reset()
        with self.assertRaises(StateError):
            self.store.save_summary_note('EIP', token, 'database', '이전 요약', {}, 'now')
        self.store.finish_summary_generation('EIP', token, [])
        self.assertEqual(self.count('summary_notes'), 0)
        self.assertFalse(self.store.summary_generation_state('EIP')['generating'])

    def test_cancelled_preparation_cannot_restart_from_an_old_browser_request(self):
        quiz = self.quiz()
        parts = [dict(topic=quiz['topic'], count=1)]
        prep.start(self.store, 'EIP', parts, 'old-request')
        def generate(**_kwargs):
            self.run_reset()
            return quiz
        engine = MagicMock()
        engine.generate_advanced_quiz.side_effect = generate
        with patch.object(prep, 'review_candidate', return_value={'kind': 'test'}):
            prep.run(self.store, 'old-request', lambda: engine)
        state = prep.view(self.store, 'old-request')
        self.assertEqual(state['status'], 'cancelled')
        self.assertEqual(self.count('mock_exams'), 0)
        self.assertEqual(self.count('question_bank'), 1)  # Shared stock is retained.
        response = self.client.post('/api/mock-preparations', json=dict(
            cert='EIP', distribution=parts, request_id='old-request'))
        self.assertEqual(response.status_code, 409)
        self.dispatch.assert_not_called()
        with self.assertRaises(StateError):
            bank.create_exam(self.store, 'EIP', parts, 'old-request')

    def test_old_generation_cannot_assemble_exam_after_reset(self):
        quiz = self.quiz()
        bank.add_question(self.store, 'EIP', quiz, {}, 'ready', {'kind': 'test'})
        old = self.store.learning_generation('EIP')
        self.run_reset()
        with self.assertRaises(StateError):
            bank.create_exam(self.store, 'EIP', [dict(topic=quiz['topic'], count=1)], 'late-exam', generation=old)
        self.assertEqual(self.count('mock_exams'), 0)

    def test_stale_progress_save_is_rejected_but_new_revision_can_save(self):
        self.progress.save('내 학습', self.lessons[0], True, reset_revision=0)
        self.reset(True)
        body = dict(learner='내 학습', lesson_id='eip-a', complete=True, reset_revision=0)
        response = self.client.post('/api/learning/progress', json=body)
        self.assertEqual(response.status_code, 409, response.text)
        self.assertEqual(response.json()['detail']['code'], 'PROGRESS_RESET')
        self.assertEqual(self.progress.states('내 학습', self.lessons), {})
        response = self.client.post('/api/learning/progress', json={**body, 'reset_revision': 1})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()['states']['eip-a']['status'], 'complete')

    def test_concurrent_reset_retries_only_advance_generation_once(self):
        self.seed()
        key = str(uuid4())
        with ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(lambda _: self.run_reset(True, key), range(4)))
        self.assertTrue(all(result == results[0] for result in results))
        self.assertEqual(self.store.learning_generation('EIP'), 1)
        self.assertEqual(self.progress.snapshot('내 학습', self.lessons, 'EIP')[1], 1)


if __name__ == '__main__':
    unittest.main()
