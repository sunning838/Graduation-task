"""Offline statistics, weakness selection, scoped retrieval and grading regressions."""
import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

import api
from backend.api_store import APIStore
from backend.cert_config import CERT_CONFIG
from backend.learning_stats import learning_stats


class StatsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.store = APIStore(Path(self.temp.name) / 'stats.db')
        api.app.dependency_overrides[api.get_store] = lambda: self.store
        self.client = TestClient(api.app)
        self.engine = MagicMock()
        self.engine.generate_advanced_quiz.side_effect = lambda **kwargs: dict(
            question='test', topic=kwargs.get('target_topic', 'database'),
            options=['1) A', '2) B', '3) C', '4) D'], answer=1, explanation='explanation')
        self.engine_patch = patch.object(api, 'get_engine', return_value=self.engine)
        self.engine_patch.start()

    def tearDown(self):
        self.client.close()
        self.engine_patch.stop()
        api.app.dependency_overrides.clear()
        self.temp.cleanup()

    def seed(self, topic, correct, total, cert='EIP', timestamp='2026-10-03 00:00:00'):
        with self.store.connect() as db:
            db.executemany('INSERT INTO quiz_logs(cert,topic,is_correct,timestamp) VALUES(?,?,?,?)',
                           [(cert, topic, int(i < correct), timestamp) for i in range(total)])

    def test_empty_stats_and_config_order(self):
        for cert in CERT_CONFIG:
            response = self.client.get('/api/stats', params={'cert': cert})
            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertFalse(data['has_records'])
            self.assertIsNone(data['accuracy'])
            self.assertEqual(data['total_solved'], 0)
            self.assertEqual([s['topic'] for s in data['subjects']], list(CERT_CONFIG[cert]['topics']))
            self.assertTrue(all(s['accuracy'] is None and s['status'] == 'unlearned' for s in data['subjects']))
            self.assertEqual(data['weakness']['status'], 'insufficient_data')

    def test_minimum_count_threshold_and_ties(self):
        self.seed('database', 0, 4)  # not enough evidence
        self.seed('software_design', 3, 5)  # exactly 60%, not weak
        self.seed('software_development', 2, 5)  # 40%
        self.seed('programming_language', 4, 10)  # same 40%, more attempts first
        data = self.client.get('/api/stats?cert=EIP').json()
        analysis = self.client.get('/api/weakness?cert=EIP').json()
        self.assertEqual(analysis['focus']['topic'], 'programming_language')
        self.assertEqual([s['topic'] for s in analysis['ranking']], ['programming_language', 'software_development'])
        self.assertEqual(data['weakness'], {k: v for k, v in analysis.items() if k != 'cert'})
        self.assertEqual(data['total_solved'], 24)
        self.assertEqual(data['correct_solved'], 9)
        self.assertEqual(data['accuracy'], 37.5)

    def test_all_correct_is_not_a_weakness(self):
        self.seed('database', 5, 5)
        data = self.client.get('/api/weakness?cert=EIP').json()
        self.assertEqual(data['status'], 'no_weakness')
        self.assertIsNone(data['focus'])
        self.assertIn('software_design', data['insufficient_subjects'])

    def test_today_uses_korean_midnight_and_is_cert_scoped(self):
        self.seed('database', 1, 1, timestamp='2026-10-02 14:59:59')
        self.seed('database', 1, 1, timestamp='2026-10-02 15:00:00')
        self.seed('database', 0, 1, timestamp='2026-10-03 14:59:59')
        self.seed('database', 1, 1, timestamp='2026-10-03 15:00:00')
        self.seed('civil_law', 1, 1, cert='LREA_1', timestamp='2026-10-03 00:00:00')
        data = learning_stats(self.store, 'EIP', now=datetime(2026, 10, 3, 0, tzinfo=timezone.utc))
        self.assertEqual(data['date'], '2026-10-03')
        self.assertEqual(data['today_solved'], 2)
        self.assertEqual(data['total_solved'], 4)
        self.assertEqual(data['goal_rate'], 4)
        self.assertEqual(self.client.get('/api/stats?cert=LREA_1').json()['total_solved'], 1)

    def test_unclassified_records_only_in_totals(self):
        self.seed('unknown_subject', 1, 8)
        data = self.client.get('/api/stats?cert=EIP').json()
        self.assertEqual(data['unclassified_solved'], 8)
        self.assertEqual(data['total_solved'], 8)
        self.assertEqual(data['weakness']['status'], 'insufficient_data')

    def test_weakness_generation_and_duplicate_submit_updates_stats_once(self):
        self.seed('database', 2, 5)
        response = self.client.post('/api/quiz', json={'cert': 'EIP', 'mode': 'weakness'})
        self.assertEqual(response.status_code, 200, response.text)
        self.engine.generate_advanced_quiz.assert_called_once_with(cert='EIP', target_topic='database', strict_subject=True)
        quiz = response.json()
        self.assertEqual(quiz['mode'], 'weakness')
        self.assertNotIn('answer', quiz)
        body = {'attempt_id': quiz['attempt_id'], 'selected_answer': 1}
        self.assertEqual(self.client.post('/api/quiz/submit', json=body).status_code, 200)
        self.assertEqual(self.client.post('/api/quiz/submit', json=body).status_code, 200)
        data = self.client.get('/api/stats?cert=EIP').json()
        self.assertEqual(data['total_solved'], 6)
        self.assertEqual(data['weakness']['focus']['accuracy'], 50.0)

    def test_no_random_fallback_when_no_weakness(self):
        response = self.client.post('/api/quiz', json={'cert': 'EIP', 'mode': 'weakness'})
        self.assertEqual(response.status_code, 409)
        self.assertEqual(response.json()['detail']['code'], 'INSUFFICIENT_HISTORY')
        self.seed('database', 3, 5)
        response = self.client.post('/api/quiz', json={'cert': 'EIP', 'mode': 'weakness'})
        self.assertEqual(response.json()['detail']['code'], 'NO_WEAK_TOPIC')
        self.engine.generate_advanced_quiz.assert_not_called()

    def test_missing_material_and_mismatched_topic_are_not_saved(self):
        self.seed('database', 0, 5)
        self.engine.generate_advanced_quiz.side_effect = None
        self.engine.generate_advanced_quiz.return_value = {'failure_code': 'NO_SUBJECT_MATERIAL', 'is_fallback': True}
        response = self.client.post('/api/quiz', json={'cert': 'EIP', 'mode': 'weakness'})
        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.json()['detail']['code'], 'NO_SUBJECT_MATERIAL')
        self.engine.generate_advanced_quiz.return_value = {'topic': 'software_design'}
        response = self.client.post('/api/quiz', json={'cert': 'EIP', 'mode': 'weakness'})
        self.assertEqual(response.status_code, 502)
        with self.store.connect() as db:
            self.assertEqual(db.execute('SELECT COUNT(*) FROM api_attempts').fetchone()[0], 0)

    def test_invalid_cert_and_mode(self):
        for route in ['/api/stats', '/api/weakness']:
            self.assertEqual(self.client.get(route, params={'cert': 'unknown'}).status_code, 422)
        self.assertEqual(self.client.post('/api/quiz', json={'mode': 'wrong'}).status_code, 422)


class SubjectRetrievalTests(unittest.TestCase):
    def make_engine(self):
        from backend.chat_engine import AITutorEngine, QuizResponse
        from langchain_core.output_parsers import JsonOutputParser
        from langchain_core.runnables import RunnableLambda
        engine = object.__new__(AITutorEngine)
        engine.vector_db = MagicMock()
        engine.quiz_parser = JsonOutputParser(pydantic_object=QuizResponse)
        engine.llm = RunnableLambda(lambda _: json.dumps(dict(question='문제', options=['1) A','2) B','3) C','4) D'], answer=1, explanation='해설')))
        engine.verify_quiz = MagicMock(return_value={'is_valid': True})
        return engine

    def test_concept_fallback_keeps_subject_scope(self):
        engine = self.make_engine()
        engine.vector_db.similarity_search.side_effect = [[], [SimpleNamespace(page_content='과목 원문')]]
        quiz = engine.generate_advanced_quiz(cert='EIP', target_topic='database', strict_subject=True)
        self.assertEqual(quiz['topic'], 'database')
        self.assertFalse(quiz.get('is_fallback'))
        calls = engine.vector_db.similarity_search.call_args_list
        self.assertEqual(len(calls), 2)
        for call in calls:
            filters = call.kwargs['filter']['$and']
            self.assertIn({'subject': 'database'}, filters)
            self.assertIn({'cert': 'EIP'}, filters)
        self.assertIn({'doc_type': 'concept'}, calls[1].kwargs['filter']['$and'])

    def test_scoped_quiz_retrieval(self):
        engine = self.make_engine()
        engine.vector_db.similarity_search.return_value = [SimpleNamespace(page_content='과목 기출')]
        quiz = engine.generate_advanced_quiz(cert='EIP', target_topic='database', strict_subject=True)
        self.assertEqual(quiz['topic'], 'database')
        self.assertIn({'subject': 'database'}, engine.vector_db.similarity_search.call_args.kwargs['filter']['$and'])

    def test_missing_both_sources_returns_explicit_reason(self):
        engine = self.make_engine()
        engine.vector_db.similarity_search.return_value = []
        quiz = engine.generate_advanced_quiz(cert='LREA_1', target_topic='housing_lease', strict_subject=True)
        self.assertEqual(quiz['failure_code'], 'NO_SUBJECT_MATERIAL')
        self.assertTrue(quiz['is_fallback'])
        self.assertEqual(engine.vector_db.similarity_search.call_count, 2)


if __name__ == '__main__':
    unittest.main()
