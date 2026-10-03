"""User-triggered preparation contracts without paid model calls."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch
from uuid import uuid4
from fastapi.testclient import TestClient
import api
from backend.api_store import APIStore
from backend import mock_exams as bank, mock_preparation as prep
from backend.question_bank_cli import worker_lock


class PreparationTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.store=APIStore(Path(self.tmp.name)/'test.db')
        self.parts=[{'topic':'database','count':2}]
        self.engine=MagicMock()
        self.engine.generate_advanced_quiz.side_effect=lambda **kw: self.quiz(kw['target_topic'])
        self.factory=MagicMock(return_value=self.engine)
        self.review=patch.object(prep,'review_candidate',return_value={'kind':'test'})
        self.review_mock=self.review.start()
        self.dispatch=patch.object(prep,'dispatch'); self.dispatch_mock=self.dispatch.start()
        api.app.dependency_overrides[api.get_store]=lambda:self.store
        self.client=TestClient(api.app)

    def tearDown(self):
        self.client.close(); api.app.dependency_overrides.clear()
        self.dispatch.stop(); self.review.stop(); self.tmp.cleanup()

    def quiz(self,topic='database'):
        return dict(topic=topic,question=str(uuid4()),options=['a','b','c','d'],answer=1,explanation='test explanation')

    def start(self,key='request'):
        return self.client.post('/api/mock-preparations',json=dict(cert='EIP',distribution=self.parts,request_id=key))

    def test_stocked_exam_needs_no_model(self):
        for _ in range(2): bank.add_question(self.store,'EIP',self.quiz(),{},'ready',{'kind':'test'})
        result=self.start().json()
        self.assertEqual(result['status'],'completed')
        self.dispatch_mock.assert_not_called()
        self.assertEqual(len(bank.exam_view(self.store,result['exam_id'])['items']),2)
        self.assertEqual(result['progress'],dict(total=2,ready=2,remaining=0,percent=100))

    def test_progress_counts_only_usable_requested_subjects_and_caps_surplus(self):
        self.parts=[{'topic':'database','count':2},{'topic':'software_design','count':2}]
        for _ in range(5): bank.add_question(self.store,'EIP',self.quiz(),{},'ready',{'kind':'test'})
        bank.add_question(self.store,'EIP',self.quiz('software_design'),{'kind':'import'})
        bank.add_question(self.store,'EIP',self.quiz('info_system'),{},'ready',{'kind':'test'})
        result=self.start().json()
        self.assertEqual(result['progress'],dict(total=4,ready=2,remaining=2,percent=50))
        bank.add_question(self.store,'EIP',self.quiz('software_design'),{},'ready',{'kind':'test'})
        restored=prep.view(APIStore(self.store.path),'request')
        self.assertEqual(restored['progress'],dict(total=4,ready=3,remaining=1,percent=75))

    def test_generates_shortage_and_restores_without_duplicate_calls(self):
        bank.add_question(self.store,'EIP',self.quiz(),{},'ready',{'kind':'test'})
        self.assertEqual(self.start().json()['status'],'queued')
        prep.run(self.store,'request',self.factory)
        result=prep.view(APIStore(self.store.path),'request')
        self.assertEqual(result['status'],'completed')
        self.engine.generate_advanced_quiz.assert_called_once()
        self.assertTrue(self.engine.generate_advanced_quiz.call_args.kwargs['strict_subject'])
        self.assertEqual(self.review_mock.call_args.kwargs['source']['kind'],'generated')
        self.assertEqual(self.start().json()['exam_id'],result['exam_id'])
        prep.run(self.store,'request',self.factory)
        self.engine.generate_advanced_quiz.assert_called_once()

    def test_pending_originals_are_checked_before_generating(self):
        for _ in range(2): bank.add_question(self.store,'EIP',self.quiz(),{'kind':'import'})
        self.start(); prep.run(self.store,'request',self.factory)
        self.assertEqual(prep.view(self.store,'request')['status'],'completed')
        self.engine.generate_advanced_quiz.assert_not_called()
        self.assertTrue(all(c.kwargs['source']['kind']=='import' for c in self.review_mock.call_args_list))

    def test_failed_candidates_are_bounded_and_no_partial_exam(self):
        self.start()
        with self.store.connect() as db:
            db.execute('UPDATE mock_preparations SET max_attempts=2')
        self.review_mock.side_effect=ValueError('PRIVATE rejection')
        prep.run(self.store,'request',self.factory)
        result=self.client.get('/api/mock-preparations/request').json()
        self.assertEqual(result['status'],'failed')
        self.assertNotIn('PRIVATE',json.dumps(result))
        self.assertNotIn('attempts',result)
        self.assertEqual(self.engine.generate_advanced_quiz.call_count,2)
        with self.store.connect() as db:
            self.assertEqual(db.execute('SELECT COUNT(*) FROM mock_exams').fetchone()[0],0)

    def test_worker_lock_prevents_overlapping_generation(self):
        self.start()
        with worker_lock(self.store): prep.run(self.store,'request',self.factory)
        self.factory.assert_not_called()
        self.assertEqual(prep.view(self.store,'request')['status'],'queued')
        prep.run(self.store,'request',self.factory)
        self.assertEqual(prep.view(self.store,'request')['status'],'completed')

    def test_crashed_running_job_resumes_with_saved_budget(self):
        self.start()
        with self.store.connect() as db:
            db.execute("UPDATE mock_preparations SET status='running',attempts=1,max_attempts=2")
        prep.run(self.store,'request',self.factory)
        self.assertEqual(self.engine.generate_advanced_quiz.call_count,1)
        self.assertEqual(prep.view(self.store,'request')['status'],'failed')

    def test_initialization_failure_stops_and_request_conflict(self):
        self.start()
        self.parts=[{'topic':'database','count':3}]
        self.assertEqual(self.start().status_code,409)
        self.factory.side_effect=RuntimeError('PRIVATE KEY failure')
        prep.run(self.store,'request',self.factory)
        self.factory.assert_called_once()
        self.assertEqual(prep.view(self.store,'request')['status'],'failed')
        self.assertNotIn('PRIVATE',json.dumps(prep.view(self.store,'request')))

    def test_status_only_resumes_existing_requests(self):
        self.assertEqual(self.client.get('/api/mock-preparations/missing').status_code,404)
        self.dispatch_mock.assert_not_called()
        self.start()
        self.client.get('/api/mock-preparations/request')
        self.assertEqual(self.dispatch_mock.call_count,2)
        self.parts=[{'topic':'database','count':101}]
        self.assertEqual(self.start('invalid').status_code,422)


if __name__=='__main__': unittest.main()
