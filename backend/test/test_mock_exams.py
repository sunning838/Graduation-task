"""Offline mock exam contracts, transactions and question preparation checks."""
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
from backend import mock_exams as bank
from backend import question_bank_cli as cli


def question(topic='database', stem=None):
    return dict(topic=topic, question=stem or str(uuid4()), options=['alpha','beta','gamma','delta'], answer=2, explanation='Because beta is correct.')


class ExamTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.store = APIStore(Path(self.temp.name)/'state.db')
        api.app.dependency_overrides[api.get_store] = lambda: self.store
        self.client = TestClient(api.app)
        self.engine_patch = patch.object(api, 'get_engine', side_effect=AssertionError('Exam must not call AI'))
        self.engine_patch.start()

    def tearDown(self):
        self.client.close()
        api.app.dependency_overrides.clear()
        self.engine_patch.stop()
        self.temp.cleanup()

    def seed(self, n=3, topic='database', status='ready'):
        return [bank.add_question(self.store, 'EIP', question(topic), {'kind':'test'}, status, {'kind':'test'}) for _ in range(n)]

    def create(self, count=2, request_id=None, **extra):
        return self.client.post('/api/mock-exams',json=dict(cert='EIP',distribution=[dict(topic='database',count=count)],request_id=request_id or str(uuid4()),**extra))

    def logs(self):
        with self.store.connect() as db:
            return db.execute('SELECT COUNT(*) FROM quiz_logs').fetchone()[0]

    def test_shortage_atomic_and_pending_not_served(self):
        self.seed(2,status='pending')
        self.seed(1)
        response=self.create()
        self.assertEqual(response.status_code,409)
        self.assertEqual(response.json()['detail']['code'],'INSUFFICIENT_QUESTIONS')
        with self.store.connect() as db:
            self.assertEqual(db.execute('SELECT COUNT(*) FROM mock_exams').fetchone()[0],0)
        data=self.client.get('/api/question-bank/availability?cert=EIP').json()
        subject=next(s for s in data['subjects'] if s['topic']=='database')
        self.assertEqual((subject['ready'],subject['pending']),(1,2))

    def test_idempotent_create_and_hidden_answers(self):
        self.seed()
        first=self.create(request_id='same').json()
        self.assertEqual(first,self.create(request_id='same').json())
        self.assertEqual(self.create(1,request_id='same').status_code,409)
        self.assertEqual(len({q['id'] for q in first['items']}),2)
        self.assertNotIn('correct_answer',first['items'][0])
        self.assertNotIn('explanation',first['items'][0])
        self.assertNotIn('answer',first['items'][0])
        self.assertIsNone(first['result'])

    def test_restore_snapshot_and_grade_once(self):
        self.seed(2)
        exam=self.create().json(); eid=exam['id']; items=exam['items']
        for item,value in zip(items,[2,1]):
            self.assertEqual(self.client.put(f"/api/mock-exams/{eid}/answers/{item['id']}",json={'selected_answer':value}).status_code,200)
        # Subsequent bank changes must not change existing exam snapshots.
        with self.store.connect() as db:
            db.execute("UPDATE question_bank SET status='disabled',question=?",(json.dumps(question(stem='changed')),))
        fresh=APIStore(self.store.path)
        self.assertEqual(bank.exam_view(fresh,eid)['items'][0]['selected_answer'],2)
        self.assertEqual(bank.exam_view(fresh,eid)['items'][0]['question'],items[0]['question'])
        result=self.client.post(f'/api/mock-exams/{eid}/submit',json={}).json()
        self.assertEqual(result['result']['score'],50)
        self.assertEqual(result,self.client.post(f'/api/mock-exams/{eid}/submit',json={}).json())
        self.assertEqual(self.logs(),2)
        self.assertEqual(self.client.put(f"/api/mock-exams/{eid}/answers/{items[0]['id']}",json={'selected_answer':1}).status_code,409)
        stats=self.client.get('/api/stats?cert=EIP').json()
        self.assertEqual(stats['total_solved'],2)
        self.assertEqual(stats['correct_solved'],1)

    def test_unanswered_confirmation_and_clear(self):
        self.seed(1); exam=self.create(1).json(); eid=exam['id']; item=exam['items'][0]['id']
        bank.save_answer(self.store,eid,item,2)
        bank.save_answer(self.store,eid,item,None)
        self.assertEqual(self.client.post(f'/api/mock-exams/{eid}/submit',json={}).status_code,409)
        self.assertEqual(self.logs(),0)
        result=self.client.post(f'/api/mock-exams/{eid}/submit',json={'confirm_unanswered':True}).json()
        self.assertEqual(result['result']['unanswered'],1)
        self.assertEqual(result['result']['correct'],0)

    def test_parallel_submit_logs_only_once(self):
        self.seed(3); exam=self.create(3).json()
        with ThreadPoolExecutor(max_workers=4) as pool:
            results=list(pool.map(lambda _: bank.submit_exam(self.store,exam['id'],True),range(4)))
        self.assertTrue(all(r['status']=='submitted' for r in results))
        self.assertEqual(self.logs(),3)

    def test_invalid_requests_and_wrong_exam_item(self):
        self.seed(2); a=self.create(1).json(); b=self.create(1).json()
        url=f"/api/mock-exams/{a['id']}/answers/{a['items'][0]['id']}"
        for value in (0,5,True,'2',1.2):
            self.assertEqual(self.client.put(url,json={'selected_answer':value}).status_code,422)
        self.assertEqual(self.client.put(f"/api/mock-exams/{a['id']}/answers/{b['items'][0]['id']}",json={'selected_answer':1}).status_code,404)
        for count in (0,101,True,'5'):
            self.assertEqual(self.create(count).status_code,422)
        self.assertEqual(self.client.get('/api/question-bank/availability?cert=unknown').status_code,422)
        self.assertEqual(self.client.get('/api/mock-exams/missing').status_code,404)
        for distribution in ([{'topic':'database','count':1}]*2, [{'topic':'database','count':60},{'topic':'software_design','count':60}]):
            self.assertEqual(self.client.post('/api/mock-exams',json={'cert':'EIP','distribution':distribution,'request_id':str(uuid4())}).status_code,422)

    def test_unused_questions_preferred(self):
        self.seed(2)
        a=self.create(1).json(); b=self.create(1).json()
        self.assertNotEqual(a['items'][0]['question'],b['items'][0]['question'])

    def test_dedup_and_invalid_candidates(self):
        q=question(stem='A sufficiently long question about relational schema normalization')
        self.assertIsNotNone(bank.add_question(self.store,'EIP',q,{}))
        self.assertIsNone(bank.add_question(self.store,'EIP',{**q,'options':list(reversed(q['options']))},{}))
        self.assertIsNone(bank.add_question(self.store,'EIP',{**q,'question':q['question']+'?'},{}))
        for changes in ({'answer':True},{'options':['a']*4},{'validation_failed':True},{'topic':'unknown'}):
            with self.assertRaises(ValueError): bank.add_question(self.store,'EIP',{**q,**changes},{})
        with self.assertRaises(ValueError): bank.add_question(self.store,'EIP',question(),{},'ready')

    def test_parser_preserves_code_and_excludes_ambiguous(self):
        text='# 과목: 데이터베이스 구축 # 키워드: SQL\n문제 : 다음 결과는?\n```sql\nSELECT 1;\n```\n 1) A\n 2) B\n 3) C\n 4) D\n- 해설: 결과 설명\n- 정답: **2) B**'
        _,q,error=list(cli.parse_markdown('EIP',text))[0]
        self.assertIsNone(error); self.assertIn('SELECT 1',q['question']); self.assertEqual(q['answer'],2)
        self.assertEqual(list(cli.parse_markdown('EIP',text.replace('결과 설명','문제 오류')))[0][2],'ambiguous_answer')
        self.assertEqual(list(cli.parse_markdown('EIP',text.replace('데이터베이스 구축','알수없음')))[0][2],'unmapped_subject')

    def test_job_budget_failure_and_subject_guard(self):
        engine=MagicMock(); engine.generate_advanced_quiz.return_value=question('software_design')
        with patch.object(cli,'review_candidate') as review:
            job=cli.run_job(self.store,'replenish',dict(cert='EIP',topic='database',target=10,max_attempts=2),engine_factory=lambda:engine)
            review.assert_not_called()
        self.assertEqual(engine.generate_advanced_quiz.call_count,2)
        with self.store.connect() as db:
            row=db.execute('SELECT * FROM question_bank_jobs WHERE id=?',(job,)).fetchone()
        self.assertEqual((row['status'],row['attempts'],row['failed']),('budget_exhausted',2,2))

    def test_verify_promotes_only_success_and_resume(self):
        self.seed(2,status='pending')
        with patch.object(cli,'review_candidate',side_effect=[KeyboardInterrupt(),{'kind':'test'}]):
            job=cli.run_job(self.store,'verify',dict(cert='EIP',max_attempts=2),engine_factory=MagicMock)
            cli.run_job(self.store,None,None,resume=job,engine_factory=MagicMock)
        with self.store.connect() as db:
            row=db.execute('SELECT * FROM question_bank_jobs WHERE id=?',(job,)).fetchone()
            ready=db.execute("SELECT COUNT(*) FROM question_bank WHERE status='ready'").fetchone()[0]
        self.assertEqual(row['status'],'completed')
        self.assertEqual(row['attempts'],2)
        self.assertEqual(ready,1)

    def test_worker_lock_and_missing_source(self):
        with cli.worker_lock(self.store):
            with self.assertRaises(ValueError):
                with cli.worker_lock(self.store): pass
        engine=MagicMock()
        with patch.object(cli,'source_material',return_value={'sections':[]}):
            with self.assertRaises(ValueError): cli.review_candidate(engine,'EIP',question())
        engine.verify_quiz.assert_not_called()

    def test_atomic_rollback_if_logging_fails(self):
        self.seed(2); exam=self.create(2).json()
        with self.store.connect() as db:
            db.execute("CREATE TRIGGER fail_log BEFORE INSERT ON quiz_logs WHEN (SELECT COUNT(*) FROM quiz_logs)>0 BEGIN SELECT RAISE(ABORT,'test failure'); END")
        with self.assertRaises(sqlite3.IntegrityError): bank.submit_exam(self.store,exam['id'],True)
        self.assertEqual(self.logs(),0)
        self.assertEqual(bank.exam_view(self.store,exam['id'])['status'],'taking')
        with self.store.connect() as db:
            self.assertEqual(db.execute('SELECT COUNT(*) FROM mock_exam_items WHERE log_id IS NOT NULL').fetchone()[0],0)

    def test_parallel_creation_uses_same_request_once(self):
        self.seed(2)
        with ThreadPoolExecutor(max_workers=4) as pool:
            results=list(pool.map(lambda _: bank.create_exam(self.store,'EIP',[dict(topic='database',count=2)],'parallel'),range(4)))
        self.assertEqual(len({r['id'] for r in results}),1)

    def test_distribution_and_five_option_cert(self):
        self.seed(2); self.seed(1,'software_design')
        exam=bank.create_exam(self.store,'EIP',[dict(topic='database',count=2),dict(topic='software_design',count=1)],'mixed')
        self.assertEqual([q['topic'] for q in exam['items']],['software_design','database','database'])
        q=question('civil_law'); q['options'].append('epsilon'); q['answer']=5
        bank.add_question(self.store,'LREA_1',q,{},'ready',{'kind':'test'})
        exam=bank.create_exam(self.store,'LREA_1',[dict(topic='civil_law',count=1)],'law')
        bank.save_answer(self.store,exam['id'],exam['items'][0]['id'],5)
        self.assertEqual(bank.submit_exam(self.store,exam['id'])['result']['score'],100)

    def test_rejected_review_never_promotes_and_does_not_starve_others(self):
        ids=self.seed(2,status='pending'); processed=[]
        def reject(engine,cert,q,source=None):
            processed.append(q['question']); raise ValueError('unsupported answer')
        with patch.object(cli,'review_candidate',side_effect=reject):
            for _ in range(2): cli.run_job(self.store,'verify',dict(cert='EIP',max_attempts=1),engine_factory=MagicMock)
        self.assertEqual(len(set(processed)),2)
        with self.store.connect() as db:
            rows=db.execute('SELECT status,review FROM question_bank').fetchall()
        self.assertEqual(len(rows),len(ids))
        self.assertTrue(all(r['status']=='pending' and 'verification_failed' in r['review'] for r in rows))

    def test_successful_replenishment_stops_at_target(self):
        engine=MagicMock(); engine.generate_advanced_quiz.side_effect=lambda **kwargs: question(kwargs['target_topic'])
        with patch.object(cli,'review_candidate',return_value={'kind':'test'}):
            job=cli.run_job(self.store,'replenish',dict(cert='EIP',topic='database',target=2,max_attempts=2),engine_factory=lambda:engine)
        with self.store.connect() as db:
            row=db.execute('SELECT status,succeeded FROM question_bank_jobs WHERE id=?',(job,)).fetchone()
        self.assertEqual((row['status'],row['succeeded']),('completed',2))
        self.assertEqual(len(engine.generate_advanced_quiz.call_args.kwargs['generated_history']),1)

    def test_import_review_uses_original_without_concept_lookup(self):
        engine=MagicMock()
        engine.verify_imported_quiz.return_value={'is_valid':True,'feedback':'Consistent original'}
        source={'kind':'import','file':'EIP/quiz.md','file_hash':'original-hash','section':0}
        q=question()
        bank.add_question(self.store,'EIP',q,source)
        with patch.object(cli,'source_material',side_effect=AssertionError('Must not require concept data')):
            cli.run_job(self.store,'verify',dict(cert='EIP',max_attempts=1),engine_factory=lambda:engine)
        engine.verify_imported_quiz.assert_called_once_with(q,'EIP')
        engine.verify_quiz.assert_not_called()
        with self.store.connect() as db:
            row=db.execute('SELECT status,review FROM question_bank').fetchone()
        self.assertEqual(row['status'],'ready')
        review=json.loads(row['review'])
        self.assertEqual(review['source'],source)
        self.assertEqual(review['policy'],'trusted-quiz-original-v1')

    def test_import_contradiction_remains_pending(self):
        engine=MagicMock()
        engine.verify_imported_quiz.return_value={'is_valid':False,'feedback':'Answer contradicts explanation'}
        bank.add_question(self.store,'EIP',question(),{'kind':'import'})
        cli.run_job(self.store,'verify',dict(cert='EIP',max_attempts=1),engine_factory=lambda:engine)
        with self.store.connect() as db:
            row=db.execute('SELECT status,review FROM question_bank').fetchone()
        self.assertEqual(row['status'],'pending')
        self.assertIn('Answer contradicts explanation',row['review'])

    def test_generated_and_unknown_sources_still_require_concepts(self):
        engine=MagicMock()
        with patch.object(cli,'source_material',return_value={'sections':[]}):
            for source in (None,{'kind':'generated'}, {'kind':'unknown'}):
                with self.assertRaisesRegex(ValueError,'No matching subject concept source'):
                    cli.review_candidate(engine,'EIP',question(),source=source)
        engine.verify_imported_quiz.assert_not_called()
        engine.verify_quiz.assert_not_called()
        engine.verify_quiz.return_value={'is_valid':True}
        with patch.object(cli,'source_material',return_value={'hash':'concept-hash'}), patch.object(cli,'select_context',return_value=('concept text',[])):
            evidence=cli.review_candidate(engine,'EIP',question(),source={'kind':'generated'})
        self.assertEqual(evidence['source_hash'],'concept-hash')
        engine.verify_quiz.assert_called_once()

    def test_import_format_errors_prevent_ai_review(self):
        engine=MagicMock()
        with self.assertRaises(ValueError):
            cli.review_candidate(engine,'EIP',{**question(),'answer':6},source={'kind':'import'})
        engine.verify_imported_quiz.assert_not_called()

    def test_import_review_prompt_supports_configured_option_count(self):
        from backend.chat_engine import AITutorEngine, QuizVerification
        from langchain_core.output_parsers import JsonOutputParser
        from langchain_core.runnables import RunnableLambda
        engine=object.__new__(AITutorEngine)
        engine.verify_parser=JsonOutputParser(pydantic_object=QuizVerification)
        prompts=[]
        def respond(prompt):
            prompts.append(prompt.to_string())
            return json.dumps({'is_valid':True,'feedback':'Consistent'})
        engine.llm=RunnableLambda(respond)
        q=question('civil_law'); q['options'].append('epsilon'); q['answer']=5
        self.assertTrue(engine.verify_imported_quiz(q,'LREA_1')['is_valid'])
        self.assertIn('정확히 5개',prompts[0])
        self.assertIn('승인 조건이 아니다',prompts[0])
        self.assertIn(q['explanation'],prompts[0])


if __name__=='__main__': unittest.main()
