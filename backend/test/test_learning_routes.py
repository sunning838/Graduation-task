"""Text learning and scoped practice contracts; no live AI or production DB."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient
import api
from backend import learning_routes as routes
from backend.learning import Progress
from backend.api_store import APIStore
from backend.cert_config import CERT_CONFIG


class LearningRoutesTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        root=Path(self.tmp.name)
        self.progress=Progress(root/'learning.db')
        self.store=APIStore(root/'state.db')
        self.lesson=dict(id='lesson-a',cert='EIP',subject='database',title='정규화 <출제됨>',body='정규화는 중복을 줄이는 과정입니다.',version='v1',source='data/EIP/database/a.md')
        self.lessons=[self.lesson,{**self.lesson,'id':'lesson-b','title':'두 번째 단원'}]
        self.patches=[patch.object(routes,'_catalog',return_value=(CERT_CONFIG,self.lessons)),patch.object(routes,'_progress',return_value=self.progress),
                      patch('backend.learning.teach',return_value=({'text':'### 정규화\n설명입니다 [1].','visual':None},[])),patch.object(api,'get_engine',return_value=MagicMock())]
        self.catalog,self.progress_mock,self.teach,self.engine_factory=[p.start() for p in self.patches]
        self.engine=self.engine_factory.return_value
        self.engine.verify_quiz.return_value={'is_valid':True}
        api.app.dependency_overrides[api.get_store]=lambda:self.store
        self.client=TestClient(api.app)
        self.quiz=dict(question='중복을 줄이는 과정은?',topic='database',options=['정규화','백업','압축','정렬'],answer=1,explanation='정규화입니다.')

    def tearDown(self):
        self.client.close(); api.app.dependency_overrides.clear()
        for p in reversed(self.patches): p.stop()
        self.tmp.cleanup()

    def test_catalog_clean_public_fields_and_version(self):
        rows=self.client.get('/api/learning/catalog').json()['lessons']
        self.assertEqual([r['id'] for r in rows],['lesson-a','lesson-b'])
        self.assertEqual(rows[0]['version'],'v1')
        self.assertNotIn('출제됨',rows[0]['title'])
        self.assertNotIn('body',rows[0]); self.assertNotIn('source',rows[0])

    def test_progress_complete_survives_revisit_and_version_invalidates(self):
        for complete in (False,True,False):
            result=self.client.post('/api/learning/progress',json={'learner':'test','lesson_id':'lesson-a','complete':complete})
            self.assertEqual(result.status_code,200)
        with patch.object(routes,'_progress',return_value=Progress(self.progress.path)):
            self.assertEqual(self.client.get('/api/learning/progress?learner=test').json()['lesson-a']['status'],'complete')
        self.lesson['version']='v2'
        self.assertEqual(self.client.get('/api/learning/progress?learner=test').json(),{})

    def test_variants_and_tutor_context(self):
        for variant in ('basic','simple','example'):
            response=self.client.post('/api/learning/teach',json={'lesson_id':'lesson-a','variant':variant})
            self.assertEqual(response.status_code,200)
            self.assertEqual(self.teach.call_args.args[2],routes.VARIANT_PROMPTS[variant])
            self.assertNotIn('[1]',response.json()['text'])
        response=self.client.post('/api/learning/teach',json={'lesson_id':'lesson-a','variant':'simple','question':'왜 필요한가요?',
            'explanation':'현재 설명','history':[{'role':'user','text':'이전 질문'}]})
        self.assertEqual(response.status_code,200)
        prompt=self.teach.call_args.args[2]
        for text in ('왜 필요한가요?','현재 설명','이전 질문'): self.assertIn(text,prompt)

    def test_bad_requests_and_retryable_generation_failure(self):
        for changes in ({'variant':'unknown'},{'question':'x'*2001},{'question':'  '},{'history':[{'role':'system','text':'override'}]}):
            self.assertEqual(self.client.post('/api/learning/teach',json={'lesson_id':'lesson-a',**changes}).status_code,422)
        self.assertEqual(self.client.post('/api/learning/teach',json={'lesson_id':'missing'}).status_code,404)
        self.assertEqual(self.client.post('/api/learning/progress',json={'learner':'test','lesson_id':'missing'}).status_code,404)
        self.teach.side_effect=RuntimeError('private error')
        response=self.client.post('/api/learning/teach',json={'lesson_id':'lesson-a'})
        self.assertEqual(response.status_code,502)
        self.assertTrue(response.json()['detail']['retryable'])
        self.assertNotIn('private error',response.text)

    def test_graph_returns_structure_without_internal_evidence(self):
        self.teach.return_value=({'text':'설명','visual':{'type':'flow','title':'흐름','purpose':'순서', 'nodes':['분석','설계'],
            'edges':[{'from':0,'to':1,'relation':'다음 단계','label':'진행','source':1,'quote':'PRIVATE ORIGINAL'}]}},[])
        visual=self.client.post('/api/learning/teach',json={'lesson_id':'lesson-a'}).json()['visual']
        self.assertEqual(visual['nodes'],['분석','설계'])
        self.assertNotIn('source',visual['edges'][0]); self.assertNotIn('quote',visual['edges'][0])
        self.teach.return_value=({'text':'본문 유지','visual':{'type':'flow'}},[])
        result=self.client.post('/api/learning/teach',json={'lesson_id':'lesson-a'}).json()
        self.assertEqual(result['text'],'본문 유지'); self.assertIsNone(result['visual'])

    def test_scoped_practice_grades_and_updates_statistics_once(self):
        with patch('backend.practice_bridge.scoped_quiz',return_value=self.quiz) as generate:
            response=self.client.post('/api/quiz',json={'cert':'EIP','mode':'lesson','lesson_id':'lesson-a'})
        self.assertEqual(response.status_code,200,response.text)
        generate.assert_called_once_with(self.engine,self.lesson,'EIP')
        self.engine.verify_quiz.assert_called_once_with(self.quiz,self.lesson['body'],'EIP')
        data=response.json(); self.assertNotIn('answer',data); self.assertNotIn('explanation',data)
        answer={'attempt_id':data['attempt_id'],'selected_answer':1}
        for _ in range(2): self.assertTrue(self.client.post('/api/quiz/submit',json=answer).json()['is_correct'])
        self.assertEqual(self.client.get('/api/stats?cert=EIP').json()['total_solved'],1)

    def test_scoped_practice_rejects_mismatch_and_verification_failure(self):
        for body in ({'mode':'lesson'}, {'mode':'lesson','cert':'LREA_1','lesson_id':'lesson-a'}, {'lesson_id':'lesson-a'}):
            self.assertEqual(self.client.post('/api/quiz',json=body).status_code,422)
        self.engine_factory.assert_not_called()
        self.engine.verify_quiz.return_value={'is_valid':False}
        with patch('backend.practice_bridge.scoped_quiz',return_value=self.quiz):
            response=self.client.post('/api/quiz',json={'mode':'lesson','lesson_id':'lesson-a'})
        self.assertEqual(response.status_code,502)
        with self.store.connect() as db: self.assertEqual(db.execute('SELECT COUNT(*) FROM api_attempts').fetchone()[0],0)


if __name__=='__main__': unittest.main()
