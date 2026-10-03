"""Summary persistence and source-grounding tests, without live Gemini calls."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient
import api
from backend.api_store import APIStore, StateError
from backend import summary_notes


class SummaryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.store = APIStore(self.root / 'state.db')
        self.source_patch = patch.object(summary_notes, 'DATA_ROOT', self.root / 'data')
        self.source_patch.start()
        self.engine = MagicMock()
        self.engine.generate_final_note.return_value = '### 핵심 개념\n\n정규화의 정의입니다 [1].'
        self.factory = patch.object(api, 'get_engine', return_value=self.engine)
        self.factory.start()
        api.app.dependency_overrides[api.get_store] = lambda: self.store
        self.client = TestClient(api.app)

    def tearDown(self):
        self.client.close()
        api.app.dependency_overrides.clear()
        self.factory.stop()
        self.source_patch.stop()
        self.temp.cleanup()

    def source(self, cert='EIP', topic='database', text='## 데이터베이스\n정규화는 데이터 중복을 줄이는 과정입니다.'):
        path = self.root / 'data' / cert / topic / 'concept.md'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding='utf-8')
        return path

    def seed(self, cert='EIP', topic='database', total=5, correct=1):
        with self.store.connect() as db:
            db.executemany('INSERT INTO quiz_logs(cert,topic,is_correct) VALUES(?,?,?)',
                           [(cert, topic, int(i < correct)) for i in range(total)])

    def generate(self, cert='EIP'):
        return self.client.post('/api/summary-notes', json={'cert': cert})

    def view(self, cert='EIP'):
        return self.client.get('/api/summary-notes', params={'cert': cert}).json()

    def test_no_history_and_no_weakness_do_not_call_ai(self):
        self.assertFalse(self.view()['can_generate'])
        self.assertEqual(self.generate().status_code, 409)
        self.seed(correct=5)
        self.assertEqual(self.view()['weakness']['status'], 'no_weakness')
        self.assertEqual(self.generate().status_code, 409)
        self.engine.generate_final_note.assert_not_called()

    def test_generation_persists_and_hides_internal_evidence(self):
        self.seed()
        self.source()
        response = self.generate()
        self.assertEqual(response.status_code, 200, response.text)
        note = response.json()['notes'][0]
        self.assertNotIn('[1]', note['markdown'])
        self.assertNotIn('metadata', note)
        self.assertNotIn('source_hash', note)
        self.assertNotIn('contexts', note)
        self.store = APIStore(self.root / 'state.db')
        self.assertEqual(self.view()['notes'][0]['id'], note['id'])
        self.assertFalse(self.view()['needs_update'])
        self.generate()  # Fresh summaries are reused; no repeated AI cost.
        self.engine.generate_final_note.assert_called_once()
        internal = self.store.latest_summary_notes('EIP')['database']['metadata']
        self.assertTrue(internal['source_hash'])
        self.assertTrue(internal['sources'])

    def test_source_scope_and_legacy_record_without_problem_body(self):
        self.seed()
        self.source()
        self.source(topic='software_design', text='## 설계\n다른 과목 비밀 내용입니다.')
        self.source(cert='LREA_1', topic='database', text='## 계약\n다른 자격증 비밀 내용입니다.')
        response = self.generate()
        self.assertEqual(len(response.json()['notes']), 1)
        args, kwargs = self.engine.generate_final_note.call_args
        self.assertEqual(args, ('EIP', ['database']))
        self.assertIn('정규화', kwargs['contexts']['database'])
        self.assertNotIn('비밀', kwargs['contexts']['database'])
        self.assertEqual(self.view('LREA_1')['notes'], [])

    def test_empty_or_heading_only_material_does_not_generate(self):
        self.seed()
        self.source(text='# 과목\n\n## 제목만 있음\n')
        response = self.generate().json()
        self.assertEqual(response['notes'], [])
        self.assertEqual(response['issues'][0]['code'], 'NO_MATERIAL')
        self.engine.generate_final_note.assert_not_called()

    def test_record_and_source_changes_mark_stale(self):
        self.seed()
        path = self.source()
        original = self.generate().json()['notes'][0]
        self.seed(total=1, correct=0)
        self.assertTrue(self.view()['needs_update'])
        self.generate()
        self.assertFalse(self.view()['needs_update'])
        path.write_text('## 데이터베이스\n새 개념 원문 내용이 추가되었습니다.', encoding='utf-8')
        self.assertTrue(self.view()['needs_update'])
        newer = self.generate().json()['notes'][0]
        self.assertNotEqual(newer['id'], original['id'])
        with self.store.connect() as db:
            self.assertEqual(db.execute('SELECT COUNT(*) FROM summary_notes').fetchone()[0], 3)

    def test_failure_preserves_saved_note(self):
        self.seed()
        self.source()
        saved = self.generate().json()['notes'][0]
        self.seed(total=1, correct=0)
        self.engine.generate_final_note.side_effect = RuntimeError('private exception')
        response = self.generate()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()['notes'][0]['id'], saved['id'])
        self.assertTrue(response.json()['notes'][0]['stale'])
        self.assertEqual(response.json()['issues'][0]['code'], 'GENERATION_FAILED')
        self.assertNotIn('private exception', response.text)
        self.assertFalse(response.json()['generating'])

    def test_partial_success_retries_only_failed_subject(self):
        self.seed()
        self.seed(topic='software_design', correct=2)
        self.source()
        self.source(topic='software_design', text='## 설계\n소프트웨어 설계의 핵심 개념입니다.')
        def generate(cert, topics, contexts):
            if topics == ['software_design']:
                raise RuntimeError('failed')
            return '### 핵심 개념\n자료 기반 요약입니다.'
        self.engine.generate_final_note.side_effect = generate
        response = self.generate().json()
        self.assertEqual(len(response['notes']), 1)
        self.assertEqual(response['issues'][0]['topic'], 'software_design')
        self.engine.generate_final_note.reset_mock()
        self.engine.generate_final_note.side_effect = None
        result = self.generate().json()
        self.assertEqual(len(result['notes']), 2)
        self.engine.generate_final_note.assert_called_once()
        self.assertEqual(self.engine.generate_final_note.call_args.args[1], ['software_design'])

    def test_previous_note_survives_no_longer_weak(self):
        self.seed()
        self.source()
        saved = self.generate().json()['notes'][0]
        self.seed(total=10, correct=10)
        view = self.view()
        self.assertFalse(view['can_generate'])
        self.assertFalse(view['notes'][0]['is_current_topic'])
        self.assertEqual(view['notes'][0]['id'], saved['id'])

    def test_database_lock_and_fencing(self):
        self.seed()
        self.source()
        token = self.store.begin_summary_generation('EIP')
        other = APIStore(self.root / 'state.db')
        with self.assertRaises(StateError):
            other.begin_summary_generation('EIP')
        self.assertTrue(self.view()['generating'])
        self.assertEqual(self.generate().status_code, 409)
        with self.store.connect() as db:
            db.execute('UPDATE summary_generation SET expires=0')
        new_token = other.begin_summary_generation('EIP')
        with self.assertRaises(StateError):
            self.store.save_summary_note('EIP', token, 'database', 'stale', {}, 'now')
        self.store.finish_summary_generation('EIP', token, [])
        self.assertTrue(self.view()['generating'])
        other.finish_summary_generation('EIP', new_token, [])
        self.assertFalse(self.view()['generating'])

    def test_unreadable_sources_do_not_hide_saved_note(self):
        self.seed()
        path = self.source()
        self.generate()
        path.write_bytes(b'\xff')
        self.assertEqual(len(self.view()['notes']), 1)
        self.assertTrue(self.view()['needs_update'])

    def test_invalid_cert_and_empty_ai_response(self):
        self.assertEqual(self.generate('bad').status_code, 422)
        self.seed()
        self.source()
        self.engine.generate_final_note.return_value = ' '
        self.assertEqual(self.generate().json()['notes'], [])


class SummaryPromptTests(unittest.TestCase):
    def test_supplied_context_does_not_search_or_call_external_research(self):
        from backend.chat_engine import AITutorEngine
        from langchain_core.runnables import RunnableLambda
        engine = object.__new__(AITutorEngine)
        engine.vector_db = MagicMock()
        prompts = []
        engine.llm = RunnableLambda(lambda value: prompts.append(value.to_string()) or '### 핵심 개념\n요약')
        result = engine.generate_final_note('EIP', ['database'], contexts={'database': '제공된 개념 원문'})
        self.assertIn('요약', result)
        self.assertIn('제공된 개념 원문', prompts[0])
        self.assertIn('외부 지식으로 내용을 보강하지 않는다', prompts[0])
        engine.vector_db.similarity_search.assert_not_called()


if __name__ == '__main__':
    unittest.main()
