import json
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pymupdf
from fastapi.testclient import TestClient

import api
from backend import pdf_summary as pdf_tutor
from backend.pdf_summary import MAX_TEXT_LENGTH, NO_PDF_ANSWER, RELATED_WITHOUT_DETAIL


def make_pdf(*pages):
    with pymupdf.open() as document:
        for content in pages:
            page = document.new_page()
            if content:
                page.insert_text((72, 72), content)
        return document.tobytes()


class PdfSummaryTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(api.app)
        self.addCleanup(self.client.close)

    def test_summary_returns_numbered_source_pages_for_tutor(self):
        engine = Mock()
        engine.llm.invoke.return_value = SimpleNamespace(content='요약입니다.')
        with patch.object(api, 'get_engine', return_value=engine):
            response = self.client.post('/api/pdf-summary', files={
                'file': ('notes.pdf', make_pdf('First topic', '', 'Second topic'), 'application/pdf'),
            })
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()['source_pages'], [
            {'page': 1, 'text': 'First topic'},
            {'page': 3, 'text': 'Second topic'},
        ])
        self.assertEqual(response.json()['summary'], '요약입니다.')

    def test_blank_document_does_not_call_model(self):
        with patch.object(api, 'get_engine') as engine:
            response = self.client.post('/api/pdf-summary', files={
                'file': ('blank.pdf', make_pdf(''), 'application/pdf'),
            })
        self.assertEqual(response.status_code, 422)
        engine.assert_not_called()

    def test_tutor_reads_current_pdf_passages_without_general_rag(self):
        model = Mock()

        def respond(prompt, **_kwargs):
            payload = json.loads(prompt.rsplit('\n', 1)[1])
            source = payload['passages'][0]['text']
            return SimpleNamespace(content=json.dumps({
                'status': 'answered', 'answer': source, 'pdf_ids': [1],
            }))

        model.invoke.side_effect = respond
        with patch.object(api, 'pdf_tutor_model', return_value=model), patch.object(api, 'get_engine') as general:
            for policy in ('The storage period is twelve days.', 'The storage period is ninety days.'):
                result = self.client.post('/api/pdf-summary/question', json={
                    'question': '보관 기간은 얼마인가요?',
                    'source_pages': [{'page': 1, 'text': policy}],
                    'history': [{'role': 'assistant', 'text': '이전 답변은 문서 근거가 아닙니다.'}],
                })
                self.assertEqual(result.json(), {'answer': policy})
            general.assert_not_called()

    def test_tutor_rejects_invalid_evidence(self):
        model = Mock()
        model.invoke.return_value = SimpleNamespace(content=json.dumps({
            'status': 'answered', 'answer': '추측한 답변', 'pdf_ids': [1, 99],
        }))
        with patch.object(api, 'pdf_tutor_model', return_value=model):
            result = self.client.post('/api/pdf-summary/question', json={
                'question': '이 자료에 답변이 있나요?',
                'source_pages': [{'page': 1, 'text': '별개의 내용만 포함합니다.'}],
            })
        self.assertEqual(result.json(), {'answer': NO_PDF_ANSWER})

    def test_long_context_contains_matching_passage_with_a_bounded_prompt(self):
        model = Mock()
        model.invoke.return_value = SimpleNamespace(content='{"status": "unrelated"}')
        source = [{'page': i, 'text': '일반적인 설명입니다. ' * 400} for i in range(1, 8)]
        source.append({'page': 8, 'text': '프로젝트의 백업 주기는 삼십 일입니다.'})
        with patch.object(api, 'pdf_tutor_model', return_value=model):
            result = self.client.post('/api/pdf-summary/question', json={
                'question': '백업 주기는 얼마인가요?', 'source_pages': source,
            })
        self.assertEqual(result.json(), {'answer': NO_PDF_ANSWER})
        prompt = model.invoke.call_args.args[0]
        self.assertIn('프로젝트의 백업 주기는 삼십 일입니다.', prompt)
        self.assertLess(len(prompt), 12000)

    def test_machine_learning_question_combines_multiple_table_entries_without_search(self):
        request = {
            'question': '머신러닝이 뭐야?',
            'summary': '머신러닝 학습 유형의 핵심 개념 요약',
            'source_pages': [{'page': 1, 'text': '\n'.join([
                'AI 응용모델설계', '시험 직전 핵심 정리', '1', '머신러닝 기초', '지도학습',
                '정답 라벨이 있는 데이터로 학습', '비지도학습', '정답 없이 군집·패턴을 발견',
                '강화학습', '보상을 최대화하도록 행동 학습',
            ])}],
        }
        model = Mock()
        answer = '이 자료는 머신러닝을 지도학습, 비지도학습, 강화학습으로 나눠 설명합니다.'

        def combine(prompt, **_kwargs):
            payload = json.loads(prompt.rsplit('\n', 1)[1])
            self.assertEqual(payload['summary'], request['summary'])
            self.assertIn('정답 라벨이 있는 데이터로 학습', payload['passages'][0]['text'])
            self.assertIn('보상을 최대화하도록 행동 학습', payload['passages'][1]['text'])
            return SimpleNamespace(content=json.dumps({
                'status': 'answered', 'answer': answer, 'pdf_ids': [1, 2],
            }))

        model.invoke.side_effect = combine
        with patch.object(api, 'pdf_tutor_model', return_value=model), patch.object(pdf_tutor, 'search_google') as search:
            response = self.client.post('/api/pdf-summary/question', json=request)
        self.assertEqual(response.json(), {'answer': answer})
        search.assert_not_called()

    def test_related_gap_searches_only_concept_and_displays_used_authoritative_source(self):
        source_url = 'https://scikit-learn.org/stable/modules/linear_model.html'
        model = Mock()
        model.invoke.side_effect = [
            SimpleNamespace(content=json.dumps({
                'status': 'needs_external', 'answer': 'PDF는 선형 회귀를 예로 듭니다.',
                'pdf_ids': [1], 'search_concept': '선형 회귀 원리',
            })),
            SimpleNamespace(content=json.dumps({
                'supplement': '입력 특성의 선형 결합으로 값을 예측합니다.', 'source_ids': [1],
            })),
        ]
        search_result = [
            {'title': '신뢰할 수 없는 결과', 'source': 'https://unknown.example/linear', 'body': '사용하지 않는 근거'},
            {'title': 'Linear models', 'source': source_url, 'body': '선형 결합으로 예측합니다.'},
        ]
        with patch.object(api, 'pdf_tutor_model', return_value=model), patch.object(
                pdf_tutor, 'search_google', return_value=(search_result, '<span>검색 참고</span>')) as search:
            response = self.client.post('/api/pdf-summary/question', json={
                'question': '자료에서 선형 회귀의 원리는 어떻게 설명하나요?',
                'source_pages': [{'page': 1, 'text': '작성자 정보\nPDF의 예시: 선형 회귀'}],
            })
        search.assert_called_once_with(['선형 회귀 원리'], pdf_tutor.PDF_RESEARCH_POLICY)
        answer = response.json()
        self.assertIn('PDF는 선형 회귀를 예로 듭니다.', answer['answer'])
        self.assertIn('**보충 설명**', answer['answer'])
        self.assertEqual(answer['research']['sources'], [{'title': 'Linear models', 'url': source_url}])
        self.assertNotIn('사용하지 않는 근거', model.invoke.call_args.args[0])

    def test_search_failure_preserves_available_pdf_answer(self):
        model = Mock()
        model.invoke.return_value = SimpleNamespace(content=json.dumps({
            'status': 'needs_external', 'answer': 'PDF에서는 회귀가 예시로 나옵니다.',
            'pdf_ids': [1], 'search_concept': '선형 회귀',
        }))
        with patch.object(api, 'pdf_tutor_model', return_value=model), patch.object(
                pdf_tutor, 'search_google', side_effect=TimeoutError):
            response = self.client.post('/api/pdf-summary/question', json={
                'question': '회귀는 어떻게 작동하나요?', 'source_pages': [{'page': 1, 'text': '예시: 회귀'}],
            })
        self.assertEqual(response.json(), {'answer': 'PDF에서는 회귀가 예시로 나옵니다.\n\n' + RELATED_WITHOUT_DETAIL})
        model.invoke.assert_called_once()

    def test_unrelated_or_unverified_topics_do_not_search(self):
        cases = [
            ({'status': 'unrelated', 'pdf_ids': [1], 'search_concept': '주식 종목'}, NO_PDF_ANSWER),
            ({'status': 'needs_external', 'pdf_ids': [5], 'search_concept': '모델 학습'}, NO_PDF_ANSWER),
            ({'status': 'needs_external', 'pdf_ids': [1], 'search_concept': '사용자 개인자료.pdf'}, RELATED_WITHOUT_DETAIL),
        ]
        for result, expected in cases:
            with self.subTest(result=result), patch.object(api, 'pdf_tutor_model') as factory, patch.object(
                    pdf_tutor, 'search_google') as search:
                factory.return_value.invoke.return_value = SimpleNamespace(content=json.dumps(result))
                response = self.client.post('/api/pdf-summary/question', json={
                    'question': '이 질문에 답해 주세요.', 'source_pages': [{'page': 1, 'text': '학습 모델'}],
                })
                self.assertEqual(response.json(), {'answer': expected})
                search.assert_not_called()

    def test_source_size_is_validated_before_inference(self):
        with patch.object(api, 'pdf_tutor_model') as model:
            result = self.client.post('/api/pdf-summary/question', json={
                'question': '내용을 알려 주세요.',
                'source_pages': [{'page': i, 'text': 'x' * (MAX_TEXT_LENGTH // 2)} for i in (1, 2, 3)],
            })
        self.assertEqual(result.status_code, 422)
        model.assert_not_called()
