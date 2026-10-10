"""Local React API: persistent conversations and quiz attempts."""
import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Literal

from fastapi import Depends, FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel, Field, StringConstraints

from backend.api_store import APIStore, StateError
from backend.cert_config import CERT_CONFIG
from backend.db_manager import DB_PATH
from backend.learning_stats import learning_stats
from backend.summary_notes import summary_view, generate_summaries
from backend import mock_exams
from backend import mock_preparation

LOG = logging.getLogger(__name__)
app = FastAPI(title='자격증 AI 튜터 API')
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv('FRONTEND_ORIGINS', 'http://localhost:5173,http://127.0.0.1:5173').split(','),
    allow_credentials=True, allow_methods=['*'], allow_headers=['*'],
)


@lru_cache
def get_store():
    Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    return APIStore(DB_PATH)


@lru_cache
def get_engine():
    # No model download or API-key requirement when merely importing this module.
    from backend.chat_engine import AITutorEngine
    return AITutorEngine()


def fail(status, code, message, retryable=False):
    raise HTTPException(status, detail=dict(code=code, message=message, retryable=retryable))


@app.exception_handler(StateError)
async def state_error_handler(_request, exc):
    return JSONResponse(status_code=exc.status, content={'detail': {
        'code': exc.code, 'message': exc.message, 'retryable': False}})


@app.exception_handler(RequestValidationError)
async def validation_error_handler(_request, _exc):
    return JSONResponse(status_code=422, content={'detail': {
        'code': 'INVALID_REQUEST', 'message': '입력한 자격증, 질문 또는 답안 형식을 확인해 주세요.',
        'retryable': False}})


@app.exception_handler(Exception)
async def unexpected_error_handler(_request, exc):
    LOG.error('API request failed', exc_info=(type(exc), exc, exc.__traceback__))
    return JSONResponse(status_code=500, content={'detail': {
        'code': 'INTERNAL_ERROR', 'message': '요청을 처리하지 못했습니다. 잠시 후 다시 시도해 주세요.',
        'retryable': True}})


Text = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=8000)]
Identifier = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=100)]


class ChatRequest(BaseModel):
    message: Text
    cert: Identifier = 'EIP'
    answer_length: Literal['short', 'medium', 'long'] = 'medium'
    conversation_id: Identifier | None = None


class QuizRequest(BaseModel):
    cert: Identifier = 'EIP'
    mode: Literal['random', 'weakness'] = 'random'


class SummaryRequest(BaseModel):
    cert: Identifier = 'EIP'


class ExamPart(BaseModel):
    topic: Identifier
    count: int = Field(strict=True, ge=1, le=100)


class ExamRequest(BaseModel):
    cert: Identifier
    distribution: list[ExamPart] = Field(min_length=1, max_length=100)
    request_id: Identifier


class ExamAnswer(BaseModel):
    selected_answer: int | None = Field(default=None, strict=True, ge=1)


class ExamSubmit(BaseModel):
    confirm_unanswered: bool = False


class QuizSubmitRequest(BaseModel):
    attempt_id: Identifier | None = None
    quiz_id: Identifier | None = None  # Compatibility with the previous client.
    selected_answer: int = Field(strict=True, ge=1)


def certification(cert):
    if cert not in CERT_CONFIG:
        fail(422, 'INVALID_CERT', '지원하지 않는 자격증입니다.')
    return CERT_CONFIG[cert]


@app.get('/')
def root():
    return {'message': 'AI Tutor API 서버가 정상 실행 중입니다.'}


@app.get('/api/certifications')
def certifications():
    return {'certifications': [dict(id=key, label=value['label'],
        option_count=value.get('option_count', 4),
        topics=[dict(id=k, label=v) for k, v in value['topics'].items()])
        for key, value in CERT_CONFIG.items()]}


@app.post('/api/chat')
def chat(request: ChatRequest, store: APIStore = Depends(get_store)):
    certification(request.cert)
    conversation_id, revision, rows = store.conversation(request.conversation_id, request.cert)
    history = [(HumanMessage if row['role'] == 'user' else AIMessage)(content=row['content'])
               for row in rows]
    lengths = {'short': '핵심만 3~5문장으로 간결하게 설명하세요.',
               'medium': '핵심 개념과 필요한 예시를 중심으로 이해하기 쉽게 설명하세요.',
               'long': '이유, 특징, 예시, 비교를 활용해 자세히 설명하세요.'}
    try:
        answer = get_engine().generate_response(
            query=f'{request.message}\n\n[답변 길이 지침]\n{lengths[request.answer_length]}',
            chat_history=history, student_status='분석된 상태 없음', cert=request.cert)
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError('Empty AI response')
    except Exception:
        LOG.exception('Chat generation failed')
        fail(502, 'CHAT_GENERATION_FAILED', '답변을 준비하지 못했습니다. 다시 시도해 주세요.', True)
    store.append_turn(conversation_id, revision, request.message, answer)
    return {'conversation_id': conversation_id, 'answer': answer}


@app.get('/api/stats')
def stats(cert: str = 'EIP', store: APIStore = Depends(get_store)):
    certification(cert)
    return learning_stats(store, cert)


@app.get('/api/weakness')
def weakness(cert: str = 'EIP', store: APIStore = Depends(get_store)):
    certification(cert)
    return {'cert': cert, **learning_stats(store, cert)['weakness']}


@app.get('/api/summary-notes')
def get_summary_notes(cert: str = 'EIP', store: APIStore = Depends(get_store)):
    certification(cert)
    return summary_view(store, cert)


@app.get('/api/question-bank/availability')
def bank_availability(cert: str = 'EIP', store: APIStore = Depends(get_store)):
    certification(cert)
    return mock_exams.availability(store, cert)


@app.post('/api/mock-exams')
def new_exam(request: ExamRequest, store: APIStore = Depends(get_store)):
    certification(request.cert)
    return mock_exams.create_exam(store, request.cert, [p.model_dump() for p in request.distribution], request.request_id)


@app.post('/api/mock-preparations')
def prepare_exam(request: ExamRequest, store: APIStore = Depends(get_store)):
    certification(request.cert)
    result = mock_preparation.start(store,request.cert,[p.model_dump() for p in request.distribution],request.request_id)
    if result['status'] in ('queued','running'):
        mock_preparation.dispatch(store,request.request_id,get_engine)
    return result


@app.get('/api/mock-preparations/{request_id}')
def preparation_status(request_id: str, store: APIStore = Depends(get_store)):
    result = mock_preparation.view(store,request_id)
    if result['status'] in ('queued','running'):
        mock_preparation.dispatch(store,request_id,get_engine)
    return result


@app.get('/api/mock-exams/{exam_id}')
def get_exam(exam_id: str, store: APIStore = Depends(get_store)):
    return mock_exams.exam_view(store, exam_id)


@app.put('/api/mock-exams/{exam_id}/answers/{item_id}')
def put_exam_answer(exam_id: str, item_id: str, request: ExamAnswer, store: APIStore = Depends(get_store)):
    return mock_exams.save_answer(store, exam_id, item_id, request.selected_answer)


@app.post('/api/mock-exams/{exam_id}/submit')
def finish_exam(exam_id: str, request: ExamSubmit, store: APIStore = Depends(get_store)):
    return mock_exams.submit_exam(store, exam_id, request.confirm_unanswered)


@app.post('/api/summary-notes')
def create_summary_notes(request: SummaryRequest, store: APIStore = Depends(get_store)):
    certification(request.cert)
    return generate_summaries(store, request.cert, get_engine)


def validate_quiz(quiz, config):
    if not isinstance(quiz, dict) or quiz.get('is_fallback') or quiz.get('validation_failed'):
        raise ValueError('Quiz generation did not succeed')
    options = quiz.get('options')
    if (not isinstance(options, list) or len(options) != config.get('option_count', 4)
            or any(not isinstance(o, str) or not o.strip() for o in options)):
        raise ValueError('Invalid options')
    answer = quiz.get('answer')
    if type(answer) is not int or not 1 <= answer <= len(options):
        raise ValueError('Invalid answer')
    for key in ('question', 'explanation'):
        if not isinstance(quiz.get(key), str) or not quiz[key].strip():
            raise ValueError(f'Missing {key}')
    if quiz.get('topic') not in config['topics']:
        raise ValueError('Invalid topic')


@app.post('/api/quiz')
def create_quiz(request: QuizRequest, store: APIStore = Depends(get_store)):
    config = certification(request.cert)
    focus = None
    if request.mode == 'weakness':
        analysis = learning_stats(store, request.cert)['weakness']
        focus = analysis['focus']
        if focus is None:
            code = 'INSUFFICIENT_HISTORY' if analysis['status'] == 'insufficient_data' else 'NO_WEAK_TOPIC'
            fail(409, code, analysis['message'])
    try:
        if focus:
            quiz = get_engine().generate_advanced_quiz(
                cert=request.cert, target_topic=focus['topic'], strict_subject=True)
            if quiz.get('failure_code') == 'NO_SUBJECT_MATERIAL':
                fail(422, 'NO_SUBJECT_MATERIAL', '해당 과목의 학습 자료가 부족해 문제를 생성할 수 없습니다.')
            if quiz.get('topic') != focus['topic']:
                raise ValueError('Generated quiz does not match the requested subject')
        else:
            quiz = get_engine().generate_advanced_quiz(cert=request.cert)
        validate_quiz(quiz, config)
    except HTTPException:
        raise
    except Exception:
        LOG.exception('Quiz generation failed')
        fail(502, 'QUIZ_GENERATION_FAILED', '문제를 생성하지 못했습니다. 다시 시도해 주세요.', True)
    attempt_id = store.create_attempt(request.cert, quiz)
    return dict(attempt_id=attempt_id, quiz_id=attempt_id, cert=request.cert, mode=request.mode,
                question=quiz['question'], topic=quiz['topic'],
                topic_label=config['topics'][quiz['topic']], options=quiz['options'],
                code_block=quiz.get('code_block'), table_data=quiz.get('table_data'))


@app.post('/api/quiz/submit')
def submit_quiz(request: QuizSubmitRequest, store: APIStore = Depends(get_store)):
    if not (request.attempt_id or request.quiz_id):
        fail(422, 'INVALID_REQUEST', '풀이 ID가 필요합니다.')
    if request.attempt_id and request.quiz_id and request.attempt_id != request.quiz_id:
        fail(422, 'INVALID_REQUEST', '풀이 ID가 일치하지 않습니다.')
    return store.submit(request.attempt_id or request.quiz_id, request.selected_answer)

from backend.learning_routes import router as learning_router
app.include_router(learning_router)