"""학습실(개념 학습) React용 API: 단원 목록, 진도, 강의 설명, AI 음성 강의."""
import logging
import re
from functools import lru_cache
from typing import Literal

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

LOG = logging.getLogger(__name__)
router = APIRouter(prefix='/api/learning')


def _fail(status, code, message, retryable=False):
    raise HTTPException(status, detail=dict(code=code, message=message, retryable=retryable))


def _catalog():
    from backend.learning import catalog
    return catalog()


@lru_cache
def _progress():
    from backend.learning import Progress
    return Progress()


def _lesson(lesson_id):
    _, lessons = _catalog()
    for lesson in lessons:
        if lesson['id'] == lesson_id:
            return lesson
    _fail(404, 'LESSON_NOT_FOUND', '학습 항목을 찾을 수 없습니다.')


def _states(learner):
    _, lessons = _catalog()
    raw = _progress().states(learner, lessons)
    return {key: {'status': 'complete' if status == 'complete' else 'started',
                  'updatedAt': int(touched) // 1_000_000}
            for key, (status, touched) in raw.items()}


class ProgressRequest(BaseModel):
    learner: str = Field(min_length=1, max_length=100)
    lesson_id: str = Field(min_length=1, max_length=100)
    complete: bool = False


class TutorMessage(BaseModel):
    role: Literal['user', 'assistant']
    text: str = Field(min_length=1, max_length=2000)


class TeachRequest(BaseModel):
    lesson_id: str = Field(min_length=1, max_length=100)
    prompt: str | None = Field(default=None, min_length=1, max_length=20000)
    variant: Literal['basic', 'simple', 'example'] = 'basic'
    question: str | None = Field(default=None, min_length=1, max_length=2000)
    explanation: str = Field(default='', max_length=12000)
    history: list[TutorMessage] = Field(default_factory=list, max_length=6)


VARIANT_PROMPTS = {
    'basic': '처음 배우는 학생에게 이 학습 항목의 첫 강의를 시작해 주세요.',
    'simple': '현재 단원의 개념 전체를 쉬운 용어와 짧은 단계로 다시 설명하세요. 낯선 용어를 먼저 풀고 원문의 정확한 조건은 유지하세요.',
    'example': '현재 단원을 하나의 구체적인 사례로 단계별 설명하세요. 사례의 각 부분과 개념을 연결하고 예시임을 표시하세요. 원문 조건을 유지하세요.',
}


class AudioRequest(BaseModel):
    lesson_id: str = Field(min_length=1, max_length=100)
    variant: str = Field(min_length=1, max_length=50)
    text: str = Field(min_length=1, max_length=50000)


@router.get('/catalog')
def get_catalog():
    from backend.lesson_display import display_title
    config, lessons = _catalog()
    return {
        'config': {key: {'label': value.get('label', key), 'topics': dict(value.get('topics', {}))}
                   for key, value in config.items()},
        'lessons': [dict(id=l['id'], cert=l['cert'], subject=l['subject'], version=l['version'], title=display_title(l['title']))
                    for l in lessons],
    }


@router.get('/progress')
def get_progress(learner: str):
    return _states(learner)


@router.post('/progress')
def save_progress(request: ProgressRequest):
    _progress().save(request.learner, _lesson(request.lesson_id), complete=request.complete)
    return _states(request.learner)


@router.post('/teach')
def teach_lesson(request: TeachRequest):
    from backend.learning import teach, diagram
    from backend.lesson_display import learner_text
    _, lessons = _catalog()
    lesson = _lesson(request.lesson_id)
    if request.question is not None:
        if not request.question.strip():
            _fail(422, 'EMPTY_QUESTION', '질문을 입력해 주세요.')
        history = '\n'.join(f'{m.role}: {m.text}' for m in request.history)
        prompt = (f'{request.question}\n[현재 선택한 설명 방식]\n{request.variant}'
                  f'\n[현재 보고 있는 강의: 참고용]\n{request.explanation}'
                  f'\n[이전 질문 대화: 참고용]\n{history}')
    else:
        prompt = request.prompt or VARIANT_PROMPTS[request.variant]
    try:
        data, _docs = teach(lesson, lessons, prompt)
    except Exception:
        LOG.exception('Lesson generation failed: %s', request.lesson_id)
        _fail(502, 'TEACH_FAILED', '설명을 준비하지 못했습니다. 다시 시도해 주세요.', True)
    try:
        visual = diagram(data)
        if visual and visual.get('kind') == 'graph':
            spec = data['visual']
            visual.update(nodes=spec['nodes'], edges=[{k: e[k] for k in ('from','to','relation','label')} for e in spec['edges']], layout=spec['type'])
    except (KeyError, TypeError, ValueError):
        LOG.warning('Cannot render lesson visual: %s', request.lesson_id)
        visual = None
    if not isinstance(visual, dict) or visual.get('kind') not in ('table', 'graph'):
        visual = None
    research = data.get('research') or None
    return {'text': learner_text(data.get('text', '')), 'visual': visual, 'research': research}


@router.post('/tts')
def create_audio_lecture(request: AudioRequest):
    from backend.tts_engine import generate_lesson_package
    lesson = _lesson(request.lesson_id)
    try:
        package = generate_lesson_package(lesson, request.variant, request.text)
    except Exception:
        LOG.exception('Audio lecture failed: %s', request.lesson_id)
        _fail(502, 'TTS_FAILED', 'AI 음성 강의를 만들지 못했습니다. 다시 시도해 주세요.', True)
    package_id = package['audio_path'].parent.name
    return {
        'audio_url': f'/api/learning/tts/{package_id}/audio',
        'timeline': [dict(index=item['index'], start=item['start'], end=item['end'], markdown=item['markdown'])
                     for item in package['timeline']],
    }


@router.get('/tts/{package_id}/audio')
def get_audio(package_id: str):
    from backend.tts_engine import PACKAGE_CACHE_DIR
    if not re.fullmatch(r'[0-9a-f]{64}', package_id):
        _fail(404, 'AUDIO_NOT_FOUND', '음성 파일을 찾을 수 없습니다.')
    path = PACKAGE_CACHE_DIR / package_id / 'lecture.wav'
    if not path.exists():
        _fail(404, 'AUDIO_NOT_FOUND', '음성 파일을 찾을 수 없습니다.')
    return FileResponse(path, media_type='audio/wav')
