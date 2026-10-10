"""학습실(개념 학습) React용 API: 단원 목록, 진도, 강의 설명, AI 음성 강의."""
import logging
import re
from functools import lru_cache

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

LOG = logging.getLogger(__name__)
router = APIRouter(prefix='/api/learning')


def _fail(status, code, message, retryable=False):
    raise HTTPException(status, detail=dict(code=code, message=message, retryable=retryable))


@lru_cache
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


class TeachRequest(BaseModel):
    lesson_id: str = Field(min_length=1, max_length=100)
    prompt: str = Field(min_length=1, max_length=20000)


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
        'lessons': [dict(id=l['id'], cert=l['cert'], subject=l['subject'], title=display_title(l['title']))
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
    try:
        data, _docs = teach(lesson, lessons, request.prompt)
    except Exception:
        LOG.exception('Lesson generation failed: %s', request.lesson_id)
        _fail(502, 'TEACH_FAILED', '설명을 준비하지 못했습니다. 다시 시도해 주세요.', True)
    visual = diagram(data)
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