"""Presentation illustrations are confined to the learning room."""
from functools import lru_cache
from typing import Literal

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from backend.learning_routes import _catalog, _lesson
from backend.lecture_presentation import PresentationError, PresentationManager

router = APIRouter(prefix='/api/learning/presentations')


@lru_cache
def manager():
    return PresentationManager()


class PresentationRequest(BaseModel):
    lesson_id: str = Field(min_length=1, max_length=100)
    variant: Literal['basic', 'simple', 'example'] = 'basic'
    text: str = Field(min_length=1, max_length=50000)


def respond(action):
    try:
        return action()
    except PresentationError as error:
        raise HTTPException(error.status, detail=dict(code=error.code, message=str(error), retryable=error.status != 404)) from error


def _presentation_inputs(request):
    lesson = _lesson(request.lesson_id)
    if lesson['cert'] != 'EIP':
        raise HTTPException(422, detail=dict(code='UNSUPPORTED_LESSON', message='정보처리기사 강의를 선택해 주세요.'))
    _, lessons = _catalog()
    return lesson, lessons, request.variant, request.text


@router.post('/lookup')
def lookup_presentation(request: PresentationRequest):
    # The request body carries the full lecture text for an exact cache match.
    inputs = _presentation_inputs(request)
    return respond(lambda: manager().lookup(*inputs))


@router.post('')
def create_presentation(request: PresentationRequest):
    inputs = _presentation_inputs(request)
    return respond(lambda: manager().start(*inputs))


@router.get('/{package_id}')
def get_presentation(package_id: str):
    return respond(lambda: manager().get(package_id))


@router.post('/{package_id}/retry')
def retry_presentation(package_id: str):
    _, lessons = _catalog()
    return respond(lambda: manager().retry(package_id, lessons))


@router.get('/{package_id}/images/{slide_id}')
def get_image(package_id: str, slide_id: str):
    path = respond(lambda: manager().image_path(package_id, slide_id))
    return FileResponse(path, media_type='image/png', headers={'Cache-Control': 'private, max-age=86400'})
