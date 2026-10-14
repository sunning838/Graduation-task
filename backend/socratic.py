"""Persisted, revision-checked explanation practice; objective scores stay unchanged."""
import json
from pydantic import BaseModel, Field, StrictInt
from typing import Literal
from backend.api_store import StateError

SCHEMA = '''
CREATE TABLE IF NOT EXISTS socratic_sessions (
 id TEXT PRIMARY KEY, attempt_id TEXT NOT NULL, revision INTEGER NOT NULL DEFAULT 0,
 completed INTEGER NOT NULL DEFAULT 0, created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS socratic_turns (
 session_id TEXT NOT NULL, revision INTEGER NOT NULL, request_id TEXT NOT NULL,
 answer TEXT NOT NULL, evaluation TEXT NOT NULL, created_at TEXT DEFAULT CURRENT_TIMESTAMP,
 PRIMARY KEY(session_id, revision), UNIQUE(session_id, request_id)
);
'''

class Evaluation(BaseModel):
    status: Literal['continue', 'complete']
    question: str = Field(max_length=4000)
    acknowledgment: str = Field(min_length=1, max_length=4000)
    hint_level: StrictInt = Field(ge=0, le=3)

def snapshot(store, attempt_id, session_id):
    with store.connect() as db:
        attempt = db.execute('SELECT * FROM api_attempts WHERE id=?', (attempt_id,)).fetchone()
        if not attempt:
            raise StateError(404, 'ATTEMPT_NOT_FOUND', '문제를 찾을 수 없습니다.')
        if attempt['result'] is None:
            raise StateError(409, 'QUIZ_NOT_SUBMITTED', '객관식 답안을 먼저 제출해 주세요.')
        session = db.execute('SELECT * FROM socratic_sessions WHERE id=?', (session_id,)).fetchone()
        if session and session['attempt_id'] != attempt_id:
            raise StateError(409, 'SESSION_MISMATCH', '다른 문제의 문답 기록입니다.')
        rows = db.execute('SELECT * FROM socratic_turns WHERE session_id=? ORDER BY revision',
                          (session_id,)).fetchall()
    return dict(attempt), dict(session) if session else None, [dict(r) for r in rows]

def history(store, attempt_id, session_id):
    _, session, rows = snapshot(store, attempt_id, session_id)
    return {'session_id': session_id, 'revision': session['revision'] if session else 0,
            'completed': bool(session and session['completed']),
            'turns': [{'answer': r['answer'], **json.loads(r['evaluation'])} for r in rows]}

def evaluate(store, engine_factory, request):
    attempt, session, rows = snapshot(store, request.attempt_id, request.session_id)
    for row in rows:
        if row['request_id'] == request.request_id:
            if row['answer'] != request.answer or row['revision'] != request.revision + 1:
                raise StateError(409, 'REQUEST_CONFLICT', '제출 요청이 이전 답안과 다릅니다.')
            return json.loads(row['evaluation'])
    revision = session['revision'] if session else 0
    if revision != request.revision or (session and session['completed']):
        raise StateError(409, 'SESSION_CHANGED', '문답 상태가 변경되었습니다. 기록을 다시 불러와 주세요.')
    if revision >= 20:
        raise StateError(409, 'TURN_LIMIT', '문답이 길어졌습니다. 해설을 확인하고 처음부터 다시 설명해 주세요.')
    quiz = json.loads(attempt['quiz'])
    prior = [{'student': r['answer'], 'evaluation': json.loads(r['evaluation'])} for r in rows]
    result = engine_factory().socratic_dialogue_turn(
        topic=quiz['topic'], cert=attempt['cert'],
        quiz_question=json.dumps(quiz, ensure_ascii=False),
        history=json.dumps(prior, ensure_ascii=False), student_answer=request.answer,
        hint_level=min(revision, 3))
    evaluation = Evaluation.model_validate(result).model_dump()
    if evaluation['status'] == 'continue' and not evaluation['question'].strip():
        raise ValueError('Missing follow-up question')
    if not evaluation['acknowledgment'].strip():
        raise ValueError('Empty feedback')
    evaluation.update(session_id=request.session_id, revision=revision + 1)
    with store.connect() as db:
        db.execute('BEGIN IMMEDIATE')
        db.execute('INSERT OR IGNORE INTO socratic_sessions(id,attempt_id) VALUES(?,?)',
                   (request.session_id, request.attempt_id))
        existing = db.execute('SELECT * FROM socratic_turns WHERE session_id=? AND request_id=?',
                              (request.session_id, request.request_id)).fetchone()
        if existing:
            if existing['answer'] == request.answer and existing['revision'] == revision + 1:
                return json.loads(existing['evaluation'])
            raise StateError(409, 'REQUEST_CONFLICT', '이미 처리된 제출 요청입니다.')
        updated = db.execute('UPDATE socratic_sessions SET revision=revision+1, completed=? '
                             'WHERE id=? AND attempt_id=? AND revision=? AND completed=0',
                             (int(evaluation['status'] == 'complete'), request.session_id,
                              request.attempt_id, revision))
        if updated.rowcount != 1:
            raise StateError(409, 'SESSION_CHANGED', '다른 답안이 먼저 처리되었습니다. 기록을 다시 불러와 주세요.')
        db.execute('INSERT INTO socratic_turns(session_id,revision,request_id,answer,evaluation) VALUES(?,?,?,?,?)',
                   (request.session_id, revision + 1, request.request_id, request.answer,
                    json.dumps(evaluation, ensure_ascii=False)))
    return evaluation
