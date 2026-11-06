"""Scoped, replay-safe reset of learner records; shared teaching assets are retained."""
import json

from backend.api_store import StateError


def reset_learning(store, cert, learner, include_progress, request_id, progress=None, lessons=()):
    # ATTACH keeps optional concept progress and practice records in one commit.
    # Both application databases use SQLite's default rollback journal.
    with store.connect() as db:
        if include_progress:
            if progress is None:
                raise ValueError('Progress storage is required')
            db.execute('ATTACH DATABASE ? AS concepts', (str(progress.path),))
        db.execute('BEGIN IMMEDIATE')
        previous = db.execute('SELECT * FROM learning_reset_requests WHERE request_id=?', (request_id,)).fetchone()
        if previous:
            if (previous['cert'], previous['learner'], bool(previous['include_progress'])) != (cert, learner, include_progress):
                raise StateError(409, 'REQUEST_CONFLICT', '이전에 사용한 초기화 요청과 범위가 다릅니다.')
            return json.loads(previous['result'])

        db.execute('INSERT INTO learning_reset_versions(cert,generation) VALUES(?,1) '
                   'ON CONFLICT(cert) DO UPDATE SET generation=generation+1', (cert,))
        deleted = {}
        statements = {
            'socratic_turns': '''DELETE FROM socratic_turns WHERE session_id IN
                (SELECT s.id FROM socratic_sessions s JOIN api_attempts a ON a.id=s.attempt_id WHERE a.cert=?)''',
            'socratic_sessions': '''DELETE FROM socratic_sessions WHERE attempt_id IN
                (SELECT id FROM api_attempts WHERE cert=?)''',
            'attempts': 'DELETE FROM api_attempts WHERE cert=?',
            'exam_items': 'DELETE FROM mock_exam_items WHERE exam_id IN (SELECT id FROM mock_exams WHERE cert=?)',
            'exams': 'DELETE FROM mock_exams WHERE cert=?',
            'quiz_logs': 'DELETE FROM quiz_logs WHERE cert=?',
            'summary_notes': 'DELETE FROM summary_notes WHERE cert=?',
            'summary_generation': 'DELETE FROM summary_generation WHERE cert=?',
        }
        for name, sql in statements.items():
            deleted[name] = db.execute(sql, (cert,)).rowcount
        # Keep only cancelled request IDs so retries from an old tab cannot
        # recreate a discarded preparation/exam. Remove its learning payload.
        deleted['preparations'] = db.execute("""UPDATE mock_preparations SET status='cancelled',
            distribution='[]', exam_id=NULL, attempts=0, max_attempts=0, tried='[]', last_error=NULL,
            updated_at=CURRENT_TIMESTAMP WHERE cert=? AND status!='cancelled'""", (cert,)).rowcount
        deleted['progress'] = 0
        if include_progress:
            db.execute('INSERT INTO concepts.progress_reset_versions(learner,cert,revision) VALUES(?,?,1) '
                       'ON CONFLICT(learner,cert) DO UPDATE SET revision=revision+1', (learner, cert))
            ids = [(learner, lesson['id']) for lesson in lessons if lesson['cert'] == cert]
            deleted['progress'] = db.executemany('DELETE FROM concepts.progress WHERE learner=? AND lesson=?', ids).rowcount
        result = dict(cert=cert, learner=learner, include_progress=include_progress, deleted=deleted,
                      message='학습 기록을 초기화했습니다.')
        db.execute('INSERT INTO learning_reset_requests(request_id,cert,learner,include_progress,result) VALUES(?,?,?,?,?)',
                   (request_id, cert, learner, int(include_progress), json.dumps(result, ensure_ascii=False)))
    return result
