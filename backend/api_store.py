"""Persistent local API state, with atomic writes to the existing quiz log."""
import json
import sqlite3
import time
from contextlib import contextmanager
from uuid import uuid4


class StateError(Exception):
    def __init__(self, status, code, message):
        self.status, self.code, self.message = status, code, message


class APIStore:
    def __init__(self, path):
        self.path = str(path)
        from backend.mock_exams import SCHEMA as MOCK_SCHEMA
        with self.connect() as db:
            db.executescript(MOCK_SCHEMA)
            db.executescript("""
                CREATE TABLE IF NOT EXISTS quiz_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, cert TEXT, topic TEXT,
                    is_correct INTEGER, essay_score INTEGER DEFAULT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                );
                CREATE TABLE IF NOT EXISTS api_attempts (
                    id TEXT PRIMARY KEY, cert TEXT NOT NULL, quiz TEXT NOT NULL,
                    selected_answer INTEGER, result TEXT, log_id INTEGER,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP, submitted_at TEXT
                );
                CREATE TABLE IF NOT EXISTS api_conversations (
                    id TEXT PRIMARY KEY, cert TEXT NOT NULL,
                    revision INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                );
                CREATE TABLE IF NOT EXISTS api_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id TEXT NOT NULL, role TEXT NOT NULL, content TEXT NOT NULL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                );
                CREATE INDEX IF NOT EXISTS api_messages_conversation
                    ON api_messages(conversation_id, id);
                CREATE TABLE IF NOT EXISTS summary_notes (
                    id TEXT PRIMARY KEY, cert TEXT NOT NULL, topic TEXT NOT NULL,
                    markdown TEXT NOT NULL, metadata TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS summary_notes_lookup ON summary_notes(cert, topic);
                CREATE TABLE IF NOT EXISTS summary_generation (
                    cert TEXT PRIMARY KEY, token TEXT, expires REAL NOT NULL DEFAULT 0,
                    issues TEXT NOT NULL DEFAULT '[]'
                );
            """)

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=15)
        db.row_factory = sqlite3.Row
        try:
            with db:
                yield db
        finally:
            db.close()

    def conversation(self, conversation_id, cert):
        with self.connect() as db:
            if conversation_id is None:
                conversation_id = str(uuid4())
                db.execute('INSERT INTO api_conversations(id, cert) VALUES (?, ?)',
                           (conversation_id, cert))
            row = db.execute('SELECT * FROM api_conversations WHERE id=?',
                             (conversation_id,)).fetchone()
            if row is None:
                raise StateError(404, 'CONVERSATION_NOT_FOUND', '대화를 찾을 수 없습니다. 새 대화를 시작해 주세요.')
            if row['cert'] != cert:
                raise StateError(409, 'CONVERSATION_CERT_MISMATCH', '자격증이 변경되었습니다. 새 대화를 시작해 주세요.')
            rows = db.execute('SELECT role, content FROM api_messages WHERE conversation_id=? '
                              'ORDER BY id DESC LIMIT 12', (conversation_id,)).fetchall()
            return conversation_id, row['revision'], [dict(r) for r in reversed(rows)]

    def append_turn(self, conversation_id, revision, message, answer):
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            cursor = db.execute('UPDATE api_conversations SET revision=revision+1 '
                                'WHERE id=? AND revision=?', (conversation_id, revision))
            if cursor.rowcount != 1:
                raise StateError(409, 'CONVERSATION_CHANGED', '다른 질문이 먼저 처리되었습니다. 다시 질문해 주세요.')
            db.executemany('INSERT INTO api_messages(conversation_id, role, content) VALUES (?, ?, ?)',
                           [(conversation_id, 'user', message), (conversation_id, 'assistant', answer)])

    def create_attempt(self, cert, quiz):
        attempt_id = str(uuid4())
        with self.connect() as db:
            db.execute('INSERT INTO api_attempts(id, cert, quiz) VALUES (?, ?, ?)',
                       (attempt_id, cert, json.dumps(quiz, ensure_ascii=False)))
        return attempt_id

    def quiz_statistics(self, cert, day_start, day_end):
        """Read one consistent aggregate; SQLite timestamps are stored in UTC."""
        with self.connect() as db:
            rows = db.execute('''
                SELECT topic, COUNT(*) AS total,
                    SUM(CASE WHEN is_correct=1 THEN 1 ELSE 0 END) AS correct,
                    SUM(CASE WHEN timestamp>=? AND timestamp<? THEN 1 ELSE 0 END) AS today
                FROM quiz_logs WHERE cert=? GROUP BY topic
            ''', (day_start, day_end, cert)).fetchall()
            return [dict(row) for row in rows]

    def latest_summary_notes(self, cert):
        with self.connect() as db:
            rows = db.execute('SELECT * FROM summary_notes WHERE cert=? ORDER BY rowid DESC', (cert,)).fetchall()
        latest = {}
        for row in rows:
            if row['topic'] not in latest:
                latest[row['topic']] = {**dict(row), 'metadata': json.loads(row['metadata'])}
        return latest

    def summary_generation_state(self, cert):
        with self.connect() as db:
            row = db.execute('SELECT * FROM summary_generation WHERE cert=?', (cert,)).fetchone()
        return {'generating': bool(row and row['token'] and row['expires'] > time.time()),
                'issues': json.loads(row['issues']) if row else []}

    def begin_summary_generation(self, cert):
        token = str(uuid4())
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            row = db.execute('SELECT token, expires FROM summary_generation WHERE cert=?', (cert,)).fetchone()
            if row and row['token'] and row['expires'] > time.time():
                raise StateError(409, 'SUMMARY_BUSY', '요약을 생성하고 있습니다. 잠시 후 다시 조회해 주세요.')
            db.execute('INSERT INTO summary_generation(cert,token,expires) VALUES(?,?,?) '
                       'ON CONFLICT(cert) DO UPDATE SET token=excluded.token, expires=excluded.expires',
                       (cert, token, time.time() + 900))
        return token

    def save_summary_note(self, cert, token, topic, markdown, metadata, created_at):
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            row = db.execute('SELECT token FROM summary_generation WHERE cert=?', (cert,)).fetchone()
            if not row or row['token'] != token:
                raise StateError(409, 'SUMMARY_SUPERSEDED', '새 요약 생성 요청이 시작되었습니다. 다시 조회해 주세요.')
            db.execute('INSERT INTO summary_notes(id,cert,topic,markdown,metadata,created_at) VALUES(?,?,?,?,?,?)',
                       (str(uuid4()), cert, topic, markdown, json.dumps(metadata, ensure_ascii=False), created_at))
            db.execute('UPDATE summary_generation SET expires=? WHERE cert=? AND token=?',
                       (time.time() + 900, cert, token))

    def finish_summary_generation(self, cert, token, issues):
        with self.connect() as db:
            db.execute('UPDATE summary_generation SET token=NULL, expires=0, issues=? WHERE cert=? AND token=?',
                       (json.dumps(issues, ensure_ascii=False), cert, token))

    def submit(self, attempt_id, selected_answer):
        with self.connect() as db:
            # One write transaction prevents duplicate logs across concurrent processes.
            db.execute('BEGIN IMMEDIATE')
            row = db.execute('SELECT * FROM api_attempts WHERE id=?', (attempt_id,)).fetchone()
            if row is None:
                raise StateError(404, 'ATTEMPT_NOT_FOUND', '문제를 찾을 수 없습니다. 새 문제를 불러와 주세요.')
            quiz = json.loads(row['quiz'])
            if not 1 <= selected_answer <= len(quiz['options']):
                raise StateError(422, 'INVALID_OPTION', '문제에 있는 보기 번호를 선택해 주세요.')
            if row['result'] is not None:
                if selected_answer != row['selected_answer']:
                    raise StateError(409, 'ALREADY_SUBMITTED', '이미 제출한 답안은 변경할 수 없습니다.')
                return json.loads(row['result'])
            result = dict(attempt_id=attempt_id, is_correct=selected_answer == quiz['answer'],
                          selected_answer=selected_answer, correct_answer=quiz['answer'],
                          explanation=quiz['explanation'])
            cursor = db.execute('INSERT INTO quiz_logs(cert, topic, is_correct) VALUES (?, ?, ?)',
                                (row['cert'], quiz['topic'], int(result['is_correct'])))
            db.execute('UPDATE api_attempts SET selected_answer=?, result=?, log_id=?, '
                       'submitted_at=CURRENT_TIMESTAMP WHERE id=?',
                       (selected_answer, json.dumps(result, ensure_ascii=False), cursor.lastrowid, attempt_id))
            return result
