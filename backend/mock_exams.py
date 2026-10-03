"""Question bank and immutable exam snapshots. No AI calls in exam operations."""
import hashlib
import json
import random
import re
from difflib import SequenceMatcher
from uuid import uuid4

from backend.api_store import StateError
from backend.cert_config import CERT_CONFIG

SCHEMA = '''
CREATE TABLE IF NOT EXISTS question_bank (
 id TEXT PRIMARY KEY, cert TEXT NOT NULL, topic TEXT NOT NULL, question TEXT NOT NULL,
 content_hash TEXT NOT NULL, fingerprint TEXT NOT NULL, source TEXT NOT NULL,
 status TEXT NOT NULL CHECK(status IN ('pending','ready','disabled')),
 version INTEGER NOT NULL DEFAULT 1, review TEXT,
 created_at TEXT DEFAULT CURRENT_TIMESTAMP, UNIQUE(cert,content_hash)
);
CREATE INDEX IF NOT EXISTS question_bank_selection ON question_bank(cert,topic,status);
CREATE TABLE IF NOT EXISTS mock_exams (
 id TEXT PRIMARY KEY, cert TEXT NOT NULL, distribution TEXT NOT NULL,
 status TEXT NOT NULL DEFAULT 'taking', result TEXT, request_id TEXT UNIQUE NOT NULL,
 created_at TEXT DEFAULT CURRENT_TIMESTAMP, submitted_at TEXT
);
CREATE TABLE IF NOT EXISTS mock_exam_items (
 id TEXT PRIMARY KEY, exam_id TEXT NOT NULL, bank_id TEXT NOT NULL, position INTEGER NOT NULL,
 question TEXT NOT NULL, selected_answer INTEGER, is_correct INTEGER, log_id INTEGER,
 UNIQUE(exam_id, position), UNIQUE(exam_id, bank_id)
);
CREATE TABLE IF NOT EXISTS question_bank_jobs (
 id TEXT PRIMARY KEY, kind TEXT NOT NULL, params TEXT NOT NULL, status TEXT NOT NULL,
 attempts INTEGER NOT NULL DEFAULT 0, succeeded INTEGER NOT NULL DEFAULT 0,
 failed INTEGER NOT NULL DEFAULT 0, last_error TEXT,
 created_at TEXT DEFAULT CURRENT_TIMESTAMP, updated_at TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS mock_preparations (
 request_id TEXT PRIMARY KEY, cert TEXT NOT NULL, distribution TEXT NOT NULL,
 status TEXT NOT NULL DEFAULT 'queued', exam_id TEXT, attempts INTEGER NOT NULL DEFAULT 0,
 max_attempts INTEGER NOT NULL, tried TEXT NOT NULL DEFAULT '[]', last_error TEXT,
 updated_at TEXT DEFAULT CURRENT_TIMESTAMP
);
'''


def validate_question(cert, quiz):
    config = CERT_CONFIG[cert]
    if not isinstance(quiz, dict) or quiz.get('is_fallback') or quiz.get('validation_failed'):
        raise ValueError('Generation/verification failed')
    if quiz.get('topic') not in config['topics']:
        raise ValueError('Unknown subject')
    for field in ('question', 'explanation'):
        if not isinstance(quiz.get(field), str) or not quiz[field].strip():
            raise ValueError('Missing ' + field)
    options = quiz.get('options')
    if not isinstance(options, list) or len(options) != config.get('option_count', 4):
        raise ValueError('Invalid option count')
    if any(not isinstance(o, str) or not o.strip() for o in options):
        raise ValueError('Invalid option')
    cleaned = [re.sub(r'^\s*\d+[).:]\s*', '', o).strip() for o in options]
    if len(set(cleaned)) != len(cleaned):
        raise ValueError('Duplicate options')
    if type(quiz.get('answer')) is not int or not 1 <= quiz['answer'] <= len(options):
        raise ValueError('Invalid answer')
    if any(quiz.get(key) is not None and not isinstance(quiz[key], str) for key in ('code_block','table_data')):
        raise ValueError('Invalid question formatting')


def fingerprint(quiz):
    text = '\n'.join(quiz.get(k) or '' for k in ('question', 'table_data', 'code_block'))
    return re.sub(r'\s+', '', text).casefold()


def add_question(store, cert, quiz, source, status='pending', review=None):
    validate_question(cert, quiz)
    if status == 'ready' and not review:
        raise ValueError('Review evidence is required')
    fp = fingerprint(quiz)
    with store.connect() as db:
        db.execute('BEGIN IMMEDIATE')
        # Serialize deduplication with insertion, including concurrent import jobs.
        rows = db.execute('SELECT fingerprint FROM question_bank WHERE cert=?', (cert,)).fetchall()
        if any(fp == r['fingerprint'] or SequenceMatcher(None, fp, r['fingerprint'], autojunk=False).ratio() >= .95 for r in rows):
            return None
        bank_id = str(uuid4())
        db.execute('INSERT INTO question_bank(id,cert,topic,question,content_hash,fingerprint,source,status,review) VALUES(?,?,?,?,?,?,?,?,?)',
                   (bank_id, cert, quiz['topic'], json.dumps(quiz, ensure_ascii=False), hashlib.sha256(fp.encode()).hexdigest(), fp,
                    json.dumps(source, ensure_ascii=False), status, json.dumps(review, ensure_ascii=False) if review else None))
    return bank_id


def availability(store, cert):
    with store.connect() as db:
        rows = db.execute('SELECT topic,status,COUNT(*) AS n FROM question_bank WHERE cert=? GROUP BY topic,status', (cert,)).fetchall()
    counts = {(r['topic'], r['status']): r['n'] for r in rows}
    return dict(cert=cert, subjects=[dict(topic=t, label=label, ready=counts.get((t,'ready'),0),
                pending=counts.get((t,'pending'),0), disabled=counts.get((t,'disabled'),0))
                for t,label in CERT_CONFIG[cert]['topics'].items()])


def exam_view(store, exam_id):
    with store.connect() as db:
        db.execute('BEGIN')
        exam = db.execute('SELECT * FROM mock_exams WHERE id=?', (exam_id,)).fetchone()
        if not exam:
            raise StateError(404, 'EXAM_NOT_FOUND', '모의고사를 찾을 수 없습니다.')
        items = db.execute('SELECT * FROM mock_exam_items WHERE exam_id=? ORDER BY position', (exam_id,)).fetchall()
    submitted = exam['status'] == 'submitted'
    public = []
    for row in items:
        quiz = json.loads(row['question'])
        item = dict(id=row['id'], position=row['position'], topic=quiz['topic'],
                    topic_label=CERT_CONFIG[exam['cert']]['topics'].get(quiz['topic'], quiz['topic']),
                    question=quiz['question'], options=quiz['options'], code_block=quiz.get('code_block'),
                    table_data=quiz.get('table_data'), selected_answer=row['selected_answer'])
        if submitted:
            item.update(correct_answer=quiz['answer'], explanation=quiz['explanation'], is_correct=bool(row['is_correct']))
        public.append(item)
    return dict(id=exam_id, cert=exam['cert'], status=exam['status'], created_at=exam['created_at'],
                submitted_at=exam['submitted_at'], items=public, result=json.loads(exam['result']) if submitted else None)


def create_exam(store, cert, distribution, request_id):
    if not distribution or len({d['topic'] for d in distribution}) != len(distribution):
        raise StateError(422, 'INVALID_DISTRIBUTION', '중복 없이 출제 과목을 선택해 주세요.')
    if any(d['topic'] not in CERT_CONFIG[cert]['topics'] or type(d['count']) is not int or d['count'] < 1 for d in distribution):
        raise StateError(422, 'INVALID_DISTRIBUTION', '과목과 문항 수를 확인해 주세요.')
    if sum(d['count'] for d in distribution) > 100:
        raise StateError(422, 'EXAM_TOO_LARGE', '한 시험은 최대 100문항입니다.')
    distribution = sorted(distribution, key=lambda d: list(CERT_CONFIG[cert]['topics']).index(d['topic']))
    encoded = json.dumps(distribution, sort_keys=True)
    with store.connect() as db:
        db.execute('BEGIN IMMEDIATE')
        previous = db.execute('SELECT * FROM mock_exams WHERE request_id=?', (request_id,)).fetchone()
        if previous:
            if previous['cert'] != cert or previous['distribution'] != encoded:
                raise StateError(409, 'REQUEST_CONFLICT', '이미 다른 시험 설정에 사용한 요청 ID입니다.')
            exam_id = previous['id']
        else:
            chosen, missing = [], []
            for part in distribution:
                rows = list(db.execute('''SELECT b.*, MAX(e.created_at) AS last_used
                    FROM question_bank b LEFT JOIN mock_exam_items i ON b.id=i.bank_id
                    LEFT JOIN mock_exams e ON e.id=i.exam_id
                    WHERE b.cert=? AND b.topic=? AND b.status='ready' GROUP BY b.id''', (cert, part['topic'])))
                if len(rows) < part['count']:
                    missing.append(f"{CERT_CONFIG[cert]['topics'][part['topic']]}: 요청 {part['count']}개 / 가능 {len(rows)}개")
                random.SystemRandom().shuffle(rows)
                rows.sort(key=lambda r: r['last_used'] or '')
                chosen.extend(rows[:part['count']])
            if missing:
                raise StateError(409, 'INSUFFICIENT_QUESTIONS', '출제 가능한 문제가 부족합니다. ' + '; '.join(missing))
            exam_id = str(uuid4())
            db.execute('INSERT INTO mock_exams(id,cert,distribution,request_id) VALUES(?,?,?,?)', (exam_id,cert,encoded,request_id))
            for position,row in enumerate(chosen):
                db.execute('INSERT INTO mock_exam_items(id,exam_id,bank_id,position,question) VALUES(?,?,?,?,?)',
                           (str(uuid4()),exam_id,row['id'],position,row['question']))
    return exam_view(store, exam_id)


def save_answer(store, exam_id, item_id, answer):
    with store.connect() as db:
        db.execute('BEGIN IMMEDIATE')
        row = db.execute('SELECT i.question,e.status FROM mock_exam_items i JOIN mock_exams e ON e.id=i.exam_id WHERE i.id=? AND e.id=?',
                         (item_id,exam_id)).fetchone()
        if not row:
            raise StateError(404, 'ITEM_NOT_FOUND', '모의고사 문항을 찾을 수 없습니다.')
        if row['status'] != 'taking':
            raise StateError(409, 'EXAM_SUBMITTED', '제출한 시험의 답안은 변경할 수 없습니다.')
        if answer is not None and (type(answer) is not int or not 1 <= answer <= len(json.loads(row['question'])['options'])):
            raise StateError(422, 'INVALID_OPTION', '유효한 보기 번호를 선택해 주세요.')
        db.execute('UPDATE mock_exam_items SET selected_answer=? WHERE id=?', (answer,item_id))
    return {'item_id': item_id, 'selected_answer': answer, 'saved': True}


def submit_exam(store, exam_id, confirm_unanswered=False):
    with store.connect() as db:
        db.execute('BEGIN IMMEDIATE')
        exam = db.execute('SELECT * FROM mock_exams WHERE id=?', (exam_id,)).fetchone()
        if not exam:
            raise StateError(404, 'EXAM_NOT_FOUND', '모의고사를 찾을 수 없습니다.')
        if exam['status'] == 'taking':
            items = db.execute('SELECT * FROM mock_exam_items WHERE exam_id=? ORDER BY position', (exam_id,)).fetchall()
            unanswered = sum(i['selected_answer'] is None for i in items)
            if unanswered and not confirm_unanswered:
                raise StateError(409, 'UNANSWERED_ITEMS', f'미응답 {unanswered}문항이 있습니다. 미응답을 오답 처리하고 제출할지 확인해 주세요.')
            correct, subjects = 0, {}
            for item in items:
                quiz = json.loads(item['question'])
                ok = item['selected_answer'] == quiz['answer']
                correct += int(ok)
                group = subjects.setdefault(quiz['topic'], {'topic':quiz['topic'], 'label':CERT_CONFIG[exam['cert']]['topics'].get(quiz['topic'],quiz['topic']), 'total':0,'correct':0})
                group['total'] += 1
                group['correct'] += int(ok)
                cursor = db.execute('INSERT INTO quiz_logs(cert,topic,is_correct) VALUES(?,?,?)', (exam['cert'],quiz['topic'],int(ok)))
                db.execute('UPDATE mock_exam_items SET is_correct=?,log_id=? WHERE id=?', (int(ok),cursor.lastrowid,item['id']))
            for group in subjects.values():
                group['accuracy'] = round(group['correct']*100/group['total'],1)
            result = dict(total=len(items),correct=correct,score=round(correct*100/len(items),1), unanswered=unanswered, subjects=list(subjects.values()))
            db.execute("UPDATE mock_exams SET status='submitted',result=?,submitted_at=CURRENT_TIMESTAMP WHERE id=?", (json.dumps(result,ensure_ascii=False),exam_id))
    return exam_view(store,exam_id)
