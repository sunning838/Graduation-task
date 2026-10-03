"""Persistent, bounded exam preparation driven by explicit user requests."""
import json
import logging
from threading import Thread

from backend.api_store import StateError
from backend import mock_exams as bank
from backend.question_bank_cli import worker_lock, review_candidate

LOG = logging.getLogger(__name__)


def view(store, request_id):
    with store.connect() as db:
        db.execute('BEGIN')
        row = db.execute('SELECT * FROM mock_preparations WHERE request_id=?',(request_id,)).fetchone()
        counts = {} if not row else {r['topic']:r['n'] for r in db.execute(
            "SELECT topic,COUNT(*) AS n FROM question_bank WHERE cert=? AND status='ready' GROUP BY topic",(row['cert'],))}
    if not row:
        raise StateError(404,'PREPARATION_NOT_FOUND','시험 준비 요청을 찾을 수 없습니다.')
    messages = {'queued':'시험 준비를 기다리고 있습니다.', 'running':'시험 문제를 준비하고 있습니다. 잠시 기다려 주세요.',
                'completed':'시험 준비가 완료되었습니다.', 'failed':'시험 준비를 완료하지 못했습니다. 다시 시도해 주세요.'}
    parts = json.loads(row['distribution'])
    total = sum(p['count'] for p in parts)
    ready = total if row['status']=='completed' else sum(min(p['count'],counts.get(p['topic'],0)) for p in parts)
    return dict(request_id=request_id,status=row['status'],exam_id=row['exam_id'],message=messages[row['status']],
                progress=dict(total=total,ready=ready,remaining=total-ready,percent=round(ready*100/total)))


def start(store, cert, distribution, request_id):
    encoded = json.dumps(sorted(distribution,key=lambda p:p['topic']),sort_keys=True)
    with store.connect() as db:
        row = db.execute('SELECT * FROM mock_preparations WHERE request_id=?',(request_id,)).fetchone()
    if row:
        if row['cert'] != cert or row['distribution'] != encoded:
            raise StateError(409,'REQUEST_CONFLICT','다른 시험 설정에 사용한 요청 ID입니다.')
        return view(store,request_id)
    # This validates the distribution and assembles immediately if already stocked.
    exam = None
    try:
        exam = bank.create_exam(store,cert,distribution,request_id)
    except StateError as exc:
        if exc.code != 'INSUFFICIENT_QUESTIONS':
            raise
    with store.connect() as db:
        db.execute('BEGIN IMMEDIATE')
        db.execute('INSERT OR IGNORE INTO mock_preparations(request_id,cert,distribution,status,exam_id,max_attempts) VALUES(?,?,?,?,?,?)',
                   (request_id,cert,encoded,'completed' if exam else 'queued',exam['id'] if exam else None,min(300,sum(p['count'] for p in distribution)*3+10)))
        row = db.execute('SELECT * FROM mock_preparations WHERE request_id=?',(request_id,)).fetchone()
        if row['cert'] != cert or row['distribution'] != encoded:
            raise StateError(409,'REQUEST_CONFLICT','다른 시험 설정에 사용한 요청 ID입니다.')
    return view(store,request_id)


def dispatch(store, request_id, engine_factory):
    Thread(target=run, args=(store,request_id,engine_factory),daemon=True).start()


def run(store, request_id, engine_factory):
    # Same OS lock as CLI: polling cannot start overlapping paid calls, and a
    # crashed server releases its lock so the persisted job can be resumed.
    locked = False
    try:
        with worker_lock(store):
            locked = True
            _run(store,request_id,engine_factory)
    except Exception as exc:
        if not locked and isinstance(exc, ValueError):
            return  # Another worker owns the lock; a later poll retries.
        LOG.exception('Mock preparation worker failed')
        with store.connect() as db:
            db.execute("UPDATE mock_preparations SET status='failed',last_error='worker_failure' WHERE request_id=? AND status!='completed'",(request_id,))


def _run(store, request_id, engine_factory):
    engine = None
    failures = 0
    while True:
        with store.connect() as db:
            row = db.execute('SELECT * FROM mock_preparations WHERE request_id=?',(request_id,)).fetchone()
        if not row or row['status'] in ('completed','failed'):
            return
        cert, parts = row['cert'], json.loads(row['distribution'])
        try:
            exam = bank.create_exam(store,cert,parts,request_id)
        except StateError as exc:
            if exc.code != 'INSUFFICIENT_QUESTIONS':
                raise
        else:
            with store.connect() as db:
                db.execute("UPDATE mock_preparations SET status='completed',exam_id=?,updated_at=CURRENT_TIMESTAMP WHERE request_id=?",(exam['id'],request_id))
            return
        if row['attempts'] >= row['max_attempts']:
            with store.connect() as db:
                db.execute("UPDATE mock_preparations SET status='failed',last_error='attempt_limit' WHERE request_id=?",(request_id,))
            return
        counts = {s['topic']:s['ready'] for s in bank.availability(store,cert)['subjects']}
        deficits = [p for p in parts if counts[p['topic']] < p['count']]
        topic = deficits[row['attempts'] % len(deficits)]['topic']
        tried = json.loads(row['tried'])
        with store.connect() as db:
            candidates = db.execute("SELECT * FROM question_bank WHERE cert=? AND topic=? AND status='pending' ORDER BY rowid",(cert,topic)).fetchall()
            candidate = next((c for c in candidates if c['id'] not in tried),None)
            if candidate:
                tried.append(candidate['id'])
            db.execute("UPDATE mock_preparations SET status='running',attempts=attempts+1,tried=?,updated_at=CURRENT_TIMESTAMP WHERE request_id=?",(json.dumps(tried),request_id))
        try:
            if engine is None:
                engine = engine_factory()
            if candidate:
                quiz,source = json.loads(candidate['question']),json.loads(candidate['source'])
            else:
                with store.connect() as db:
                    history=[json.loads(r['question']) for r in db.execute('SELECT question FROM question_bank WHERE cert=? AND topic=? ORDER BY rowid DESC LIMIT 20',(cert,topic))]
                quiz=engine.generate_advanced_quiz(cert=cert,target_topic=topic,strict_subject=True,generated_history=history)
                if quiz.get('topic') != topic or quiz.get('is_fallback'):
                    raise ValueError('No usable subject question')
                source={'kind':'generated','preparation_id':request_id}
            evidence=review_candidate(engine,cert,quiz,source=source)
            if candidate:
                with store.connect() as db:
                    db.execute("UPDATE question_bank SET status='ready',review=? WHERE id=? AND status='pending'",(json.dumps(evidence,ensure_ascii=False),candidate['id']))
            else:
                if not bank.add_question(store,cert,quiz,source,'ready',evidence):
                    raise ValueError('Duplicate candidate')
            failures = 0
        except Exception as exc:
            failures += 1
            LOG.warning('Preparation %s candidate failed: %s',request_id,exc)
            with store.connect() as db:
                db.execute('UPDATE mock_preparations SET last_error=?,updated_at=CURRENT_TIMESTAMP WHERE request_id=?',(str(exc),request_id))
            # Initialization/service failures should not repeatedly retry paid calls.
            if engine is None or failures >= 3:
                with store.connect() as db:
                    db.execute("UPDATE mock_preparations SET status='failed' WHERE request_id=?",(request_id,))
                return
