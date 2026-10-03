"""Developer tools. Import is offline; verify/replenish explicitly call Gemini."""
import argparse
import hashlib
import json
import re
import os
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from backend.api_store import APIStore
from backend.cert_config import CERT_CONFIG
from backend.db_manager import DB_PATH
from backend.mock_exams import add_question, availability, validate_question
from backend.summary_notes import source_material, select_context

QUIZ_ROOT = Path(__file__).parent / 'storage' / 'quiz'


def parse_markdown(cert, text):
    """Conservative parser; questionable or unmapped entries are excluded."""
    labels = {label: topic for topic,label in CERT_CONFIG[cert]['topics'].items()}
    for index, block in enumerate(re.split(r'(?m)^\s*---\s*$', text)):
        if not block.strip():
            continue
        try:
            subject = re.search(r'#\s*과목\s*:\s*([^#\n]+)', block)
            if not subject or subject[1].strip() not in labels:
                raise ValueError('unmapped_subject')
            if re.search(r'복수\s*정답|모두\s*정답|정답\s*없|문제\s*오류|정답\s*처리', block):
                raise ValueError('ambiguous_answer')
            body = re.search(r'문제\s*:\s*([\s\S]*?)(?=^\s*-\s*(?:해설|정답)\s*:)', block, re.M)
            answer = re.search(r'^\s*-\s*정답\s*:\s*\*{0,2}(\d+)\)', block, re.M)
            explanation = re.search(r'^\s*-\s*해설\s*:\s*([\s\S]*?)(?=^\s*-\s*정답\s*:|\Z)', block, re.M)
            if not body or not answer or not explanation:
                raise ValueError('incomplete_fields')
            options = list(re.finditer(r'(?m)^\s*(\d+)\)\s*', body[1]))
            if [int(m[1]) for m in options] != list(range(1, CERT_CONFIG[cert].get('option_count',4)+1)):
                raise ValueError('invalid_options')
            choices = [body[1][m.end():options[i+1].start() if i+1<len(options) else len(body[1])].strip() for i,m in enumerate(options)]
            quiz = dict(topic=labels[subject[1].strip()], question=body[1][:options[0].start()].strip(),
                        options=choices, answer=int(answer[1]), explanation=explanation[1].strip(), code_block=None, table_data=None)
            validate_question(cert,quiz)
            yield index, quiz, None
        except ValueError as exc:
            yield index, None, str(exc)


def import_files(store, cert, dry_run=False):
    result = dict(imported=0, candidates=0, duplicates=0, excluded=0, reasons={})
    for path in sorted((QUIZ_ROOT/cert).rglob('*.md')):
        text = path.read_text(encoding='utf-8-sig')
        for index,quiz,error in parse_markdown(cert,text):
            if error:
                result['excluded'] += 1
                result['reasons'][error] = result['reasons'].get(error,0)+1
                continue
            result['candidates'] += 1
            if not dry_run:
                bank_id = add_question(store,cert,quiz,dict(kind='import',file=path.relative_to(QUIZ_ROOT).as_posix(),
                    file_hash=hashlib.sha256(text.encode()).hexdigest(),section=index))
                result['imported' if bank_id else 'duplicates'] += 1
    return result


def review_candidate(engine, cert, quiz, source=None):
    validate_question(cert,quiz)
    # Only imported bank entries use the trusted-original policy. Unknown sources
    # and generated candidates retain the existing concept-grounding requirement.
    if source and source.get('kind') == 'import':
        verdict = engine.verify_imported_quiz(quiz,cert)
        if verdict.get('is_valid') is not True:
            raise ValueError('Verification rejected: '+str(verdict.get('feedback','')))
        return dict(kind='import_consistency_verified',policy='trusted-quiz-original-v1',
                    model='gemini-2.5-flash',source=source,verdict=verdict)
    material = source_material(cert,quiz['topic'])
    context,sources = select_context(material,CERT_CONFIG[cert]['topics'][quiz['topic']]+' '+quiz['question'])
    if not context:
        raise ValueError('No matching subject concept source')
    verdict = engine.verify_quiz(quiz,context,cert)
    if verdict.get('is_valid') is not True:
        raise ValueError('Verification rejected: '+str(verdict.get('feedback','')))
    return dict(kind='ai_verified',model='gemini-2.5-flash',source_hash=material['hash'],sources=sources,verdict=verdict)


@contextmanager
def worker_lock(store):
    """One preparation process per database; OS releases the lock on a crash."""
    path = Path(store.path).with_suffix('.bank.lock')
    with path.open('a+b') as handle:
        handle.seek(0, 2)
        if handle.tell() == 0:
            handle.write(b'0'); handle.flush()
        handle.seek(0)
        try:
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise ValueError('Another question-bank worker is running for this database') from exc
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == 'nt':
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def run_job(store, kind, params, resume=None, engine_factory=None):
    with worker_lock(store):
        return _run_job(store, kind, params, resume, engine_factory)


def _run_job(store, kind, params, resume=None, engine_factory=None):
    if resume:
        with store.connect() as db:
            job = db.execute('SELECT * FROM question_bank_jobs WHERE id=?',(resume,)).fetchone()
            if not job or job['status'] not in ('interrupted','failed','running'):
                raise ValueError('Only interrupted/failed/crashed jobs can resume; inspect status first')
            kind,params = job['kind'],json.loads(job['params'])
            job_id = resume
            db.execute("UPDATE question_bank_jobs SET status='running' WHERE id=?",(job_id,))
    else:
        job_id = str(uuid4())
        if kind == 'verify':
            with store.connect() as db:
                params['ids'] = [r['id'] for r in db.execute("SELECT id FROM question_bank WHERE cert=? AND status='pending' ORDER BY COALESCE(json_extract(review,'$.last_attempt'),''),rowid LIMIT ?",(params['cert'],params['max_attempts']))]
        with store.connect() as db:
            db.execute("INSERT INTO question_bank_jobs(id,kind,params,status) VALUES(?,?,?,'running')",(job_id,kind,json.dumps(params)))
    print('Job:',job_id, flush=True)
    engine = None
    status = 'completed'
    last_error = None
    try:
        while True:
            with store.connect() as db:
                job = db.execute('SELECT * FROM question_bank_jobs WHERE id=?',(job_id,)).fetchone()
            index = job['attempts']
            if kind == 'verify' and index >= len(params['ids']):
                break
            cert = params['cert']
            if kind == 'verify':
                if index >= len(params['ids']):
                    break
                with store.connect() as db:
                    row = db.execute('SELECT * FROM question_bank WHERE id=?',(params['ids'][index],)).fetchone()
                topic = row['topic']
            else:
                topics = [s for s in availability(store,cert)['subjects'] if (not params.get('topic') or s['topic']==params['topic']) and s['ready']<params['target']]
                if not topics:
                    break
                topic = topics[index % len(topics)]['topic']
            if index >= params['max_attempts']:
                status = 'budget_exhausted'
                break
            if engine is None:
                if engine_factory:
                    engine = engine_factory()
                else:
                    from backend.chat_engine import AITutorEngine
                    engine = AITutorEngine()
            # Count before the call: a process interruption cannot erase spent budget.
            with store.connect() as db:
                db.execute('UPDATE question_bank_jobs SET attempts=attempts+1,updated_at=CURRENT_TIMESTAMP WHERE id=?',(job_id,))
            success = False
            try:
                if kind == 'verify':
                    quiz = json.loads(row['question'])
                else:
                    with store.connect() as db:
                        history = [json.loads(r['question']) for r in db.execute('SELECT question FROM question_bank WHERE cert=? AND topic=? ORDER BY rowid DESC LIMIT 20',(cert,topic))]
                    quiz = engine.generate_advanced_quiz(cert=cert,target_topic=topic,strict_subject=True,generated_history=history)
                    if quiz.get('topic') != topic:
                        raise ValueError('Generated question does not match requested subject')
                source = json.loads(row['source']) if kind == 'verify' else {'kind':'generated'}
                evidence = review_candidate(engine,cert,quiz,source=source)
                if kind == 'verify':
                    with store.connect() as db:
                        updated = db.execute("UPDATE question_bank SET status='ready',review=? WHERE id=? AND status='pending'",(json.dumps(evidence,ensure_ascii=False),row['id']))
                        success = updated.rowcount == 1
                else:
                    success = add_question(store,cert,quiz,dict(kind='generated',job_id=job_id),status='ready',review=evidence) is not None
                    if not success:
                        raise ValueError('Duplicate candidate')
            except Exception as exc:
                last_error = str(exc)
                print('Candidate rejected:',last_error,flush=True)
                if kind == 'verify':
                    with store.connect() as db:
                        db.execute("UPDATE question_bank SET review=? WHERE id=? AND status='pending'",
                                   (json.dumps(dict(kind='verification_failed',last_attempt=datetime.now(timezone.utc).isoformat(),feedback=last_error)),row['id']))
            with store.connect() as db:
                db.execute('UPDATE question_bank_jobs SET succeeded=succeeded+?,failed=failed+?,last_error=?,updated_at=CURRENT_TIMESTAMP WHERE id=?',
                           (int(success),int(not success),last_error,job_id))
    except KeyboardInterrupt:
        status = 'interrupted'
    except Exception as exc:
        status,last_error = 'failed',str(exc)
    finally:
        with store.connect() as db:
            db.execute('UPDATE question_bank_jobs SET status=?,last_error=?,updated_at=CURRENT_TIMESTAMP WHERE id=?',(status,last_error,job_id))
    return job_id


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--db', default=DB_PATH)
    sub = parser.add_subparsers(dest='command',required=True)
    for name in ('status','import-files','verify','replenish'):
        command = sub.add_parser(name)
        command.add_argument('--cert',required=True,choices=list(CERT_CONFIG))
        if name=='import-files':
            command.add_argument('--dry-run',action='store_true')
        if name in ('verify','replenish'):
            command.add_argument('--max-attempts',type=int,default=20)
        if name=='replenish':
            command.add_argument('--topic')
            command.add_argument('--target','--target-per-subject',dest='target',type=int,default=50)
    sub.add_parser('jobs')
    command = sub.add_parser('list'); command.add_argument('--cert', required=True, choices=list(CERT_CONFIG)); command.add_argument('--status', choices=['pending','ready','disabled'], default='pending')
    command = sub.add_parser('show'); command.add_argument('question_id')
    command = sub.add_parser('resume'); command.add_argument('job_id')
    for name in ('approve','disable'):
        command = sub.add_parser(name); command.add_argument('question_id'); command.add_argument('--reason',required=True)
    args = parser.parse_args()
    store = APIStore(args.db)
    if args.command=='status':
        output = availability(store,args.cert)
    elif args.command=='import-files':
        output = import_files(store,args.cert,args.dry_run)
    elif args.command in ('verify','replenish'):
        if args.max_attempts < 1 or (args.command=='replenish' and (args.target<1 or (args.topic and args.topic not in CERT_CONFIG[args.cert]['topics']))):
            parser.error('Check topic, target and positive attempt limit')
        output = {'job_id':run_job(store,args.command,dict(cert=args.cert,max_attempts=args.max_attempts,topic=getattr(args,'topic',None),target=getattr(args,'target',None)))}
    elif args.command=='resume':
        output = {'job_id':run_job(store,None,None,resume=args.job_id)}
    elif args.command=='jobs':
        with store.connect() as db:
            output = [dict(r) for r in db.execute('SELECT * FROM question_bank_jobs ORDER BY rowid DESC LIMIT 30')]
    elif args.command=='list':
        with store.connect() as db:
            output = [dict(r) for r in db.execute('SELECT id,topic,status,question FROM question_bank WHERE cert=? AND status=? ORDER BY rowid', (args.cert,args.status))]
    elif args.command=='show':
        with store.connect() as db:
            row = db.execute('SELECT * FROM question_bank WHERE id=?',(args.question_id,)).fetchone()
            if not row: parser.error('Question not found')
            output = dict(row)
            for field in ('question','source','review'):
                output[field] = json.loads(output[field]) if output[field] else None
    else:
        with store.connect() as db:
            row = db.execute('SELECT * FROM question_bank WHERE id=?',(args.question_id,)).fetchone()
            if not row: parser.error('Question not found')
            validate_question(row['cert'],json.loads(row['question']))
            db.execute('UPDATE question_bank SET status=?,review=? WHERE id=?',
                       ('ready' if args.command=='approve' else 'disabled',json.dumps({'kind':'manual','reason':args.reason}),args.question_id))
        output = {'updated':args.question_id}
    print(json.dumps(output,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
