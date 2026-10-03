"""Source-grounded summaries. No external search, no mutations to original material."""
import hashlib
import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path

from backend.api_store import StateError
from backend.cert_config import CERT_CONFIG
from backend.learning_stats import learning_stats
from backend.lesson_display import learner_text, display_title

LOG = logging.getLogger(__name__)
DATA_ROOT = Path(__file__).parent / 'storage' / 'data'
PROMPT_VERSION = 'weakness-summary-v1'
MODEL = 'gemini-2.5-flash'


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def source_material(cert, topic):
    """Read current files so a rebuilt vector index is not needed after source edits."""
    files, sections = [], []
    for path in sorted((DATA_ROOT / cert / topic).glob('*.md')):
        text = path.read_text(encoding='utf-8-sig')
        files.append({'source': path.relative_to(DATA_ROOT).as_posix(), 'hash': digest(text)})
        for index, part in enumerate(re.split(r'(?m)^## ', text)):
            body = part.partition('\n')[2] if index else part
            body = re.sub(r'(?m)^#+[^\n]*$', '', body).strip()
            if body:
                sections.append((path.relative_to(DATA_ROOT).as_posix(), part.strip()))
    return {'hash': digest(files), 'files': files, 'sections': sections}


def select_context(material, label):
    sections = material['sections']
    if not sections:
        return '', []
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    texts = [section for _, section in sections]
    matrix = TfidfVectorizer(analyzer='char', ngram_range=(1, 3), max_features=20000).fit_transform(
        texts + [label + ' 핵심 개념 정의 특징 비교 주의사항'])
    scores = cosine_similarity(matrix[-1], matrix[:-1]).ravel()
    ranked = sorted(range(len(texts)), key=lambda i: (-scores[i], i))[:6]
    snippets = [(sections[i][0], texts[i][:2500]) for i in ranked]
    return '\n\n'.join(text for _, text in snippets), [dict(source=source, hash=digest(text)) for source, text in snippets]


def snapshot(store, cert):
    stats = learning_stats(store, cert)
    # Ignore calendar-only changes; invalidate when actual performance changes.
    signature = digest({k: stats[k] for k in ('subjects', 'total_solved', 'correct_solved')})
    return stats, signature


def is_stale(note, signature, material):
    meta = note['metadata']
    return (meta.get('stats_hash') != signature or meta.get('source_hash') != material['hash']
            or meta.get('prompt_version') != PROMPT_VERSION or meta.get('model') != MODEL)


def summary_view(store, cert):
    stats, signature = snapshot(store, cert)
    selected = {s['topic'] for s in stats['weakness']['ranking']}
    saved = store.latest_summary_notes(cert)
    materials = {}
    for topic in set(saved) | selected:
        try:
            materials[topic] = source_material(cert, topic)
        except (OSError, UnicodeError):
            LOG.exception('Cannot read summary source for %s/%s', cert, topic)
            materials[topic] = {'hash': None}
    notes = []
    for topic, note in saved.items():
        material = materials[topic]
        notes.append(dict(id=note['id'], topic=topic,
                          label=CERT_CONFIG[cert]['topics'].get(topic, topic), markdown=note['markdown'],
                          created_at=note['created_at'], basis=note['metadata']['basis'],
                          stale=is_stale(note, signature, material), is_current_topic=topic in selected))
    order = {s['topic']: i for i, s in enumerate(stats['weakness']['ranking'])}
    notes.sort(key=lambda n: (order.get(n['topic'], 99), n['topic']))
    pending = [t for t in selected if t not in saved or is_stale(saved[t], signature, materials[t])]
    state = store.summary_generation_state(cert)
    return dict(cert=cert, label=CERT_CONFIG[cert]['label'], weakness=stats['weakness'], notes=notes,
                needs_update=bool(pending), can_generate=bool(selected), **state)


def generate_summaries(store, cert, engine_factory):
    token = store.begin_summary_generation(cert)
    issues = []
    try:
        stats, signature = snapshot(store, cert)
        analysis = stats['weakness']
        if not analysis['ranking']:
            raise StateError(409, 'NO_SUMMARY_TARGET', analysis['message'])
        saved = store.latest_summary_notes(cert)
        engine = None
        for basis in analysis['ranking']:
            topic = basis['topic']
            try:
                material = source_material(cert, topic)
                if topic in saved and not is_stale(saved[topic], signature, material):
                    continue
                context, sources = select_context(material, basis['label'])
                if not context:
                    issues.append(dict(topic=topic, label=basis['label'], code='NO_MATERIAL', message='개념 자료가 부족해 요약을 생성하지 못했습니다.'))
                    continue
                if engine is None:
                    engine = engine_factory()
                result = engine.generate_final_note(cert, [topic], contexts={topic: context})
                if not isinstance(result, str) or not result.strip():
                    raise ValueError('Empty summary')
                text = learner_text(result).strip()
                if not text:
                    raise ValueError('Empty display summary')
                metadata = dict(stats_hash=signature, source_hash=material['hash'], files=material['files'],
                                sources=sources, basis=basis, model=MODEL, prompt_version=PROMPT_VERSION)
                store.save_summary_note(cert, token, topic, text, metadata, datetime.now(timezone.utc).isoformat())
            except StateError:
                raise
            except Exception:
                LOG.exception('Summary generation failed for %s/%s', cert, topic)
                issues.append(dict(topic=topic, label=display_title(basis['label']), code='GENERATION_FAILED',
                                   message='요약을 준비하지 못했습니다. 다시 시도해 주세요. 기존 요약은 유지됩니다.'))
    finally:
        store.finish_summary_generation(cert, token, issues)
    return summary_view(store, cert)
