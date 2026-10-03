"""Shared dashboard/weakness policy, independent of UI and AI generation."""
from datetime import datetime, timedelta, timezone

from backend.cert_config import CERT_CONFIG

MIN_ATTEMPTS = 5
WEAK_ACCURACY_THRESHOLD = 60
DAILY_GOAL = 50
KST = timezone(timedelta(hours=9))


def learning_stats(store, cert, now=None):
    config = CERT_CONFIG[cert]
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        raise ValueError('now must include a timezone')
    local_start = now.astimezone(KST).replace(hour=0, minute=0, second=0, microsecond=0)
    start = local_start.astimezone(timezone.utc)
    end = start + timedelta(days=1)
    rows = store.quiz_statistics(cert, start.strftime('%Y-%m-%d %H:%M:%S'),
                                end.strftime('%Y-%m-%d %H:%M:%S'))
    by_topic = {row['topic']: row for row in rows}
    subjects = []
    for topic, label in config['topics'].items():
        row = by_topic.get(topic, {'total': 0, 'correct': 0})
        total, correct = row['total'], row['correct']
        subjects.append(dict(topic=topic, label=label, total=total, correct=correct,
                             accuracy=round(correct * 100 / total, 1) if total else None,
                             status='unlearned' if not total else
                                    ('insufficient_data' if total < MIN_ATTEMPTS else 'analyzed')))
    eligible = [s for s in subjects if s['total'] >= MIN_ATTEMPTS]
    # Compare raw ratios, not rounded percentages, at the threshold.
    ranked = [s for s in eligible if s['correct'] * 100 < s['total'] * WEAK_ACCURACY_THRESHOLD]
    ranked.sort(key=lambda s: (s['correct'] / s['total'], -s['total']))
    focus = ranked[0] if ranked else None
    status = 'ready' if focus else ('no_weakness' if eligible else 'insufficient_data')
    if focus:
        message = f"{focus['label']}: {focus['total']}문제 중 {focus['correct']}문제 정답, 정답률 {focus['accuracy']}%"
    elif eligible:
        message = '분석 가능한 과목 중 취약 기준에 해당하는 과목이 없습니다.'
    else:
        message = f'취약점 분석을 위해 한 과목에서 최소 {MIN_ATTEMPTS}문제를 풀어 주세요.'
    total = sum(row['total'] for row in rows)
    correct = sum(row['correct'] for row in rows)
    today = sum(row['today'] for row in rows)
    return dict(cert=cert, date=local_start.date().isoformat(), timezone='Asia/Seoul',
                has_records=total > 0, total_solved=total, correct_solved=correct,
                accuracy=round(correct * 100 / total, 1) if total else None,
                today_solved=today, daily_goal=DAILY_GOAL,
                goal_rate=min(round(today * 100 / DAILY_GOAL), 100), subjects=subjects,
                unclassified_solved=sum(r['total'] for r in rows if r['topic'] not in config['topics']),
                weakness=dict(status=status, message=message, focus=focus, ranking=ranked[:3],
                              min_attempts=MIN_ATTEMPTS, accuracy_threshold=WEAK_ACCURACY_THRESHOLD,
                              insufficient_subjects=[s['topic'] for s in subjects if s['total'] < MIN_ATTEMPTS]))
