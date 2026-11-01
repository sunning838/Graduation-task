"""Opt-in smoke: one image, or two with --two-slides; a rejected image may be repaired once."""
import json
from pathlib import Path
import sys

from backend.learning import catalog
from backend.lecture_presentation import PresentationManager


def main():
    if '--run-live' not in sys.argv:
        print('Paid API smoke is opt-in: python -m backend.test.presentation_live_smoke --run-live')
        return
    _, lessons = catalog()
    lesson = next(l for l in lessons if l['cert'] == 'EIP' and '소프트웨어 생명 주기' in l['title'])
    text = ('### 프로토타입 모형\n\n사용자의 요구사항을 정확히 파악하기 위해 견본인 시제품을 먼저 만드는 개발 모형입니다.\n\n'
            '- 시제품은 사용자와 개발자가 함께 확인하는 참조 모델입니다.\n- 사용자가 원하는 기능을 시제품으로 확인합니다.')
    expected = 2 if '--two-slides' in sys.argv else 1
    if expected == 2:
        text += ('\n\n### 폭포수 모형\n\n각 개발 단계를 확실히 마무리하고 검토와 승인을 거쳐 다음 단계로 진행하는 선형 순차적 개발 방법입니다.'
                 '\n\n- 각 단계의 결과물을 검토합니다.\n- 승인 후 다음 단계로 진행합니다.')
    manager = PresentationManager(workers=1)
    try:
        result = manager.start(lesson, lessons, 'basic', text)
        if result['status'] in ('failed', 'partial') and '--retry' in sys.argv:
            result = manager.retry(result['package_id'], lessons)
        future = manager.futures.get(result['package_id'])
        if future:
            future.result(420 * expected)
        result = manager.get(result['package_id'])
        target = Path(__file__).parent / '.presentation-live.json'
        target.write_text(json.dumps(dict(lesson_id=lesson['id'], text=text, result=result), ensure_ascii=False), encoding='utf-8')
        print(json.dumps(dict(status=result['status'], images=result['images'], artifact=str(target)), ensure_ascii=False))
        if result['images']['ready'] != expected or result['images']['total'] != len(result['slides']):
            raise SystemExit('Live generation did not produce one reviewed image per slide; see developer logs.')
    finally:
        manager.executor.shutdown(wait=True)


if __name__ == '__main__':
    main()
