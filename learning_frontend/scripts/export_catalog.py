"""백엔드의 단원 목록을 React용 src/data/catalog.json으로 내보내는 스크립트.
백엔드 파일은 읽기만 하고 수정하지 않음.

사용법 (React 프로젝트 폴더에서):
    <Graduation-task의 venv python 경로> scripts/export_catalog.py [Graduation-task 경로]
"""
import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / 'src' / 'data' / 'catalog.json'

# Graduation-task 위치 (다르면 실행할 때 경로를 뒤에 붙여 주면 됨)
DEFAULT_REPO = Path.home() / 'OneDrive' / 'Desktop' / 'Graduation-task'
REPO = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_REPO

if not (REPO / 'backend').exists():
    print(f'Graduation-task 폴더를 찾지 못했습니다: {REPO}')
    print('실행할 때 경로를 직접 붙여 주세요. 예) python scripts/export_catalog.py C:\\경로\\Graduation-task')
    sys.exit(1)

sys.path.insert(0, str(REPO))
from backend.learning import catalog  # noqa: E402
from backend.lesson_display import display_title  # noqa: E402

config, lessons = catalog()

data = {
    'config': {
        cert: {'label': entry.get('label', cert), 'topics': dict(entry.get('topics', {}))}
        for cert, entry in config.items()
    },
    'lessons': [
        {
            'id': l['id'],
            'cert': l['cert'],
            'subject': l['subject'],
            'title': display_title(l['title']),
        }
        for l in lessons
    ],
}

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')

print(f'저장 완료: {OUT}')
for cert, count in Counter(l['cert'] for l in lessons).items():
    print(f"  {config[cert].get('label', cert)}: {count}개 단원")