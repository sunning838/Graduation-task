# 백엔드 테스트

모든 명령은 프로젝트 루트 `D:\Capstone_Project`에서 실행합니다.
다른 위치에 복제한 팀원도 해당 프로젝트 루트에서 같은 상대 경로 명령을 사용하면 됩니다.

## 자동 테스트 전체 실행

```powershell
.\venv\Scripts\python.exe -m backend.test
```

`test_*.py`를 자동으로 찾아 실행합니다. 새 자동 테스트도 이 이름 규칙으로 추가하면 됩니다.
임시 DB와 모의 AI를 사용하며 실제 학습 기록을 변경하거나 Gemini API를 호출하지 않습니다.
실패하면 종료 코드 1을 반환하므로 CI에서도 사용할 수 있습니다.

특정 파일만 실행하려면:

```powershell
.\venv\Scripts\python.exe -m unittest backend.test.test_mock_preparation -v
```

| 파일 | 역할 |
|---|---|
| test_api_connections.py | API 연결, 대화·일반 문제풀이 |
| test_learning_stats_api.py | 학습 통계·취약점·과목별 검색 |
| test_summary_notes_api.py | 취약점 요약 저장·근거·실패 처리 |
| test_mock_exams.py | 문제은행 검수·모의고사 구성·답안·채점 |
| test_mock_preparation.py | 버튼으로 시작한 자동 보충·작업 복원·진행률 |
| benchmark_mock_exams.py | 임시 문제은행에서 시험 구성 시간 측정 |
| tensor_search_smoke.py | 실제 Chroma DB 검색 수동 점검 |

## 성능 측정 (별도 실행)

```powershell
.\venv\Scripts\python.exe -m backend.test.benchmark_mock_exams
```

임시 DB에서 20/50/100문항 구성 시간을 측정합니다. 실제 데이터와 Gemini는 사용하지 않습니다.
AI 생성 시간이나 브라우저 표시 시간 측정은 아닙니다.

## 실제 검색 점검 (별도 실행)

```powershell
.\venv\Scripts\python.exe -m backend.test.tensor_search_smoke
```

`backend/chroma_db`를 조회하므로 해당 DB와 임베딩 모델이 필요합니다.
모델이 로컬에 없으면 최초 다운로드가 발생할 수 있습니다. 기본 자동 테스트에는 포함하지 않습니다.

파일 경로로 직접 실행하지 말고 위처럼 `-m`으로 실행해야 프로젝트 모듈을 정상적으로 찾습니다.
React 전용 검사는 기존 `react_frontend/scripts`에서 관리합니다.
