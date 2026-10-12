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
| test_learning_routes.py | 개념학습 목록·진도·강의·도식 응답 및 단원 문제 생성/채점 |
| check_concept_ui.mjs | 개념학습 SVG·Markdown·초기 화면 렌더링 점검 |
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

새 개념학습 화면 검사는 테스트 파일 관리 방침에 따라 이 폴더에 추가했습니다.
프로젝트 루트에서 `node backend/test/check_concept_ui.mjs`로 실행합니다. React 의존성 설치가 필요합니다.

## 개념학습 연동 확인

정보처리기사 학습실은 `lessonClient.js`에서 공통 API 주소 설정을 사용합니다.
`/api/learning/teach`는 variant(basic/simple/example), 또는 question/explanation/history를 받습니다.
기존 prompt 형식도 호환됩니다. 그래프 응답은 nodes/edges/layout을 포함하며 내부 근거 원문은 제외합니다.
강의와 튜터 답변은 단원 ID·자료 버전별 브라우저 메모리에 재사용하며 새로고침 후에는 재요청합니다.
진도는 기존 learning.db에 저장합니다. 완료 저장 및 다음 단원 시작 저장에 성공한 뒤 이동합니다.

`POST /api/quiz`의 mode=lesson, lesson_id, cert=EIP 요청은 등록된 단원 원문으로 문제를 생성하고
검수한 뒤 기존 풀이 저장/채점/통계 흐름에 연결합니다. 외부 검색은 이 출제 경로에서 사용하지 않습니다.
강의→단원 문제풀이→학습하던 단원으로 돌아가기의 UI 상태는 같은 앱 실행 중 유지합니다.

실제 Gemini 생성 품질과 시간은 별도 실제 실행 확인이 필요합니다. 자동 테스트는 모의 AI를 사용합니다.
AudioLecture.jsx, 기존 음성 api.js, TTS 경로·음성 요청 형식은 이번 연동에서 변경하지 않았습니다.
