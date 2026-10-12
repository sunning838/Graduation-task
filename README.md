# 🤖 AI Tutor for IT Certification (RAG Project)

정보처리기사 자격증 취득을 돕는 AI 튜터 시스템입니다.

## 🚀 Key Features

- **Data Pipeline**: 560+개의 정보처리기사 지식 블록을 텐서화
- **Embedding**: Google Gemini (`gemini-embedding-001`) 최신 모델 적용
- **Vector DB**: ChromaDB를 활용한 로컬 지식 베이스 구축
- **RAG Architecture**: 검색 증강 생성 방식을 통한 정확한 답변 생성

## 🛠 Tech Stack

- Python 3.14
- LangChain, ChromaDB
- Google Gemini API

## 텐서 구동 환경 설치 방법

프로젝트를 다운로드(Clone)한 후 해당 폴더로 이동합니다.

터미널에서 아래 명령어를 실행하여 텐서 처리 및 데이터베이스 구동에 필요한 모든 모듈을 한 번에 설치합니다.

pip install -r requirements.txt

streamlit run frontend/app.py 명령어로 서버를 실행합니다

## React 화면 간편 실행

프로젝트 폴더의 `start_web.cmd`를 더블클릭하면 API(8000)와 React(5173)를 실행하고,
준비 완료 후 브라우저를 엽니다. 실행 창은 유지하고 종료할 때 그 창에서 `Ctrl+C`를 누르세요.
기존 서버가 같은 포트에서 실행 중이면 먼저 종료해야 합니다.
최초 Python 가상환경·requirements 및 React의 `npm install`은 미리 완료해야 합니다.
기존 `start.cmd`는 이전 Streamlit 화면용입니다.

## 백엔드 테스트 실행

프로젝트 루트에서 `.\venv\Scripts\python.exe -m backend.test`로 자동 테스트를 실행합니다.
테스트 파일과 성능 측정·검색 점검 도구는 `backend/test`에서 관리합니다.
개별 실행 방법은 [백엔드 테스트 안내](backend/test/README.md)를 확인하세요.

-자료 출처 - 
시나공 기출문제집 정보처리기사 필기(길벗알앤디 지음/ 강윤석, 김용갑, 감우경, 김종일)
국가법령정보센터
