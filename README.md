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
