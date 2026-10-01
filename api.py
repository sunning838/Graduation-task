from uuid import uuid4

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from backend.chat_engine import AITutorEngine
from backend import db_manager
from backend.cert_config import TOPIC_KOR_MAP


app = FastAPI()


# =========================================================
# CORS
# React(localhost:5173) → FastAPI(localhost:8000)
# =========================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================================================
# 시스템 초기화
# =========================================================

db_manager.init_db()

# AI 엔진은 서버 시작 시 한 번만 생성
tutor_engine = AITutorEngine()


# 문제 데이터를 임시로 서버 메모리에 보관
# 추후 로그인/사용자 기능을 만들 때 DB 구조로 변경 가능
quiz_store = {}


# =========================================================
# 요청 / 응답 모델
# =========================================================

class ChatRequest(BaseModel):
    message: str
    cert: str = "EIP"
    answer_length: str = "medium"


class ChatResponse(BaseModel):
    answer: str


class QuizRequest(BaseModel):
    cert: str = "EIP"


class QuizSubmitRequest(BaseModel):
    quiz_id: str
    selected_answer: int


# =========================================================
# 서버 상태 확인
# =========================================================

@app.get("/")
def root():
    return {
        "message": "AI Tutor API 서버가 정상 실행 중입니다."
    }


# =========================================================
# AI Tutor 채팅
# =========================================================

@app.post("/api/chat", response_model=ChatResponse)
def chat(request: ChatRequest):

    length_instructions = {
        "short": (
            "답변은 핵심만 매우 간결하게 설명하세요. "
            "가능하면 3~5문장 정도로 답변하고 "
            "불필요한 세부 설명이나 긴 예시는 생략하세요."
        ),

        "medium": (
            "핵심 개념을 중심으로 이해하기 쉽게 설명하세요. "
            "필요한 경우 짧은 예시나 목록을 사용할 수 있지만 "
            "지나치게 길게 설명하지 마세요."
        ),

        "long": (
            "개념을 충분히 이해할 수 있도록 자세하게 설명하세요. "
            "필요한 경우 이유, 특징, 예시, 비교, 표, 목록 등을 활용하세요."
        ),
    }

    length_instruction = length_instructions.get(
        request.answer_length,
        length_instructions["medium"]
    )

    query = f"""
사용자 질문:
{request.message}

[답변 길이 지침]
{length_instruction}
"""

    answer = tutor_engine.generate_response(
        query=query,
        chat_history=[],
        student_status="분석된 상태 없음",
        cert=request.cert,
    )

    return ChatResponse(
        answer=answer
    )


# =========================================================
# 일반 문제 생성
# =========================================================

@app.post("/api/quiz")
def create_quiz(request: QuizRequest):

    try:
        quiz = tutor_engine.generate_advanced_quiz(
            cert=request.cert
        )

        quiz_id = str(uuid4())

        # 정답과 해설을 포함한 전체 데이터는
        # 서버에만 저장
        quiz_store[quiz_id] = {
            "cert": request.cert,
            "quiz": quiz,
        }

        topic = quiz.get("topic", "알 수 없음")

        # React에는 문제 풀이에 필요한 데이터만 전달
        # 정답(answer)은 아직 보내지 않음
        return {
            "quiz_id": quiz_id,
            "question": quiz.get(
                "question",
                "문제를 불러오지 못했습니다."
            ),
            "topic": topic,
            "topic_label": TOPIC_KOR_MAP.get(
                topic,
                topic
            ),
            "options": quiz.get(
                "options",
                []
            ),
            "code_block": quiz.get(
                "code_block"
            ),
            "table_data": quiz.get(
                "table_data"
            ),
        }

    except Exception as e:
        print(f"[API 오류] 문제 생성 실패: {e}")

        raise HTTPException(
            status_code=500,
            detail="문제를 생성하지 못했습니다."
        )


# =========================================================
# 문제 정답 제출 및 채점
# =========================================================

@app.post("/api/quiz/submit")
def submit_quiz(request: QuizSubmitRequest):

    stored = quiz_store.get(
        request.quiz_id
    )

    if not stored:
        raise HTTPException(
            status_code=404,
            detail="문제 정보를 찾을 수 없습니다."
        )

    cert = stored["cert"]
    quiz = stored["quiz"]

    correct_answer = int(
        quiz.get("answer", -1)
    )

    is_correct = (
        request.selected_answer
        == correct_answer
    )

    topic = quiz.get(
        "topic",
        "알 수 없음"
    )

    # 기존 학습 기록 DB에 저장
    db_manager.log_quiz_result(
        cert,
        topic,
        is_correct
    )

    return {
        "is_correct": is_correct,
        "selected_answer": request.selected_answer,
        "correct_answer": correct_answer,
        "explanation": quiz.get(
            "explanation",
            "해설이 없습니다."
        ),
    }