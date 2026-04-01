import sys
import os
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage

# 백엔드 경로 주입 및 모듈 로드
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
sys.path.append(PROJECT_ROOT)

from backend.chat_engine import AITutorEngine
from backend import db_manager

# 1. 페이지 레이아웃 설정
st.set_page_config(page_title="AI 자격증 일타 강사", layout="wide")

# 2. 애플리케이션 시작 시 DB 텐서 공간 초기화
db_manager.init_db()

# 3. 텐서 엔진 로드 (캐싱)
@st.cache_resource
def load_engine():
    return AITutorEngine()

tutor_engine = load_engine()

# 4. 세션 상태(메모리 텐서) 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []
if "memory" not in st.session_state:
    st.session_state.memory = []
if "mode" not in st.session_state:
    st.session_state.mode = "study" # 기본 모드: 학습(대화)

# --- 5. 좌측 사이드바 제어반 ---
with st.sidebar:
    st.title("일타 강사 튜터 시스템")
    st.subheader("정보처리기사")
    st.write("---")
    
    # 모드 전환 버튼 텐서 (버튼 클릭 시 이전 퀴즈 텐서를 파괴하여 새 문제 강제 출제)
    if st.button("💬 AI 튜터와 대화 (개념 학습)", use_container_width=True):
        st.session_state.mode = "study"
    
    if st.button(" 일반 기출문제 풀기", use_container_width=True):
        st.session_state.mode = "quiz_random"
        if "current_quiz" in st.session_state:
            del st.session_state.current_quiz
            
    if st.button(" 취약점 집중 공략 문제 풀기" , type="primary", use_container_width=True):
        st.session_state.mode = "quiz_weakness"
        if "current_quiz" in st.session_state:
            del st.session_state.current_quiz
    
    st.write("---")
    # 1. 학습 진도 초기화 진행 버튼
    if st.button(" 학습 진도 초기화", type="secondary"):
        st.session_state.show_confirm = True

    # 2. '확인 대기 상태'일 때만 나타나는 경고 UI 텐서 렌더링
    if st.session_state.get("show_confirm", False):
        st.warning("⚠️ 지금까지의 문제풀이 기록이 전부 삭제됩니다. 정말로 모든 오답 기록을 삭제하시겠습니까? (복구 불가)")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("✅ 확인", type="primary"):
                # 삭제 연산 실행
                db_manager.reset_quiz_logs()
                if "current_quiz" in st.session_state:
                    del st.session_state.current_quiz
                
                # 확인 창을 닫기 위해 상태를 False로 되돌림
                st.session_state.show_confirm = False
                st.rerun() # 화면 즉시 새로고침
                
        with col2:
            if st.button("❌ 취소"):
                # 삭제 없이 창만 닫음
                st.session_state.show_confirm = False
                st.rerun()

# --- 6. 메인 콘텐츠 영역 ---

# [모드 A: 개념 학습 모드]
if st.session_state.mode == "study":
    st.header("🤖 AI 튜터와 대화 (개념 학습)")
    
    chat_container = st.container(height=500)
    with chat_container:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    if user_input := st.chat_input("질문을 입력해주세요 (예: 객체지향 설계 원칙이 뭐야?)"):
        with chat_container:
            with st.chat_message("user"):
                st.markdown(user_input)
            st.session_state.messages.append({"role": "user", "content": user_input})
            
            with st.chat_message("assistant"):
                with st.spinner("지식 및 상태 텐서 분석 중..."):
                    # 1. DB에서 현재 가장 많이 틀린 취약점 텐서 인출
                    weak_topic = db_manager.get_weakest_topic()
                    status_tensor = f"현재 학생이 가장 많이 틀린 오답률 1위 취약 단원은 [{weak_topic}] 파트입니다."
                    
                    # 2. 엔진에 질문, 기억, 그리고 상태 텐서를 모두 융합하여 전달
                    answer = tutor_engine.generate_response(user_input, st.session_state.memory, status_tensor)
                    st.markdown(answer)
                    
                    st.session_state.memory.append(HumanMessage(content=user_input))
                    st.session_state.memory.append(AIMessage(content=answer))
                    st.session_state.messages.append({"role": "assistant", "content": answer})

# [모드 B: 문제 풀이 모드 (랜덤 & 취약점 공통 로직)]
elif st.session_state.mode in ["quiz_random", "quiz_weakness"]:
    
    # 상단 타이틀 동적 렌더링
    if st.session_state.mode == "quiz_random":
        st.header("🎲 실전 대비 무작위 모의고사")
    else:
        st.header("🎯 AI 분석: 나의 취약 개념 집중 타격")

    # 문제 텐서 생성
    if "current_quiz" not in st.session_state:
        if st.session_state.mode == "quiz_random":
            with st.spinner("지식 공간에서 무작위 기출문제를 생성 중..."):
                st.session_state.current_quiz = tutor_engine.generate_quiz()
        else:
            weak_topic = db_manager.get_weakest_topic()
            with st.spinner(f"분석된 취약점 [{weak_topic}] 기반의 타겟 문제를 생성 중..."):
                st.session_state.current_quiz = tutor_engine.generate_quiz(target_topic=weak_topic)
        
        st.session_state.submitted = False

    quiz = st.session_state.current_quiz

    # 문제 화면 렌더링
    with st.container(border=True):
        st.subheader(f"Q. {quiz['question']}")
        st.caption(f"🏷️ 출제 파트: {quiz.get('topic', '정보처리기사 개념')}")
        
        choice = st.radio("정답을 선택하세요", quiz["choices"], index=None, key="quiz_radio")
        
        col1, col2 = st.columns([1, 4])
        with col1:
            if st.button("정답 제출", type="primary"):
                st.session_state.submitted = True

        with col2:
            if st.button("🔄 다음 문제"):
                del st.session_state.current_quiz
                st.rerun()

        # 채점 및 기록 텐서 발송
        if st.session_state.submitted:
            if choice:
                selected_num = int(choice[0]) 
                is_correct = (selected_num == quiz["answer"])
                
                # 정답 여부를 판단한 즉시 SQLite 데이터베이스에 영구 기록 텐서 발송!
                db_manager.log_quiz_result(quiz["topic"], is_correct)
                
                if is_correct:
                    st.success("🎉 정답입니다!")
                else:
                    st.error(f"❌ 오답입니다! (정답: {quiz['answer']}번)")
                
                st.info(f"💡 [AI 해설]: {quiz['explanation']}")
            else:
                st.warning("보기를 선택한 후 제출해 주세요!")