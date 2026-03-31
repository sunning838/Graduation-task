import sys
import os
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
#streamlit run frontend/app.py 로 실행

# 백엔드 경로 주입
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
sys.path.append(PROJECT_ROOT)

from backend.chat_engine import AITutorEngine

# 1. 페이지 레이아웃 설정
st.set_page_config(page_title="AI 자격증 일타 강사", layout="wide")

# 2. 텐서 엔진 로드 (캐싱)
@st.cache_resource
def load_engine():
    return AITutorEngine()

tutor_engine = load_engine()

# 3. 세션 상태 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []
if "memory" not in st.session_state:
    st.session_state.memory = []
if "mode" not in st.session_state:
    st.session_state.mode = "study" # 기본 모드: 학습(대화)

# --- 4. 좌측 사이드바 구현 ---
with st.sidebar:
    st.title(" 일타 강사 튜터 시스템")
    st.subheader(f"정보처리기사")
    st.write("---")
    
    # 모드 전환 버튼
    if st.button("💬 AI 튜터와 대화 (개념 학습)", use_container_width=True):
        st.session_state.mode = "study"
    
    if st.button("✨ 기출문제 풀기", use_container_width=True):
        st.session_state.mode = "quiz"
    
    #st.write("---")
    #if st.button("로그아웃", type="secondary"):
    #    st.write("로그아웃 되었습니다.")

# --- 5. 메인 콘텐츠 영역 ---

# [모드 A: 개념 학습 모드]
if st.session_state.mode == "study":
    st.header("🤖 AI 튜터와 대화 (개념 학습)")
    
    # 대화창 컨테이너
    chat_container = st.container(height=500)
    
    with chat_container:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    # 입력창
    if user_input := st.chat_input("결합도와 응집도의 차이가 뭐야?"):
        with chat_container:
            # 사용자 메시지 표시
            with st.chat_message("user"):
                st.mrkdown(user_input)
            st.session_state.messages.append({"role": "user", "content": user_input})
            
            # AI 답변 생성
            with st.chat_message("assistant"):
                with st.spinner("지식 텐서 분석 중..."):
                    answer = tutor_engine.generate_response(user_input, st.session_state.memory)
                    st.markdown(answer)
                    
                    st.session_state.memory.append(HumanMessage(content=user_input))
                    st.session_state.memory.append(AIMessage(content=answer))
                    st.session_state.messages.append({"role": "assistant", "content": answer})

# [모드 B: 문제 풀이 모드]
elif st.session_state.mode == "quiz":
    st.header("📝 AI가 출제한 오답 집중 공략")

    # 세션 상태에 현재 문제 텐서가 없다면 새로 생성하여 영점 주입
    if "current_quiz" not in st.session_state:
        with st.spinner("지식 공간에서 새로운 문제 텐서를 생성하는 중..."):
            st.session_state.current_quiz = tutor_engine.generate_quiz()
            st.session_state.submitted = False # 제출 여부 초기화

    quiz = st.session_state.current_quiz

    # 문제 화면 렌더링
    with st.container(border=True):
        st.subheader(f"Q. {quiz['question']}")
        
        # 사용자의 선택 텐서 추적
        choice = st.radio(
            "정답을 선택하세요",
            quiz["choices"],
            index=None,
            key="quiz_radio"
        )
        
        col1, col2 = st.columns([1, 4])
        with col1:
            if st.button("정답 제출", type="primary"):
                st.session_state.submitted = True

        with col2:
            # 새로운 문제 생성 버튼
            if st.button("🔄 다른 문제 풀기"):
                del st.session_state.current_quiz # 기존 문제 텐서 파괴
                st.rerun() # 화면 재갱신하여 새 문제 생성 유도

        # 제출 완료 후 결과 및 해설 텐서 출력
        if st.session_state.submitted:
            if choice:
                # 선택한 인덱스 번호 추출 (예: "3) 이행 함수..." -> 숫자 3)
                selected_num = int(choice[0]) 
                
                if selected_num == quiz["answer"]:
                    st.success("🎉 정답입니다!")
                else:
                    st.error(f"❌ 오답입니다! (정답: {quiz['answer']}번)")
                
                # 해설 텐서 분출
                st.info(f"💡 [AI 해설]: {quiz['explanation']}")
            else:
                st.warning("보기를 선택한 후 제출해 주세요!")