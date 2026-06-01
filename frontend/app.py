import sys
import os
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
import re
import pandas as pd
import plotly.express as px

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

# 자격증 매핑 딕셔너리
CERT_MAP = {
    "정보처리기사": "EIP",
    "공인중개사 1차": "LREA_1"
}

TOPIC_KOR_MAP = {
    #정보처리기사
    "software_design": "소프트웨어 설계",
    "software_development": "소프트웨어 개발",
    "database": "데이터베이스 구축",
    "programming_language": "프로그래밍 언어 활용",
    "info_system": "정보시스템 구축 관리",

    #공인중개사
    "civil_law": "민법 및 민사특별법",
    "housing_lease": "주택임대차보호법",
    "commercial_lease": "상가건물 임대차보호법",
    "aggregate_building": "집합건물법",
    "provisional_registration": "가등기담보법",
    "real_name_registration": "부동산실명법"
}

# 4. 세션 상태 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []
if "memory" not in st.session_state:
    st.session_state.memory = []
if "mode" not in st.session_state:
    st.session_state.mode = "study"

def reset_quiz_state():
    if "current_quiz" in st.session_state:
        del st.session_state.current_quiz
    st.session_state.submitted = False

# --- 5. 좌측 사이드바 제어반 ---
with st.sidebar:
    st.title("일타 강사 튜터 시스템")
    
    # 자격증 선택 UI (드롭다운)
    selected_cert_label = st.selectbox("🎓 자격증을 선택하세요", list(CERT_MAP.keys()))
    selected_cert = CERT_MAP[selected_cert_label]

    # 자격증 변경 감지 및 세션(컨텍스트) 완벽 초기화
    if "current_cert" not in st.session_state:
        st.session_state.current_cert = selected_cert

    if st.session_state.current_cert != selected_cert:
        st.session_state.current_cert = selected_cert
        st.session_state.messages = []  # 대화 내역 초기화
        st.session_state.memory = []    # AI 문맥 초기화
        reset_quiz_state()              # 풀던 문제 초기화
        st.rerun()                      # 화면 새로고침하여 바뀐 자격증으로 시작

    st.subheader(f"[{selected_cert_label}] 모드")
    st.write("---")
    
    if st.button("💬 AI 튜터와 대화 (개념 학습)", use_container_width=True):
        st.session_state.mode = "study"
    
    if st.button("📝 일반 기출문제 풀기", use_container_width=True):
        st.session_state.mode = "quiz_random"
        reset_quiz_state()
            
    if st.button("🎯 취약점 집중 공략 문제 풀기" , type="primary", use_container_width=True):
        st.session_state.mode = "quiz_weakness"
        reset_quiz_state()

    if st.button("📝 약점 족집게 개념노트 생성", type="secondary", use_container_width=True):
        st.session_state.mode = "final_note"
        reset_quiz_state()
    
    st.write("---")
    if st.button("🗑️ 학습 진도 초기화", type="secondary"):
        st.session_state.show_confirm = True

    if st.session_state.get("show_confirm", False):
        st.warning("⚠️ 지금까지의 문제풀이 기록이 전부 삭제됩니다. 정말로 모든 오답 기록을 삭제하시겠습니까? (복구 불가)")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("✅ 확인", type="primary"):
                db_manager.reset_quiz_logs()
                reset_quiz_state()
                
                if "final_note_content" in st.session_state:
                    del st.session_state["final_note_content"]
                    
                st.session_state.show_confirm = False
                st.rerun()
        with col2:
            if st.button("❌ 취소"):
                st.session_state.show_confirm = False
                st.rerun()

# --- 6. 메인 콘텐츠 영역 ---

with st.expander("나의 학습 취약점 분석", expanded=False):
    stats = db_manager.get_subject_stats(selected_cert)
    
    if stats:
        # 데이터프레임으로 변환 (영문 topic을 한글로 번역)
        df_stats = pd.DataFrame({
            '과목': [TOPIC_KOR_MAP.get(k, k) for k in stats.keys()],
            '정답률(%)': list(stats.values())
        })
        
        # Plotly 레이더 차트 생성
        fig = px.line_polar(
            df_stats, 
            r='정답률(%)', 
            theta='과목', 
            line_close=True, 
            range_r=[0, 100],
            markers=True
        )
        fig.update_traces(fill='toself', fillcolor='rgba(0, 150, 255, 0.3)', line_color='rgba(0, 150, 255, 1)')
        fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])), margin=dict(l=40, r=40, t=20, b=20))
        
        st.plotly_chart(fig, use_container_width=True)
        
        # 간단한 분석 코멘트
        weakest = min(stats, key=stats.get)
        st.caption(f"💡 **분석 결과**: 현재 **[{TOPIC_KOR_MAP.get(weakest, weakest)}]** 과목이 가장 취약합니다. 집중 공부가 필요합니다.")
    else:
        st.info("아직 푼 문제가 없습니다. 모의고사를 풀면 여기에 분석 레이더가 나타납니다!")

#st.write("---") # 구분선

if st.session_state.mode == "study":
   # st.header(f"🤖 {selected_cert_label} AI 튜터와 대화")
    
    chat_container = st.container(height=600)
    with chat_container:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    if user_input := st.chat_input("질문을 입력해주세요"):
        with chat_container:
            with st.chat_message("user"):
                st.markdown(user_input)
            st.session_state.messages.append({"role": "user", "content": user_input})
            
            with st.chat_message("assistant"):
                with st.spinner("지식 및 상태 텐서 분석 중..."):
                    weak_topic = db_manager.get_weakest_topic(selected_cert)
                    status_tensor = f"현재 학생이 가장 많이 틀린 취약 단원은 [{weak_topic}] 파트입니다."
                    
                    # 엔진에 현재 자격증 코드(selected_cert)도 함께 넘겨줌
                    answer = tutor_engine.generate_response(user_input, st.session_state.memory, status_tensor, selected_cert)
                    st.markdown(answer)
                    
                    st.session_state.memory.append(HumanMessage(content=user_input))
                    st.session_state.memory.append(AIMessage(content=answer))
                    st.session_state.messages.append({"role": "assistant", "content": answer})

elif st.session_state.mode in ["quiz_random", "quiz_weakness"]:
    if st.session_state.mode == "quiz_random":
        st.header(f"📝 {selected_cert_label} 실전 무작위 모의고사")
    else:
        st.header(f"🎯 {selected_cert_label} 취약 개념 집중 타격")

    if "current_quiz" not in st.session_state:
        with st.status("🔍 AI 에이전트들이 문제를 구성하고 있습니다...", expanded=True) as status:
            if st.session_state.mode == "quiz_random":
                st.write("✍️ **출제위원**: 기출 데이터를 분석하여 변형 문제를 출제 중...")
                # 출제 엔진에 자격증 파라미터 전달
                st.session_state.current_quiz = tutor_engine.generate_advanced_quiz(cert=selected_cert)
            else:
                weak_topic = db_manager.get_weakest_topic(selected_cert)
                st.write(f"✍️ **출제위원**: 취약점 [{weak_topic}] 타겟 문제를 생성 중...")
                # 타겟팅된 단원과 함께 자격증 전달
                st.session_state.current_quiz = tutor_engine.generate_advanced_quiz(target_topic=weak_topic, cert=selected_cert)
            
            status.update(label="문제 제출 완료.", state="complete", expanded=False)
        st.session_state.submitted = False

    quiz = st.session_state.current_quiz

    with st.container(border=True):
        st.markdown(f"### Q. {quiz.get('question', '문제 불러오기 실패')}")

        raw_topic = quiz.get('topic', '알 수 없음')
        display_topic = TOPIC_KOR_MAP.get(raw_topic, raw_topic)
        st.caption(f"🏷️ 출제 파트: {display_topic}")

        if quiz.get("code_block"):
            st.code(quiz["code_block"])
        if quiz.get("table_data"):
            st.markdown(quiz.get("table_data"))
        
        options = quiz.get("options", ["보기 오류"])
        choice = st.radio("정답을 선택하세요", options, index=None, key="quiz_radio", disabled=st.session_state.submitted)
        
        if not st.session_state.submitted:
            if st.button("정답 제출", type="primary"):
                if choice:
                    st.session_state.submitted = True
                    match = re.search(r'\d+', choice)
                    selected_num = int(match.group()) if match else -1 

                    is_correct = (selected_num == quiz.get("answer"))
                    db_manager.log_quiz_result(selected_cert, quiz.get("topic"), is_correct)

                    #문제풀이 db가 업데이트되면 오답노트 갱신
                    if "final_note_content" in st.session_state:
                        del st.session_state["final_note_content"]

                    st.rerun() 
                else:
                    st.warning("보기를 선택한 후 제출해 주세요!")

        if st.session_state.submitted:
            match = re.search(r'\d+', choice)
            selected_num = int(match.group()) if match else -1
            is_correct = (selected_num == quiz.get("answer"))
            
            st.write("---") 
            if is_correct:
                st.success("🎉 정답입니다!")
            else:
                st.error(f"❌ 오답입니다! (정답: {quiz.get('answer', '알 수 없음')}번)")
            
            st.info(f"💡 [AI 해설]: {quiz.get('explanation', '해설을 불러오지 못했습니다.')}")
            
            col1, col2, col3 = st.columns([1, 1, 1])
            with col2: 
                if st.button("🔄 다음 문제", use_container_width=True):
                    reset_quiz_state()
                    st.rerun()

elif st.session_state.mode == "final_note":
    st.header(f"📖 {selected_cert_label} 파이널 요약 오답노트")
    
    with st.spinner("학생의 오답 패턴을 분석하여 맞춤형 요약본을 생성하고 있습니다... "):
        top_weak_topics = db_manager.get_top_weakest_topics(selected_cert, n=3)
        
        if not top_weak_topics:
            st.warning("아직 틀린 문제가 없거나 데이터가 부족합니다. 모의고사를 더 풀어주세요")
        else:
            display_topics = [TOPIC_KOR_MAP.get(t, t) for t in top_weak_topics]
            st.info(f"🔍 **집중 분석된 핵심 취약 단원 Top 3**: {', '.join(display_topics)}")
            
            if "final_note_content" not in st.session_state or st.session_state.get("final_note_cert") != selected_cert:
                final_note = tutor_engine.generate_final_note(selected_cert, top_weak_topics)
                st.session_state.final_note_content = final_note
                st.session_state.final_note_cert = selected_cert
            else:
                final_note = st.session_state.final_note_content
                
            # 화면에 렌더링
            with st.container(border=True):
                st.markdown(final_note)
            
            st.download_button(
                label="📥 나만의 오답노트 다운로드 (.md)",
                data=final_note,
                file_name=f"{selected_cert_label}_파이널_오답노트.md",
                mime="text/markdown",
                type="primary"
            )
            st.caption("※ 다운로드한 .md 파일은 마크다운 뷰어 혹은 브라우저에서 열어 PDF로 인쇄(Ctrl+P)할 수 있습니다.")