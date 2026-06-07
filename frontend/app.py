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
from backend.cert_config import CERT_CONFIG, CERT_MAP, TOPIC_KOR_MAP

# 1. 페이지 레이아웃 설정
st.set_page_config(page_title="AI 자격증 일타 강사", layout="wide")

# 2. 애플리케이션 시작 시 DB 텐서 공간 초기화
db_manager.init_db()

# 3. 텐서 엔진 로드 (캐싱)
@st.cache_resource
def load_engine():
    return AITutorEngine()

tutor_engine = load_engine()
# 4. 세션 상태 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []
if "memory" not in st.session_state:
    st.session_state.memory = []
if "mode" not in st.session_state:
    st.session_state.mode = "study"
if "mock_logged" not in st.session_state:
    st.session_state.mock_logged = False

def reset_quiz_state():
    if "current_quiz" in st.session_state:
        del st.session_state.current_quiz
    st.session_state.submitted = False
    st.session_state.show_explanation = False
    st.session_state.show_essay_mode = False
    st.session_state.essay_graded = False
    st.session_state.socratic_active = False
    st.session_state.socratic_history = []
    st.session_state.socratic_status = "continue"
    st.session_state.hint_level = 0
    if "understanding_map" in st.session_state:
        del st.session_state["understanding_map"]

def get_selected_option_number(selected_option: str, options: list[str]) -> int:
    """
    사용자가 선택한 보기 문자열을 1-based 번호로 변환
    예: options[0] 선택 -> 1
    """
    if not selected_option:
        return -1

    try:
        return options.index(selected_option) + 1
    except ValueError:
        return -1

# 모의고사 상태 초기화 함수
def reset_mock_exam_state():
    st.session_state.mock_step = "setup" # setup(설정), generating(생성중), taking(응시중), result(결과), review(복기)
    st.session_state.mock_questions = [] # 출제된 문제 리스트 보관
    st.session_state.mock_answers = []   # 유저가 선택한 정답 리스트 보관
    st.session_state.mock_current_q = 0  # 현재 풀고 있는 문제 인덱스
    st.session_state.mock_logged = False # 결과 DB 저장 여부

# 모의고사 문제 중복 판정용 함수
def normalize_quiz_signature(quiz: dict) -> str:

    # 문제 본문
    question = (quiz.get("question") or "").strip()

    # 보기 목록을 하나의 문자열로 합침
    options = " | ".join((opt or "").strip() for opt in quiz.get("options", []))

    # 코드가 있는 문제라면 코드 내용도 비교 대상에 포함
    code_block = (quiz.get("code_block") or "").strip()

    # 표가 있는 문제라면 표 내용도 비교 대상에 포함
    table_data = (quiz.get("table_data") or "").strip()

    # 최종적으로 하나의 비교용 문자열을 반환
    return f"{question}__{options}__{code_block}__{table_data}"

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
        reset_mock_exam_state()         # 모의고사 상태 초기화
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

    if st.button("🔥 실전 모의고사 풀어보기", use_container_width=True, type="primary"):
        st.session_state.mode = "mock_exam"
        reset_mock_exam_state()
        st.rerun()

    if st.button("📝 약점 족집게 개념노트 생성", type="primary", use_container_width=True):
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
                reset_mock_exam_state()
                
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

st.markdown("""
<style>
.equal-card {
    border: 1px solid #d9d9d9;
    border-radius: 12px;
    padding: 20px 24px;
    min-height: 210px;
    background-color: white;
}
</style>
""", unsafe_allow_html=True)

if st.session_state.mode == "study":
   # st.header(f"🤖 {selected_cert_label} AI 튜터와 대화")

    # ====================================================================
    # 실시간 자격증별 미니 통계 대시보드
    # ====================================================================
    total, correct, accuracy = db_manager.get_cert_stats(selected_cert)

    DAILY_GOAL = 50
    today_solved = db_manager.get_today_solved_count(selected_cert)
    progress_ratio = min(today_solved / DAILY_GOAL, 1.0)

    left_col, right_col = st.columns([0.60, 0.40])

    with left_col:
        with st.container(border=True):
            st.markdown("#### 🎯 오늘의 목표 달성률")
            st.metric("오늘 풀이 수", f"{today_solved} / {DAILY_GOAL}")
            st.progress(progress_ratio)
            st.caption(f"현재 목표 달성률: {int(progress_ratio * 100)}%")

    with right_col:
        with st.container(border=True):
            st.markdown("### 📊 전체 학습 현황")
            m1, m2, m3 = st.columns(3)
            m1.metric("📝 푼 문제", f"{total}개")
            m2.metric("🎯 맞춘 문제", f"{correct}개")
            m3.metric("🔥 정답률", f"{accuracy:.1f}%")

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
                    selected_num = get_selected_option_number(choice, options)

                    is_correct = (selected_num == quiz.get("answer"))
                    db_manager.log_quiz_result(selected_cert, quiz.get("topic"), is_correct)

                    #문제풀이 db가 업데이트되면 오답노트 갱신
                    if "final_note_content" in st.session_state:
                        del st.session_state["final_note_content"]

                    st.rerun() 
                else:
                    st.warning("보기를 선택한 후 제출해 주세요!")

        if st.session_state.submitted:
            selected_num = get_selected_option_number(choice, options)
            is_correct = (selected_num == quiz.get("answer"))
            
            st.write("---") 
            if is_correct:
                st.success("🎉 정답입니다!")
                
                # -------------------------------------------------------------
                # Step 1: 최초 정답 상태 - 두 개의 선택 버튼 제공
                # -------------------------------------------------------------
                if not st.session_state.show_explanation and not st.session_state.show_essay_mode:
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("💡 AI 해설 보기", use_container_width=True):
                            st.session_state.show_explanation = True
                            st.rerun()
                    with col2:
                        if st.button("🧠 서술형으로 증명받기", type="primary", use_container_width=True):
                            st.session_state.show_essay_mode = True
                            st.rerun()

                # -------------------------------------------------------------
                # 루트 A: [AI 해설 보기]를 선택한 경우
                # -------------------------------------------------------------
                if st.session_state.show_explanation:
                    st.info(f"💬 [AI 해설]: {quiz.get('explanation', '해설을 불러오지 못했습니다.')}")
                    
                    
                    if st.button("➡️ 다음 문제 풀러가기", type="primary", use_container_width=True):
                        reset_quiz_state()
                        st.rerun()

                # -------------------------------------------------------------
                # 루트 B: [서술형으로 증명받기]를 선택한 경우
                # -------------------------------------------------------------
                if st.session_state.show_essay_mode:
                    st.write("---")
                    st.write("### ✍️ 진짜 이해했나요? 서술형으로 증명해보세요!")
                    st.caption(f"객관식은 찍기가 가능합니다. 문제에 대해 자유롭게 해설해보세요")
                    
                    # 채점 완료 전까지만 텍스트 입력창 노출
                    essay_input = st.text_area(
                        "답안 작성란", 
                        placeholder="예: 관계형 데이터베이스는 데이터의 독립성을 보장하고...", 
                        key="essay_input",
                        disabled=st.session_state.essay_graded # 채점 완료 시 수정 불가 락(Lock)
                    )
                    
                    # 아직 채점을 받지 않은 상태일 때만 채점 버튼 노출
                    if not st.session_state.essay_graded:
                        if st.button("📝 AI 채점 요청하기", type="primary"):
                            if len(essay_input) < 10:
                                st.warning("내용이 너무 짧습니다. 조금 더 성의 있게 서술해 주세요!")
                            else:
                                with st.spinner("깐깐한 AI 튜터가 채점 기준에 따라 분석 중입니다..."):
                                    # 핀셋 문제 기반 검색 채점 API 호출
                                    grade = tutor_engine.grade_essay(quiz.get("question"), selected_cert, essay_input)
                                    
                                    # 세션에 채점 데이터 박제
                                    st.session_state.essay_grade_result = grade
                                    st.session_state.essay_graded = True
                                    
                                    # DB 로그 공간에 서술형 스코어 업데이트
                                    db_manager.update_latest_essay_score(selected_cert, quiz.get("topic"), grade["score"])
                                    st.rerun()
                    
                    # 채점이 완료되었다면 결과 리포트 렌더링
                    if st.session_state.essay_graded:
                        grade = st.session_state.essay_grade_result
                        
                        st.write("---")
                        st.subheader(f"🏆 주관식 검증 점수: {grade['score']}점 / 100점")
                        st.progress(grade['score'] / 100)
                        
                        col_a, col_b, col_c = st.columns(3)
                        col_a.metric("🔑 키워드", f"{grade['keyword_score']} / 40")
                        col_b.metric("🔗 논리력", f"{grade['logic_score']} / 30")
                        col_c.metric("🎯 정확도", f"{grade['accuracy_score']} / 30")
                        
                        with st.container(border=True):
                            if grade['found_keywords']:
                                st.success(f"**✅ 발견된 핵심 키워드:** {', '.join(grade['found_keywords'])}")
                            if grade['missing_keywords']:
                                st.error(f"**❌ 누락된 필수 키워드:** {', '.join(grade['missing_keywords'])}")
                            if grade['misconceptions']:
                                st.warning(f"**⚠️ 오개념 주의보:** {', '.join(grade['misconceptions'])}")
                            
                            st.write(f"💬 **튜터의 피드백:** {grade['feedback']}")
                            st.caption(f"💡 **모범 답안 힌트:** {grade['model_answer_hint']}")
                        
                        # 채점 리포트 밑에 객관식용 기존 AI 해설도 함께 덧붙임
                        st.write("---")
                        st.info(f"💡 ** 객관식 문제 AI 해설**:\n\n{quiz.get('explanation')}")
                        
                        # 최종 다음 문제 버튼
                        if st.button("➡️ 완벽하게 이해했나요? 다음 문제로 넘어가기", type="primary", use_container_width=True):
                            reset_quiz_state()
                            st.rerun()
            
            else:
                st.error(f"❌ 오답입니다! (선택한 답: {selected_num}번)")
                
                # -------------------------------------------------------------
                #  오답 분기: [해설 보기] vs [소크라테스 문답]
                # -------------------------------------------------------------
                if not st.session_state.get("show_explanation") and not st.session_state.get("socratic_active"):
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("💡 정답 및 해설 바로 보기", use_container_width=True):
                            st.session_state.show_explanation = True
                            st.rerun()
                    with col2:
                        if st.button("🤔 소크라테스 문답으로 원리 깨닫기", type="primary", use_container_width=True):
                            st.session_state.socratic_active = True
                            st.rerun()

                # -------------------------------------------------------------
                #  루트 A: 정답/해설 바로 포기하고 보기
                # -------------------------------------------------------------
                if st.session_state.get("show_explanation"):
                    st.success(f"정답은 **{quiz.get('answer')}번** 입니다.")
                    st.info(f"💬 [AI 해설]: {quiz.get('explanation')}")
                    if st.button("➡️ 다음 문제 풀러가기", type="primary", use_container_width=True):
                        reset_quiz_state()
                        st.rerun()

                # -------------------------------------------------------------
                #  루트 B: 소크라테스 모드 발동 (채팅 UI)
                # -------------------------------------------------------------
                if st.session_state.get("socratic_active"):
                    st.write("---")
                    st.write("### 🏛️ AI 튜터와의 소크라테스 문답")
                    st.caption("강사가 던지는 꼬리 질문에 답하며 스스로 정답의 원리를 찾아보세요!")
                    
                    # 1. 채팅 기록 렌더링
                    for msg in st.session_state.socratic_history:
                        with st.chat_message(msg["role"]):
                            st.markdown(msg["content"])
                            
                    # 2. 대화가 진행 중일 때
                    if st.session_state.socratic_status == "continue":
                        # 최초 진입 시 AI의 첫 유도 질문 생성
                        if len(st.session_state.socratic_history) == 0:
                            with st.spinner("AI 튜터가 첫 번째 유도 질문을 준비 중입니다..."):
                                turn_res = tutor_engine.socratic_dialogue_turn(
                                    quiz.get("topic"), selected_cert, quiz.get("question"), 
                                    "", "학생이 소크라테스 모드를 시작했습니다. 첫 유도 질문을 던져주세요.", 0
                                )
                                st.session_state.socratic_history.append({"role": "assistant", "content": turn_res["question"]})
                                st.rerun()

                        # 학생 채팅 입력창
                        if user_ans := st.chat_input("튜터의 질문에 답해보세요..."):
                            # 화면에 즉시 학생 답변 추가
                            st.session_state.socratic_history.append({"role": "user", "content": user_ans})
                            
                            with st.spinner("답변을 분석하여 다음 꼬리 질문을 고민 중입니다..."):
                                # 문맥 기록 생성 (최근 6개 정도만 묶어서 넘겨도 됨)
                                history_str = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.socratic_history])
                                st.session_state.hint_level += 1
                                
                                # 다음 턴 평가
                                turn_res = tutor_engine.socratic_dialogue_turn(
                                    quiz.get("topic"), selected_cert, quiz.get("question"), 
                                    history_str, user_ans, st.session_state.hint_level
                                )
                                
                                st.session_state.socratic_status = turn_res["status"]
                                
                                # AI의 칭찬 멘트가 있으면 기록
                                if turn_res["acknowledgment"]:
                                    st.session_state.socratic_history.append({"role": "assistant", "content": f"💡 {turn_res['acknowledgment']}"})
                                
                                # 질문이 있으면 기록
                                if turn_res["status"] == "continue" and turn_res["question"]:
                                    st.session_state.socratic_history.append({"role": "assistant", "content": turn_res["question"]})
                                    
                            st.rerun()

                        # expanded=False 로 두어 평소에는 닫혀있게 만듦
                        st.write("---")
                        with st.expander("정답과 해설을 보고 다음 문제로 넘어가기(문답 포기)", expanded=False):
                            st.warning("포기하고 정답을 보시겠습니까? 튜터와 함께 끝까지 고민하는 과정이 가장 큰 학습입니다")
                            
                            
                            # 진짜 포기 버튼 
                            if st.button("네, 정답과 해설 보고 다음 문제로 넘어갈게요", type="primary", use_container_width=True, key="surrender_action_btn"):
                                st.session_state.socratic_active = False  # 1. 소크라테스 모드 강제 종료
                                st.session_state.show_explanation = True  # 2. 해설 보기 모드 On
                                st.rerun()                                
                                
                            st.markdown('</div>', unsafe_allow_html=True)

                            
                    # 3. 대화 종료
                    elif st.session_state.socratic_status == "complete":
                        st.success("🎉 훌륭합니다! 튜터의 질문에 답하며 스스로 정답의 원리를 완벽하게 깨달으셨네요!")
                        
                        # 이해 경로 맵 생성
                        if "understanding_map" not in st.session_state:
                            with st.spinner("대화 기록을 분석하여 '이해 경로 맵'을 생성 중입니다..."):
                                history_str = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.socratic_history])
                                umap = tutor_engine.generate_understanding_map(history_str)
                                st.session_state.understanding_map = umap
                        
                        umap = st.session_state.understanding_map
                        with st.expander("🗺️ 나의 이해 경로 맵 보기", expanded=True):
                            st.write(f"**🎯 최종 타겟 개념:** {umap['concept']}")
                            st.write(f"**🚧 내가 헷갈렸던 부분:** {', '.join(umap['struggle_points'])}")
                            st.write(f"**💡 깨달음의 순간:** {umap['breakthrough_moment']}")
                            st.write(f"**🔥 이해도(자신감):** {umap['confidence_level']}")
                            st.write(f"**➡️ 다음 권장 학습:** {umap['recommended_next']}")
                            
                        st.write("---")
                        st.info(f"💡 **객관식 오답 ai 해설**:\n\n{quiz.get('explanation')}")
                        if st.button("➡️ 완벽하게 이해했나요?. 다음 문제로 넘어가기", type="primary", use_container_width=True):
                            reset_quiz_state()
                            st.rerun()

elif st.session_state.mode == "mock_exam":
    st.header(f"⏱️ {selected_cert_label} 실전 모의고사")
    
    current_cert_topics = CERT_CONFIG[selected_cert]["topics"]

    # 멀티셀렉트에는 한글 과목명을 보여주기 위해 values()를 사용
    available_topics_kor = list(current_cert_topics.values())

    # 사용자가 선택한 한글 과목명을 다시 영문 topic key로 되돌리기 위한 매핑
    kor_to_eng_map = {topic_kor: topic_eng for topic_eng, topic_kor in current_cert_topics.items()}
    
    # ---------------------------------------------------------
    # [Step 1] 모의고사 조건 설정
    # ---------------------------------------------------------
    if st.session_state.mock_step == "setup":
        with st.container(border=True):
            st.subheader("⚙️ 모의고사 설정")
            selected_topics = st.multiselect("📚 응시할 과목을 선택하세요 (다중 선택 가능)", available_topics_kor, default=available_topics_kor)
            total_q_count = st.select_slider("📝 총 문제 수를 선택하세요", options=list(range(10, 101, 10)), value=20)
            
            if st.button("🚀 모의고사 시작 (문제 생성)", type="primary", use_container_width=True):
                if not selected_topics:
                    st.error("최소 1개 이상의 과목을 선택해야 합니다.")
                else:
                    # 🚀 [핵심 알고리즘] 과목별 문제 수 균등 분배
                    num_subjects = len(selected_topics)
                    base_count = total_q_count // num_subjects
                    remainder = total_q_count % num_subjects
                    
                    distribution = []
                    for i in range(num_subjects):
                        count = base_count + 1 if i < remainder else base_count
                        distribution.append({"topic_kor": selected_topics[i], "topic_eng": kor_to_eng_map[selected_topics[i]], "count": count})
                    
                    st.session_state.mock_distribution = distribution
                    st.session_state.mock_total_q = total_q_count
                    st.session_state.mock_step = "generating"
                    st.rerun()

    # ---------------------------------------------------------
    # [Step 2] 문제 일괄 생성 (Progress Bar)
    # ---------------------------------------------------------
    elif st.session_state.mock_step == "generating":
        st.info("AI 튜터가 지정된 조건에 맞춰 실전 문제를 출제하고 있습니다. 잠시만 기다려주세요...")
        progress_bar = st.progress(0)
        
        if len(st.session_state.mock_questions) == 0: # 아직 생성 안됨
            generated_questions = []
            total_to_gen = st.session_state.mock_total_q
            current_gen = 0
            
            for dist in st.session_state.mock_distribution:
                topic_eng = dist["topic_eng"]
                topic_kor = dist["topic_kor"]
                count = dist["count"]
                
                for _ in range(count):
                    max_retries = 3  # 똑같은 문제 나오면 최대 3번까지 LLM 다시 호출
                    is_success = False
                    
                    for attempt in range(max_retries):
                        # 1. 앞서 만든 문제 리스트(generated_questions)를 백엔드에 넘겨줌
                        quiz_data = tutor_engine.generate_advanced_quiz(
                            target_topic=topic_eng,
                            cert=selected_cert,
                            generated_history=generated_questions
                            )


                        new_signature = normalize_quiz_signature(quiz_data)
                        
                        # 2. 클라이언트 단 중복 검사 (새로 받은 문제가 이미 리스트에 있는지 텍스트 비교)
                        #  question만 보지 않고 options / code_block / table_data까지 합쳐 비교함
                        is_duplicate = any(
                            normalize_quiz_signature(existing_quiz) == new_signature
                            for existing_quiz in generated_questions
                            )
                        
                        if not is_duplicate:
                            # 중복이 아니면 정상 등록
                            quiz_data["topic_kor"] = topic_kor
                            generated_questions.append(quiz_data)
                            is_success = True
                            break  # 성공했으니 재시도(attempt) 루프 탈출
                        else:
                            print(f"⚠️ [경고] 중복 문제 발생. LLM 재호출 중... (시도 {attempt+1}/{max_retries})")
                    
                   
                    if not is_success:
                        quiz_data["topic_kor"] = topic_kor
                        generated_questions.append(quiz_data)
                        
                    # 진행률 바 업데이트
                    current_gen += 1
                    progress_bar.progress(current_gen / total_to_gen)
            
            st.session_state.mock_questions = generated_questions
            st.session_state.mock_step = "taking"
            st.rerun()

    # ---------------------------------------------------------
    # [Step 3] 모의고사 응시
    # ---------------------------------------------------------
    elif st.session_state.mock_step == "taking":
        curr_idx = st.session_state.mock_current_q
        total_q = st.session_state.mock_total_q
        current_quiz = st.session_state.mock_questions[curr_idx]
        
        st.progress((curr_idx) / total_q)
        st.markdown(f"### 문제 {curr_idx + 1} / {total_q}")
        st.caption(f"과목: {current_quiz['topic_kor']}")
        
        with st.container(border=True):
            st.write(current_quiz['question'])
            if current_quiz.get('code_block'):
                st.code(current_quiz['code_block'])
                
            # 라디오 버튼으로 사용자 답 입력 받기 (key를 동적으로 부여)
            user_choice = st.radio("정답을 선택하세요", current_quiz['options'], key=f"mock_radio_{curr_idx}")
            

        col1, col2 = st.columns(2)
        with col2:
            if curr_idx < total_q - 1:
                if st.button("다음 문제 ➡️", use_container_width=True, type="primary"):
                    if user_choice is None:
                        st.warning("보기를 선택한 후 다음 문제로 이동해주세요.")
                    else:
                        # 답 저장 후 인덱스 증가
                        selected_num = get_selected_option_number(user_choice, current_quiz["options"])
                        st.session_state.mock_answers.append(selected_num)
                        st.session_state.mock_current_q += 1
                        st.rerun()
            else:
                if st.button("✅ 최종 답안 제출 및 채점하기", use_container_width=True, type="primary"):
                    if user_choice is None:
                        st.warning("보기를 선택한 후 다음 문제로 제출해주세요.")
                    else:
                        selected_num = get_selected_option_number(user_choice, current_quiz["options"])
                        st.session_state.mock_answers.append(selected_num)
                        st.session_state.mock_step = "result"
                        st.rerun()

    # ---------------------------------------------------------
    # [Step 4] 채점 결과창
    # ---------------------------------------------------------
    elif st.session_state.mock_step == "result":
        st.success("🎉 모의고사가 종료되었습니다. 수고하셨습니다!")

        # 모의고사 결과를 DB에 1회만 반영
        if not st.session_state.mock_logged:
            for idx, q in enumerate(st.session_state.mock_questions):
                is_correct = (st.session_state.mock_answers[idx] == q["answer"])
                db_manager.log_quiz_result(selected_cert, q.get("topic"), is_correct)

            st.session_state.mock_logged = True

            # 통계 기반 노트 캐시가 있으면 삭제
            if "final_note_content" in st.session_state:
                del st.session_state["final_note_content"]
        
        # 1. 전체 채점 계산
        total_q = st.session_state.mock_total_q
        correct_count = 0
        subject_stats = {dist["topic_kor"]: {"total": 0, "correct": 0} for dist in st.session_state.mock_distribution}
        
        for idx, q in enumerate(st.session_state.mock_questions):
            subj = q["topic_kor"]
            is_correct = (st.session_state.mock_answers[idx] == q["answer"])
            
            subject_stats[subj]["total"] += 1
            if is_correct:
                correct_count += 1
                subject_stats[subj]["correct"] += 1
                
        # 2. 100점 환산 점수
        final_score = int((correct_count / total_q) * 100)
        
        st.markdown(f"<h2 style='text-align: center; color: #FF4B4B;'>총점: {final_score}점 / 100점</h2>", unsafe_allow_html=True)
        st.markdown(f"<p style='text-align: center;'>({total_q}문제 중 {correct_count}문제 정답)</p>", unsafe_allow_html=True)
        st.write("---")
        
        # 3. 과목별 상세 성적
        st.subheader("📊 과목별 성적표")
        cols = st.columns(len(subject_stats))
        for i, (subj, stats) in enumerate(subject_stats.items()):
            with cols[i]:
                st.metric(subj, f"{stats['correct']} / {stats['total']}")
        
        st.write("---")
        if st.button("🔍 푼 문제 복기하기 (오답 확인)", use_container_width=True, type="primary"):
            st.session_state.mock_step = "review"
            st.rerun()

    # ---------------------------------------------------------
    # [Step 5] 푼 문제 복기 기능
    # ---------------------------------------------------------
    elif st.session_state.mock_step == "review":
        st.subheader("🔍 모의고사 복기")
        
        # 필터링 기능
        filter_opt = st.radio("문제 필터", ["전체 보기", "맞춘 문제만", "틀린 문제만"], horizontal=True)
        
        for idx, q in enumerate(st.session_state.mock_questions):
            user_ans = st.session_state.mock_answers[idx]
            real_ans = q["answer"]
            is_correct = (user_ans == real_ans)
            
            # 필터 적용
            if filter_opt == "맞춘 문제만" and not is_correct: continue
            if filter_opt == "틀린 문제만" and is_correct: continue
            
            icon = "✅" if is_correct else "❌"
            expander_title = f"{icon} 문제 {idx + 1} [{q['topic_kor']}]"
            
            with st.expander(expander_title, expanded=not is_correct): # 틀린 문제는 기본으로 열어둠
                st.markdown(f"**Q. {q['question']}**")
                if q.get('code_block'):
                    st.code(q['code_block'])
                
                st.write("**[선택지]**")
                for opt in q['options']:
                    st.write(f"- {opt}")
                    
                st.markdown("---")
                colA, colB = st.columns(2)

                user_ans_num = st.session_state.mock_answers[idx]
                real_ans_num = q["answer"]

                user_ans_text = q["options"][user_ans_num - 1] if 1 <= user_ans_num <= len(q["options"]) else "선택 없음"
                real_ans_text = q["options"][real_ans_num - 1] if 1 <= real_ans_num <= len(q["options"]) else "정답 데이터 오류"

                colA.info(f"🙋‍♂️ 내가 선택한 답:\n**{user_ans_num}번. {user_ans_text}**")
                colB.success(f"🎯 실제 정답:\n**{real_ans_num}번. {real_ans_text}**")
                
                st.markdown(f"💬 **AI 해설:**\n{q.get('explanation', '해설이 없습니다.')}")
                
        if st.button("돌아가기 (모의고사 종료)"):
            st.session_state.mode = "study" # 기본 학습 모드로 복귀
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


st.markdown(
    """
    <div style='text-align: center; color: #888888; font-size: 11px; padding-bottom: 20px;'>
        © 2026 자격증 변형 문제 출제 AI 튜터 시스템 (AI Tutor Project Team). All Rights Reserved.<br>
        본 시스템은 캡스톤 디자인 졸업과제용으로 제작되었으며, 무단 복제 및 전재를 금합니다.<br>
        -자료 출처 :
        시나공 기출문제집 정보처리기사 필기(길벗알앤디 지음/ 강윤석, 김용갑, 감우경, 김종일) / 
        국가법령정보센터
    </div>
    """, 
    unsafe_allow_html=True
)