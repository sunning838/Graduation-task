import logging
import sys
from urllib.parse import urlencode
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import streamlit as st
from backend.learning import catalog, Progress, teach, diagram
from backend.lesson_display import learner_text, display_title

st.set_page_config(page_title='자격증 AI 학습실', page_icon='📚', layout='wide')

# Readable line length and typography, scoped to the learning screen.
st.markdown("""
<style>
[data-testid="stMainBlockContainer"] {
    width: 100%;
    max-width: 1720px;
    padding: 1.5rem 2rem 3rem;
}
[data-testid="stChatMessage"] {
    padding: 1rem;
    border-radius: 16px;
    margin-bottom: 1rem;
}
[data-testid="stChatMessageContent"] {
    min-width: 0;
}
[data-testid="stChatMessageContent"] [data-testid="stMarkdownContainer"] p,
[data-testid="stChatMessageContent"] [data-testid="stMarkdownContainer"] li {
    font-size: 1.0625rem;
    line-height: 1.9;
    word-break: keep-all;
    overflow-wrap: anywhere;
}
[data-testid="stChatMessageContent"] [data-testid="stMarkdownContainer"] p {
    margin-bottom: 1.1rem;
}
[data-testid="stChatMessageContent"] [data-testid="stMarkdownContainer"] h3 {
    font-size: 1.35rem;
    margin-top: 1.25rem;
    padding-bottom: .55rem;
    border-bottom: 1px solid rgba(128,128,128,.25);
}
[data-testid="stChatMessageContent"] [data-testid="stMarkdownContainer"] blockquote {
    border-left: 3px solid #4f83cc;
    padding: .8rem 1rem;
    background: rgba(79,131,204,.07);
    margin: 1.1rem 0 1.5rem;
}
[data-testid="stChatMessageContent"] [data-testid="stMarkdownContainer"] > :first-child {
    margin-top: 0;
}
@media (max-width: 640px) {
    [data-testid="stMainBlockContainer"] { padding: 1.2rem 1rem; }
    [data-testid="stChatMessage"] { padding: .8rem; }
}
</style>
""", unsafe_allow_html=True)

(ROOT / 'logs').mkdir(exist_ok=True)
logging.basicConfig(filename=ROOT / 'logs/developer.log', level=logging.WARNING, encoding='utf-8')

config, all_lessons = catalog()
progress = Progress()
st.sidebar.title('📚 자격증 AI 학습실')
learner = st.sidebar.text_input('학습 프로필', '내 학습', help='같은 이름으로 들어오면 이전 진도를 이어갑니다. 로그인 기능은 아닙니다.').strip() or '내 학습'
available = [c for c in config if any(l['cert'] == c for l in all_lessons)]
if not available:
    st.info('준비된 강의가 없습니다.')
    st.stop()
cert = st.sidebar.selectbox('자격증', available, format_func=lambda c: config[c]['label'])
lessons = [l for l in all_lessons if l['cert'] == cert]
states = progress.states(learner, lessons)
done = sum(states.get(l['id'], ('', 0))[0] == 'complete' for l in lessons)
st.sidebar.progress(done / len(lessons), text=f'전체 {done} / {len(lessons)} 학습 완료')
st.sidebar.caption('진행도는 학습한 분량이며 문제 정답률과 별개입니다.')
if st.sidebar.button('실전 문제풀이 열기', use_container_width=True):
    st.session_state.practice = True
    st.session_state.practice_lesson = None
if st.session_state.get('practice'):
    st.title('실전 문제풀이')
    st.write('기존 문제풀이·모의고사·오답노트 화면에서 실전 학습을 진행하세요.')
    selected = st.session_state.get('practice_lesson')
    params = {'cert': cert}
    if selected and selected['cert'] == cert:
        params['lesson'] = selected['id']
        st.info(f"학습 범위: {display_title(selected['title'])}")
    st.link_button('문제풀이 시작', 'http://localhost:8512/?' + urlencode(params))
    if st.button('개념 학습으로 돌아가기'):
        st.session_state.practice = False
        st.rerun()
    st.stop()

st.title(f"{config[cert]['label']} 개념 학습")
st.write('개념을 이해하고, 문제를 풀며 실력을 확인하세요.')
st.progress(done / len(lessons), text=f'{done / len(lessons):.0%} · {done} / {len(lessons)} 완료')
scope = f'{learner}:{cert}'
if st.session_state.get('scope') != scope:
    st.session_state.scope = scope
    st.session_state.active_lesson = None
    st.session_state.lesson_output = None
    st.session_state.practice_lesson = None

def open_lesson(lesson):
    st.session_state.active_lesson = lesson['id']
    st.session_state.lesson_output = None
    st.session_state.lecture_visit = None
    progress.save(learner, lesson)
    st.rerun()

a, b, c = st.columns(3)
if a.button('처음부터 공부하기', use_container_width=True):
    open_lesson(lessons[0])
if b.button('특정 단원 선택', use_container_width=True):
    st.session_state.active_lesson = None
    st.rerun()
last = max(states, key=lambda key: states[key][1]) if states else None
if c.button('이어서 공부하기', disabled=last is None, use_container_width=True):
    open_lesson(next(l for l in lessons if l['id'] == last))

active = next((l for l in lessons if l['id'] == st.session_state.get('active_lesson')), None)
if active is None:
    st.subheader('학습 목차')
    subjects = list(dict.fromkeys(l['subject'] for l in lessons))
    for subject in subjects:
        group = [l for l in lessons if l['subject'] == subject]
        complete = sum(states.get(l['id'], ('', 0))[0] == 'complete' for l in group)
        label = config[cert]['topics'].get(subject, subject)
        with st.expander(f'{label} · {complete}/{len(group)} 완료', expanded=len(subjects) == 1):
            st.progress(complete / len(group))
            selection = st.selectbox('학습 항목', [l['id'] for l in group], key=f'unit_{cert}_{subject}', format_func=lambda key, group=group: ('✓ ' if states.get(key, ('', 0))[0] == 'complete' else '◌ ') + next(display_title(l['title']) for l in group if l['id'] == key))
            if st.button('선택한 단원 공부하기', key=f'open_{subject}'):
                open_lesson(next(l for l in group if l['id'] == selection))
    st.stop()

st.caption(config[cert]['topics'].get(active['subject'], active['subject']))
st.header(display_title(active['title']))
# Cache explanation variants for this lesson visit; keep direct questions separate.
visit = (scope, active['id'])
if st.session_state.get('lecture_visit') != visit:
    st.session_state.lecture_visit = visit
    st.session_state.lecture_messages = []
    st.session_state.lecture_variants = {}
    st.session_state.variant_errors = {}
    st.session_state.lecture_tab = '기본 설명'
    st.session_state.tutor_error = False
    st.session_state.tutor_pending = None
# Migrate an already open session without discarding its original lecture.
if 'lecture_variants' not in st.session_state:
    st.session_state.lecture_variants = {}
    if st.session_state.get('lecture_messages'):
        st.session_state.lecture_variants['기본 설명'] = st.session_state.lecture_messages[0]
if 'variant_errors' not in st.session_state:
    st.session_state.variant_errors = {}

variant_prompts = {
    '기본 설명': '처음 배우는 학생에게 이 학습 항목의 첫 강의를 시작해 주세요.',
    '쉬운 설명': '현재 단원의 개념 전체를 쉬운 용어와 짧은 단계로 다시 설명하세요. 낯선 용어를 먼저 풀고 기본 설명의 정확한 조건은 유지하세요. 독립적으로 읽을 수 있는 강의로 작성하세요.',
    '예시로 이해하기': '현재 단원을 하나의 구체적인 사례로 단계별 설명하세요. 사례의 각 부분과 해당 개념을 명시적으로 연결하세요. 예시임을 표시하고 원문 조건을 유지하세요. 독립적인 사례 강의로 작성하세요.',
}
variant_request = None
request = None

def render_message(message):
    with st.chat_message(message['role']):
        st.markdown(learner_text(message['text']) if message['role'] == 'assistant' else message['text'])
        if not message.get('data'):
            return
        visual = diagram(message['data'])
        if visual is not None and not isinstance(visual, dict):
            logging.warning('Skipping legacy visual result: %s', type(visual).__name__)
            visual = None
        if isinstance(visual, dict) and visual.get('kind') in ('table', 'graph'):
            st.subheader(display_title(visual['title']))
            st.caption(visual['purpose'])
            if visual['kind'] == 'table':
                import pandas as pd
                st.table(pd.DataFrame(visual['rows'], columns=visual['columns']))
            else:
                st.graphviz_chart(visual['dot'])
        research = message['data'].get('research', {})
        if research.get('status') == 'supplemented':
            st.caption('웹 검색 참고')
            if hasattr(st, 'iframe'):
                st.iframe(research['suggestions_html'], height='content')
            else:
                import streamlit.components.v1 as components
                components.html(research['suggestions_html'], height=180, scrolling=True)
            for source in research.get('sources', []):
                st.link_button(source['title'], source['url'])

lecture_pane, tutor_pane = st.columns([1.65, 1], gap='large')
with lecture_pane:
    st.subheader('개념 강의')
    tabs = st.tabs(list(variant_prompts), key='lecture_tab', on_change='rerun')
    selected_variant = st.session_state.lecture_tab
    for label, tab in zip(variant_prompts, tabs):
        if not tab.open:
            continue
        with tab:
            with st.container(height=650, border=True):
                saved = st.session_state.lecture_variants.get(label)
                if saved:
                    render_message(saved)
                elif st.session_state.variant_errors.get(label):
                    st.info('설명을 준비하지 못했습니다. 다시 시도해 주세요.')
                    if st.button('설명 다시 준비하기', key='retry_' + label):
                        variant_request = label
                else:
                    variant_request = label
                    st.caption('설명을 준비하고 있습니다.')
                lecture_status = st.empty()

current_explanation = st.session_state.lecture_variants.get(selected_variant)
with tutor_pane:
    st.subheader('튜터에게 질문')
    st.caption(f'현재 보고 있는 설명: {selected_variant}')
    with st.container(height=530, border=True):
        if len(st.session_state.lecture_messages) <= 1:
            st.caption('직접 입력한 질문과 튜터의 답변이 여기에 쌓입니다.')
        for message in st.session_state.lecture_messages[1:]:
            render_message(message)
        tutor_status = st.empty()
    if st.session_state.get('tutor_error'):
        st.info('답변을 준비하지 못했습니다. 다시 시도해 주세요.')
        if st.button('답변 다시 요청'):
            request = st.session_state.get('tutor_pending')
    question = st.chat_input('이 부분이 궁금해요…', key='tutor_question_' + active['id'], disabled=not current_explanation)
    if question:
        request = {'question': question, 'variant': selected_variant, 'explanation': current_explanation['text']}

if variant_request:
    try:
        with lecture_status.container():
            with st.spinner(f'{variant_request}을 준비하고 있어요…'):
                data, docs = teach(active, all_lessons, variant_prompts[variant_request])
        message = dict(role='assistant', text=data['text'], data=data, docs=docs)
        st.session_state.lecture_variants[variant_request] = message
        st.session_state.variant_errors.pop(variant_request, None)
        if variant_request == '기본 설명' and not st.session_state.lecture_messages:
            st.session_state.lecture_messages = [message]
    except Exception:
        logging.exception('Lecture variant failed: %s / %s', active['id'], variant_request)
        st.session_state.variant_errors[variant_request] = True
    st.rerun()

if request:
    # Snapshot the visible explanation so a retry cannot accidentally use another tab.
    if not isinstance(request, dict):
        request = {'question': request, 'variant': selected_variant, 'explanation': current_explanation['text'] if current_explanation else ''}
    st.session_state.tutor_pending = request
    history = '\n'.join(m['role'] + ': ' + m['text'] for m in st.session_state.lecture_messages[1:][-6:])
    context = request['question'] + '\n[현재 선택한 설명 방식]\n' + request['variant'] + '\n[현재 보고 있는 강의: 참고용]\n' + request['explanation'] + '\n[이전 질문 대화: 참고용]\n' + history
    try:
        with tutor_status.container():
            with st.spinner('답변을 준비하고 있어요…'):
                data, docs = teach(active, all_lessons, context)
        # Reserve the first slot for the original lecture even if a different tab was opened first.
        if not st.session_state.lecture_messages:
            st.session_state.lecture_messages = [st.session_state.lecture_variants.get('기본 설명', current_explanation)]
        st.session_state.lecture_messages.append(dict(role='user', text=request['question']))
        st.session_state.lecture_messages.append(dict(role='assistant', text=data['text'], data=data, docs=docs))
        st.session_state.tutor_error = False
        st.session_state.tutor_pending = None
    except Exception:
        logging.exception('Tutor answer failed: %s', active['id'])
        st.session_state.tutor_error = True
    st.rerun()

st.divider()
previous, following, practice = st.columns(3)
index = lessons.index(active)
if previous.button('이전 항목', disabled=index == 0):
    open_lesson(lessons[index - 1])
if following.button('학습 완료하고 다음으로', type='primary'):
    progress.save(learner, active, complete=True)
    if index + 1 < len(lessons):
        open_lesson(lessons[index + 1])
    else:
        st.session_state.active_lesson = None
        st.rerun()
if practice.button('이 단원 문제 풀기'):
    st.session_state.practice = True
    st.session_state.practice_lesson = active
    st.query_params.clear()
    st.rerun()
