import { useEffect, useMemo, useRef, useState } from 'react';
import { useAuth } from '../context/AuthContext.jsx';
import * as api from '../api/api.js';
import Sidebar from '../components/Sidebar.jsx';
import LessonIndex from '../components/LessonIndex.jsx';
import LecturePane, { VARIANT_PROMPTS, DEFAULT_VARIANT } from '../components/LecturePane.jsx';
import TutorPane from '../components/TutorPane.jsx';
import PracticePanel from '../components/PracticePanel.jsx';
import ProgressBar from '../components/ProgressBar.jsx';

// 단원을 새로 열 때마다 강의/질문 상태를 비움
const emptyLecture = () => ({
  tab: DEFAULT_VARIANT,
  variants: {}, // 탭별 설명 { '기본 설명': message, ... }
  errors: {}, // 탭별 실패 여부
  loading: {}, // 탭별 로딩 여부
  messages: [], // 튜터 질문/답변
  tutorLoading: false,
  tutorError: false,
  pending: null, // 다시 요청할 질문
});

export default function LearningPage() {
  const { user } = useAuth();
  const learner = user.email; // 파이썬의 '학습 프로필' 대신 로그인 계정으로 진도 저장

  const [catalog, setCatalog] = useState(null);
  const [loadError, setLoadError] = useState(false);
  const [cert, setCert] = useState('');
  const [states, setStates] = useState({});
  const [activeId, setActiveId] = useState(null);
  const [practice, setPractice] = useState(false);
  const [practiceLesson, setPracticeLesson] = useState(null);
  const [lecture, setLecture] = useState(emptyLecture);

  const visitRef = useRef(0); // 단원이 바뀌면 이전 요청 결과는 버림
  const inflightRef = useRef(new Set()); // 같은 설명을 두 번 요청하지 않도록

  useEffect(() => {
    api
      .getCatalog()
      .then((data) => {
        setCatalog(data);
        const first = Object.keys(data.config).find((c) => data.lessons.some((l) => l.cert === c));
        setCert(first ?? '');
      })
      .catch(() => setLoadError(true));
  }, []);

  useEffect(() => {
    api.getProgress(learner).then(setStates).catch(() => setStates({}));
  }, [learner]);

  const certs = useMemo(
    () => (catalog ? Object.keys(catalog.config).filter((c) => catalog.lessons.some((l) => l.cert === c)) : []),
    [catalog]
  );
  const lessons = useMemo(
    () => (catalog ? catalog.lessons.filter((l) => l.cert === cert) : []),
    [catalog, cert]
  );
  const active = lessons.find((l) => l.id === activeId) ?? null;
  const done = lessons.filter((l) => states[l.id]?.status === 'complete').length;
  const last = useMemo(() => {
    let best = null;
    for (const l of lessons) {
      const s = states[l.id];
      if (s && (!best || s.updatedAt > states[best.id].updatedAt)) best = l;
    }
    return best;
  }, [lessons, states]);

  const closeLesson = () => {
    visitRef.current += 1;
    setActiveId(null);
  };

  const changeCert = (next) => {
    setCert(next);
    setPracticeLesson(null);
    closeLesson();
  };

  const openLesson = async (lesson) => {
    visitRef.current += 1;
    setActiveId(lesson.id);
    setLecture(emptyLecture());
    window.scrollTo(0, 0);
    setStates(await api.saveProgress(learner, lesson.id));
  };

  const loadVariant = async (label) => {
    const visit = visitRef.current;
    const key = `${visit}:${label}`;
    if (inflightRef.current.has(key)) return;
    inflightRef.current.add(key);
    setLecture((p) => ({ ...p, loading: { ...p.loading, [label]: true }, errors: { ...p.errors, [label]: false } }));
    try {
      const data = await api.teach(active.id, VARIANT_PROMPTS[label]);
      if (visit !== visitRef.current) return;
      setLecture((p) => ({
        ...p,
        loading: { ...p.loading, [label]: false },
        variants: { ...p.variants, [label]: { role: 'assistant', ...data } },
      }));
    } catch {
      if (visit !== visitRef.current) return;
      setLecture((p) => ({ ...p, loading: { ...p.loading, [label]: false }, errors: { ...p.errors, [label]: true } }));
    } finally {
      inflightRef.current.delete(key);
    }
  };

  // 탭을 열었는데 설명이 없으면 자동으로 요청
  useEffect(() => {
    if (!active) return;
    const { tab, variants, errors, loading } = lecture;
    if (!variants[tab] && !errors[tab] && !loading[tab]) loadVariant(tab);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [active?.id, lecture.tab, lecture.variants, lecture.errors, lecture.loading]);

  const askTutor = async (req) => {
    const visit = visitRef.current;
    const history = lecture.messages
      .slice(-6)
      .map((m) => `${m.role}: ${m.text}`)
      .join('\n');
    const context =
      `${req.question}\n[현재 선택한 설명 방식]\n${req.variant}` +
      `\n[현재 보고 있는 강의: 참고용]\n${req.explanation}` +
      `\n[이전 질문 대화: 참고용]\n${history}`;

    setLecture((p) => ({ ...p, pending: req, tutorLoading: true, tutorError: false }));
    try {
      const data = await api.teach(active.id, context);
      if (visit !== visitRef.current) return;
      setLecture((p) => ({
        ...p,
        tutorLoading: false,
        pending: null,
        messages: [...p.messages, { role: 'user', text: req.question }, { role: 'assistant', ...data }],
      }));
    } catch {
      if (visit !== visitRef.current) return;
      setLecture((p) => ({ ...p, tutorLoading: false, tutorError: true }));
    }
  };

  const completeAndNext = async () => {
    setStates(await api.saveProgress(learner, active.id, true));
    const i = lessons.indexOf(active);
    if (i + 1 < lessons.length) openLesson(lessons[i + 1]);
    else closeLesson();
  };

  if (loadError) return <p className="notice">학습 목록을 불러오지 못했습니다. 새로고침해 주세요.</p>;
  if (!catalog) return <p className="muted">학습 목록을 불러오고 있어요…</p>;
  if (certs.length === 0) return <p className="notice">준비된 강의가 없습니다.</p>;

  const { config } = catalog;
  const ratio = lessons.length ? done / lessons.length : 0;
  const index = active ? lessons.indexOf(active) : -1;
  const currentExplanation = lecture.variants[lecture.tab];

  return (
    <div className="learning">
      <Sidebar
        config={config}
        certs={certs}
        cert={cert}
        onCertChange={changeCert}
        done={done}
        total={lessons.length}
        onOpenPractice={() => {
          setPracticeLesson(null);
          setPractice(true);
        }}
      />

      <div className="learning-main">
        {practice ? (
          <PracticePanel cert={cert} lesson={practiceLesson} onBack={() => setPractice(false)} />
        ) : (
          <>
            <h1 className="page-title">{config[cert].label} 개념 학습</h1>
            <p className="muted">개념을 이해하고, 문제를 풀며 실력을 확인하세요.</p>
            <ProgressBar value={ratio} label={`${Math.round(ratio * 100)}% · ${done} / ${lessons.length} 완료`} />

            <div className="row three">
              <button type="button" className="btn btn-ghost" onClick={() => openLesson(lessons[0])}>
                처음부터 공부하기
              </button>
              <button type="button" className="btn btn-ghost" onClick={closeLesson}>
                특정 단원 선택
              </button>
              <button type="button" className="btn btn-ghost" disabled={!last} onClick={() => openLesson(last)}>
                이어서 공부하기
              </button>
            </div>

            {active ? (
              <>
                <p className="muted small">{config[cert].topics[active.subject] ?? active.subject}</p>
                <h2 className="lesson-title">{active.title}</h2>

                <div className="lesson-grid">
                  <LecturePane
                    tab={lecture.tab}
                    onTabChange={(tab) => setLecture((p) => ({ ...p, tab }))}
                    variants={lecture.variants}
                    errors={lecture.errors}
                    loading={lecture.loading}
                    onRetry={loadVariant}
                  />
                  <TutorPane
                    tab={lecture.tab}
                    messages={lecture.messages}
                    canAsk={Boolean(currentExplanation)}
                    loading={lecture.tutorLoading}
                    error={lecture.tutorError}
                    onAsk={(question) =>
                      askTutor({ question, variant: lecture.tab, explanation: currentExplanation.text })
                    }
                    onRetry={() => lecture.pending && askTutor(lecture.pending)}
                  />
                </div>

                <div className="row three lesson-nav">
                  <button
                    type="button"
                    className="btn btn-ghost"
                    disabled={index === 0}
                    onClick={() => openLesson(lessons[index - 1])}
                  >
                    이전 항목
                  </button>
                  <button type="button" className="btn btn-primary" onClick={completeAndNext}>
                    학습 완료하고 다음으로
                  </button>
                  <button
                    type="button"
                    className="btn btn-ghost"
                    onClick={() => {
                      setPracticeLesson(active);
                      setPractice(true);
                    }}
                  >
                    이 단원 문제 풀기
                  </button>
                </div>
              </>
            ) : (
              <LessonIndex config={config} cert={cert} lessons={lessons} states={states} onOpen={openLesson} />
            )}
          </>
        )}
      </div>
    </div>
  );
}