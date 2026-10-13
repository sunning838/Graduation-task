import { useEffect, useMemo, useRef, useState } from 'react';
import * as api from '../api/lessonClient.js';
import LessonIndex from '../components/LessonIndex.jsx';
import LecturePane, { DEFAULT_VARIANT } from '../components/LecturePane.jsx';
import TutorPane from '../components/TutorPane.jsx';
import ProgressBar from '../components/ProgressBar.jsx';

const LEARNER = '내 학습';
const CERT = 'EIP';
const cache = new Map();
const emptyLecture = () => ({ tab: DEFAULT_VARIANT, variants: {}, errors: {}, loading: {}, messages: [], tutorLoading: false, tutorError: false, pending: null });
const cacheKey = lesson => `${lesson.id}:${lesson.version}`;

export default function LearningPage({ returnLessonId, onLessonChange, onPractice }) {
  const [catalog, setCatalog] = useState(null);
  const [loadError, setLoadError] = useState('');
  const [reload, setReload] = useState(0);
  const [states, setStates] = useState({});
  const [progressError, setProgressError] = useState('');
  const [saving, setSaving] = useState(false);
  const [activeId, setActiveId] = useState(returnLessonId || null);
  const activeIdRef = useRef(returnLessonId || null);
  const [lecture, setLecture] = useState(emptyLecture);
  const visit = useRef(0);
  const inflight = useRef(new Map());
  const progressLock = useRef(false);
  const tutorLock = useRef(false);
  const alive = useRef(false);
  const pendingSave = useRef(null);
  const progressRevision = useRef(0);
  const lessons = useMemo(() => catalog?.lessons || [], [catalog]);
  const active = lessons.find(l => l.id === activeId);
  const last = lessons.reduce((best, l) => states[l.id] && (!best || states[l.id].updatedAt > states[best.id].updatedAt) ? l : best, null);
  const done = lessons.filter(l => states[l.id]?.status === 'complete').length;

  useEffect(() => {
    alive.current = true;
    const requests = inflight.current;
    return () => {
      alive.current = false; visit.current += 1;
      requests.forEach(controller => controller.abort()); requests.clear();
    };
  }, []);

  useEffect(() => {
    const controller = new AbortController();
    const revision = progressRevision.current;
    api.getCatalog(controller.signal).then(data => {
      if (controller.signal.aborted) return;
      const restored = data.lessons.find(l => l.id === activeIdRef.current);
      setLecture({ ...emptyLecture(), ...(restored ? cache.get(cacheKey(restored)) : {}) });
      setCatalog(data); setLoadError('');
    }).catch(e => { if (!controller.signal.aborted) setLoadError(e.message); });
    api.getProgress(LEARNER, controller.signal).then(data => {
      if (!controller.signal.aborted && revision === progressRevision.current) { setStates(data); setProgressError(''); }
    }).catch(e => { if (!controller.signal.aborted && revision === progressRevision.current) setProgressError(e.message); });
    return () => controller.abort();
  }, [reload]);

  const changeLesson = lesson => {
    visit.current += 1;
    inflight.current.forEach(controller => controller.abort()); inflight.current.clear();
    tutorLock.current = false;
    activeIdRef.current = lesson?.id || null;
    setActiveId(lesson?.id || null); onLessonChange?.(lesson?.id || null);
    setLecture(lesson ? { ...emptyLecture(), ...cache.get(cacheKey(lesson)) } : emptyLecture());
    document.querySelector('.concept-learning')?.scrollTo(0, 0);
  };

  const saveAndMove = async (lesson, complete = false) => {
    if (!lesson || progressLock.current) return;
    progressRevision.current += 1;
    progressLock.current = true; setSaving(true); setProgressError('');
    pendingSave.current = { lesson, complete };
    try {
      const data = await api.saveProgress(LEARNER, lesson.id, complete);
      if (!alive.current) return;
      setStates(data); pendingSave.current = null;
      if (complete) {
        const next = lessons[lessons.indexOf(lesson) + 1] || null;
        if (next) {
          pendingSave.current = { lesson: next, complete: false };
          const nextStates = await api.saveProgress(LEARNER, next.id, false);
          if (!alive.current) return;
          setStates(nextStates); pendingSave.current = null;
        }
        changeLesson(next);
      } else changeLesson(lesson);
    } catch (e) { if (alive.current) setProgressError(e.message); }
    finally { progressLock.current = false; if (alive.current) setSaving(false); }
  };

  const loadVariant = async label => {
    if (!active) return;
    const version = visit.current;
    const key = `${version}:${active.id}:${label}`;
    if (inflight.current.has(key)) return;
    const stored = cache.get(cacheKey(active));
    if (stored?.variants[label]) {
      setLecture(p => ({ ...p, variants: { ...p.variants, [label]: stored.variants[label] } })); return;
    }
    const controller = new AbortController(); inflight.current.set(key, controller);
    setLecture(p => ({ ...p, loading: { ...p.loading, [label]: true }, errors: { ...p.errors, [label]: false } }));
    try {
      const data = await api.teach(active.id, label, controller.signal);
      if (!alive.current || version !== visit.current) return;
      const message = { role: 'assistant', ...data };
      const previous = cache.get(cacheKey(active)) || { variants: {}, messages: [] };
      cache.set(cacheKey(active), { ...previous, variants: { ...previous.variants, [label]: message } });
      setLecture(p => ({ ...p, loading: { ...p.loading, [label]: false }, variants: { ...p.variants, [label]: message } }));
    } catch (e) {
      if (!alive.current || version !== visit.current || e.name === 'AbortError') return;
      setLecture(p => ({ ...p, loading: { ...p.loading, [label]: false }, errors: { ...p.errors, [label]: true } }));
    } finally { inflight.current.delete(key); }
  };

  useEffect(() => {
    if (active && cache.get(cacheKey(active))?.tab && cache.get(cacheKey(active)).tab !== lecture.tab) return;
    if (active && !lecture.variants[lecture.tab] && !lecture.errors[lecture.tab] && !lecture.loading[lecture.tab]) {
      const timer = setTimeout(() => loadVariant(lecture.tab), 0);
      return () => clearTimeout(timer);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [active?.id, active?.version, lecture.tab, lecture.variants, lecture.errors, lecture.loading]);

  const askTutor = async req => {
    if (!active || tutorLock.current) return;
    tutorLock.current = true;
    const version = visit.current, key = `${version}:tutor`;
    const controller = new AbortController(); inflight.current.set(key, controller);
    setLecture(p => ({ ...p, pending: req, tutorLoading: true, tutorError: false }));
    try {
      const data = await api.teach(active.id, req.variant, controller.signal, { ...req, history: lecture.messages });
      if (!alive.current || version !== visit.current) return;
      const messages = [...lecture.messages, { role: 'user', text: req.question }, { role: 'assistant', ...data }];
      const previous = cache.get(cacheKey(active)) || { variants: {} };
      cache.set(cacheKey(active), { ...previous, messages });
      setLecture(p => ({ ...p, messages, pending: null, tutorLoading: false }));
    } catch (e) {
      if (alive.current && version === visit.current && e.name !== 'AbortError') setLecture(p => ({ ...p, tutorLoading: false, tutorError: true }));
    } finally {
      inflight.current.delete(key);
      if (version === visit.current) tutorLock.current = false;
    }
  };

  const changeTab = tab => {
    setLecture(p => ({ ...p, tab }));
    if (active) {
      const previous = cache.get(cacheKey(active)) || { variants: {}, messages: [] };
      cache.set(cacheKey(active), { ...previous, tab });
    }
  };

  if (loadError) return <div className="notice">{loadError}<button className="btn" onClick={() => setReload(n => n + 1)}>다시 불러오기</button></div>;
 if (!catalog)
  return (
    <div className="learn-loading">
      <div className="quiz-loading">
        <div className="quiz-loading-spinner" />
        <h3>학습 목록을 불러오고 있어요</h3>
      </div>
    </div>
  );
  if (!lessons.length) return <p className="notice">준비된 강의가 없습니다.</p>;
  const index = active ? lessons.indexOf(active) : -1;
  const explanation = lecture.variants[lecture.tab];
  return <div className="learning"><div className="learning-main">
    <h1 className="page-title">정보처리기사 개념 학습</h1>
    <p className="muted">개념을 이해하고, 문제를 풀며 실력을 확인하세요.</p>
    {progressError && <div role="alert" className="notice">학습 진도를 저장하거나 불러오지 못했습니다. {progressError}
      <button className="btn" disabled={saving} onClick={() => pendingSave.current ? saveAndMove(pendingSave.current.lesson, pendingSave.current.complete) : setReload(n => n + 1)}>다시 시도</button></div>}
    <ProgressBar value={done / lessons.length} label={`${Math.round(done / lessons.length * 100)}% · ${done} / ${lessons.length} 완료`} />
    <fieldset disabled={saving} style={{ border: 0, padding: 0, margin: 0 }}>
      <div className="row three">
        <button className="btn btn-ghost" onClick={() => saveAndMove(lessons[0])}>처음부터 공부하기</button>
        <button className="btn btn-ghost" onClick={() => changeLesson(null)}>특정 단원 선택</button>
        <button className="btn btn-ghost" disabled={!last} onClick={() => saveAndMove(last)}>이어서 공부하기</button>
      </div>
    </fieldset>
    {active ? <>
      <p className="muted small">{catalog.config[CERT].topics[active.subject] || active.subject}</p>
      <h2 className="lesson-title">{active.title}</h2>
      <div className="lesson-grid">
        <LecturePane tab={lecture.tab} onTabChange={changeTab} variants={lecture.variants} errors={lecture.errors} loading={lecture.loading} onRetry={loadVariant} />
        <TutorPane key={active.id} tab={lecture.tab} messages={lecture.messages} canAsk={Boolean(explanation?.text)} loading={lecture.tutorLoading} error={lecture.tutorError}
          onAsk={question => askTutor({ question, variant: lecture.tab, explanation: explanation.text })} onRetry={() => lecture.pending && askTutor(lecture.pending)} />
      </div>
      <div className="row three lesson-nav">
        <button className="btn btn-ghost" disabled={saving || index === 0} onClick={() => saveAndMove(lessons[index - 1])}>이전 항목</button>
        <button className="btn btn-primary" disabled={saving} onClick={() => saveAndMove(active, true)}>{saving ? '진도 저장 중…' : '학습 완료하고 다음으로'}</button>
        <button className="btn btn-ghost" disabled={saving} onClick={() => onPractice(active)}>이 단원 문제 풀기</button>
      </div>
    </> : <fieldset disabled={saving} style={{ border: 0, padding: 0, margin: 0 }}><LessonIndex config={catalog.config} cert={CERT} lessons={lessons} states={states} onOpen={saveAndMove} /></fieldset>}
  </div></div>;
}
