import { createPortal } from 'react-dom';
import { useEffect, useMemo, useRef, useState } from 'react';
import * as api from '../api/lessonClient.js';
import LessonIndex from '../components/LessonIndex.jsx';
import LecturePane from '../components/LecturePane.jsx';
import { DEFAULT_VARIANT } from '../components/lectureVariants.js';
import TutorPane from '../components/TutorPane.jsx';
import TutorDrawer from '../components/TutorDrawer.jsx';
import { ChevronLeft, ChevronRight, ListIcon, PencilIcon, ChatIcon } from '../components/icons.jsx';
import '../learning-soft.css';

const LEARNER = '내 학습';
const CERT = 'EIP';
const cache = new Map();
const emptyLecture = () => ({ message: null, error: false, loading: false });
const cacheKey = lesson => `${lesson.id}:${lesson.version}`;

// tutorChat: 사이드바 'AI 튜터'와 공유하는 대화 (App.jsx에서 전달)
export default function LearningPage({ returnLessonId, onLessonChange, onPractice, tutorChat, headerTitleTarget, headerActionsTarget }) {
  const [catalog, setCatalog] = useState(null);
  const [loadError, setLoadError] = useState('');
  const [reload, setReload] = useState(0);
  const [states, setStates] = useState({});
  const [progressError, setProgressError] = useState('');
  const [saving, setSaving] = useState(false);
  const [activeId, setActiveId] = useState(returnLessonId || null);
  const activeIdRef = useRef(returnLessonId || null);
  const [lecture, setLecture] = useState(emptyLecture);
  const [tutorOpen, setTutorOpen] = useState(false);
  const visit = useRef(0);
  const inflight = useRef(new Map());
  const progressLock = useRef(false);
  const alive = useRef(false);
  const pendingSave = useRef(null);
  const progressRevision = useRef(0);
  const serverProgressRevision = useRef(null);
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
      const saved = restored ? cache.get(cacheKey(restored)) : null;
      setLecture({ ...emptyLecture(), message: saved || null });
      setCatalog(data); setLoadError('');
    }).catch(e => { if (!controller.signal.aborted) setLoadError(e.message); });
    api.getProgress(LEARNER, controller.signal).then(data => {
      if (!controller.signal.aborted && revision === progressRevision.current) {
        serverProgressRevision.current = data.resetRevision;
        setStates(data.states); setProgressError('');
      }
    }).catch(e => { if (!controller.signal.aborted && revision === progressRevision.current) setProgressError(e.message); });
    return () => controller.abort();
  }, [reload]);

  const changeLesson = lesson => {
    setTutorOpen(false);
    visit.current += 1;
    inflight.current.forEach(controller => controller.abort()); inflight.current.clear();
    activeIdRef.current = lesson?.id || null;
    setActiveId(lesson?.id || null); onLessonChange?.(lesson?.id || null);
    setLecture({ ...emptyLecture(), message: lesson ? cache.get(cacheKey(lesson)) || null : null });
    document.querySelector('.concept-learning')?.scrollTo(0, 0);
  };

  const saveAndMove = async (lesson, complete = false) => {
    if (!lesson || progressLock.current) return;
    progressRevision.current += 1;
    progressLock.current = true; setSaving(true); setProgressError('');
    pendingSave.current = { lesson, complete };
    try {
      if (serverProgressRevision.current === null) {
        const snapshot = await api.getProgress(LEARNER);
        if (!alive.current) return;
        serverProgressRevision.current = snapshot.resetRevision;
      }
      const data = await api.saveProgress(LEARNER, lesson.id, complete, serverProgressRevision.current);
      if (!alive.current) return;
      setStates(data.states); pendingSave.current = null;
      if (complete) {
        const next = lessons[lessons.indexOf(lesson) + 1] || null;
        if (next) {
          pendingSave.current = { lesson: next, complete: false };
          const nextStates = await api.saveProgress(LEARNER, next.id, false, serverProgressRevision.current);
          if (!alive.current) return;
          setStates(nextStates.states); pendingSave.current = null;
        }
        changeLesson(next);
      } else changeLesson(lesson);
    } catch (e) {
      if (alive.current) {
        setProgressError(e.message);
        if (e.code === 'PROGRESS_RESET') {
          pendingSave.current = null; serverProgressRevision.current = null;
          changeLesson(null); setReload(value => value + 1);
        }
      }
    }
    finally { progressLock.current = false; if (alive.current) setSaving(false); }
  };

  const loadLecture = async () => {
    if (!active) return;
    const version = visit.current;
    const key = `${version}:${active.id}`;
    if (inflight.current.has(key)) return;
    const stored = cache.get(cacheKey(active));
    if (stored) { setLecture({ ...emptyLecture(), message: stored }); return; }
    const controller = new AbortController(); inflight.current.set(key, controller);
    setLecture({ message: null, error: false, loading: true });
    try {
      const data = await api.teach(active.id, DEFAULT_VARIANT, controller.signal);
      if (!alive.current || version !== visit.current) return;
      const message = { role: 'assistant', ...data };
      cache.set(cacheKey(active), message);
      setLecture({ message, error: false, loading: false });
    } catch (e) {
      if (!alive.current || version !== visit.current || e.name === 'AbortError') return;
      setLecture({ message: null, error: true, loading: false });
    } finally { inflight.current.delete(key); }
  };

  useEffect(() => {
    if (active && !lecture.message && !lecture.error && !lecture.loading) {
      const timer = setTimeout(loadLecture, 0);
      return () => clearTimeout(timer);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [active?.id, active?.version, lecture.message, lecture.error, lecture.loading]);

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
  const isLast = active && index === lessons.length - 1;
  const percent = Math.round(done / lessons.length * 100);

  const progressAlert = progressError && <div role="alert" className="notice">학습 진도를 저장하거나 불러오지 못했습니다. {progressError}
    <button className="btn" disabled={saving} onClick={() => pendingSave.current ? saveAndMove(pendingSave.current.lesson, pendingSave.current.complete) : setReload(n => n + 1)}>다시 시도</button></div>;

  // 단원 목차 화면
  if (!active) return <div className="learning"><div className="learning-main">
    {headerTitleTarget ? createPortal(<><span className="learning-header-separator" aria-hidden="true">–</span><h1 className="learning-header-page-title">정보처리기사 개념 학습</h1></>, headerTitleTarget) : <h1 className="page-title">정보처리기사 개념 학습</h1>}
    {headerActionsTarget && createPortal(<span className="learning-header-progress" aria-label={`학습 진도 ${lessons.length}개 중 ${done}개 완료`}>학습 진도 {done} / {lessons.length} · {percent}%</span>, headerActionsTarget)}
    {progressAlert}
    <fieldset disabled={saving} style={{ border: 0, padding: 0, margin: 0 }}>
      <LessonIndex config={catalog.config} cert={CERT} lessons={lessons} states={states} onOpen={saveAndMove} />
      <div className="learning-index-footer">
        <button type="button" className="btn btn-ghost" onClick={() => saveAndMove(lessons[0])}>처음부터</button>
        <button type="button" className="btn btn-ghost" disabled={!last} onClick={() => saveAndMove(last)}>이어서 학습</button>
      </div>
    </fieldset>
  </div></div>;

  // 단원 학습 화면
  return <div className="learning is-presenting"><div className="learning-main ls-lesson">
    {headerTitleTarget && createPortal(<><span className="learning-header-separator" aria-hidden="true">–</span><h1 className="learning-header-page-title">정보처리기사 개념 학습</h1></>, headerTitleTarget)}
    <header className="ls-header">
      <div className="ls-title">
        <div className="ls-ring" style={{ '--ls-progress': `${percent}%` }} role="img" aria-label={`학습 진도 ${percent}%`}>
          <span>{percent}%</span>
        </div>
        <div>
          <h1>{active.title}</h1>
          <span className="ls-sub">{done} / {lessons.length}</span>
        </div>
      </div>
      <div className="ls-header-actions">
        <button className="ls-pill ls-tutor-trigger" type="button" onClick={() => setTutorOpen(true)}><ChatIcon />튜터에게 질문</button>
        <button className="ls-pill" type="button" disabled={saving} onClick={() => onPractice(active)}><PencilIcon />문제 풀기</button>
        <button className="ls-pill" type="button" disabled={saving} onClick={() => changeLesson(null)}><ListIcon />목차</button>
      </div>
    </header>

    {progressAlert}

    <LecturePane message={lecture.message} error={lecture.error} onRetry={loadLecture} />

    <TutorDrawer open={tutorOpen} onClose={() => setTutorOpen(false)}>
      <TutorPane key={active.id} tab={DEFAULT_VARIANT} lessonTitle={active.title} chat={tutorChat} />
    </TutorDrawer>

    <nav className="ls-unit-nav" aria-label="단원 이동">
      <button className="ls-unit" type="button" disabled={saving || index === 0} onClick={() => saveAndMove(lessons[index - 1])}>
        <ChevronLeft />이전 단원
      </button>
      <button className="ls-unit is-next" type="button" disabled={saving} onClick={() => saveAndMove(active, true)}>
        {saving ? '저장 중…' : isLast ? '학습 마치기' : '다음 단원'}<ChevronRight />
      </button>
    </nav>
  </div></div>;
}