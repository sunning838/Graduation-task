import { useEffect, useRef, useState } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { apiRequest } from './api';
import './MockExam.css';

function distribute(subjects, count) {
  if (!subjects.length) return [];
  return subjects.map((topic, i) => ({ topic, count: Math.floor(count / subjects.length) + Number(i < count % subjects.length) })).filter(p => p.count > 0);
}

function readSaved(key) {
  try { return JSON.parse(localStorage.getItem(key) || 'null'); } catch { return null; }
}

export function ExamQuestion({ item, submitted, busy, onAnswer }) {
  return <article className="bank-question">
    <p>{item.position + 1}번 · {item.topic_label}</p>
    <ReactMarkdown remarkPlugins={[remarkGfm]}>{item.question}</ReactMarkdown>
    {item.code_block && <pre><code>{item.code_block}</code></pre>}
    {item.table_data && <ReactMarkdown remarkPlugins={[remarkGfm]}>{item.table_data}</ReactMarkdown>}
    <div className="bank-options">{item.options.map((option, i) => <button type="button" key={i}
      disabled={busy || submitted} aria-pressed={item.selected_answer === i + 1}
      className={item.selected_answer === i + 1 ? 'chosen' : ''} onClick={() => onAnswer(i + 1)}>
      <span>{i + 1}.</span><ReactMarkdown remarkPlugins={[remarkGfm]}>{option.replace(/^\s*\d+[).:]\s+/, '')}</ReactMarkdown>
    </button>)}</div>
    {submitted && <div className="bank-explanation"><strong>{item.is_correct ? '정답' : '오답'} · 내 답: {item.selected_answer ?? '미응답'} · 정답: {item.correct_answer}</strong>
      <ReactMarkdown remarkPlugins={[remarkGfm]}>{item.explanation}</ReactMarkdown></div>}
  </article>;
}

export function PreparationProgress({ preparation }) {
  const progress = preparation.progress;
  return <div role="status" className="bank-confirm" aria-live="polite">
    <h2>모의고사 준비 중</h2>
    {progress && <>
      <p><strong>총 {progress.total}문항 중 {progress.ready}문항 준비 완료</strong> · {progress.remaining}문항 남음</p>
      <progress aria-label="시험 문제 준비 진행률" max={progress.total} value={progress.ready} style={{ width: '100%', height: '18px' }} />
      <p>{progress.percent}% 완료</p>
    </>}
    <p>{preparation.message}</p>
    <p>기존에 준비된 문제와 생성·검수를 마친 문제를 합산합니다. 준비가 끝나면 시험이 자동으로 시작됩니다.</p>
    <p>처음 준비할 때는 시간이 걸릴 수 있습니다. 새로고침 후에도 진행 상태를 확인할 수 있습니다.</p>
  </div>;
}

export default function MockExam({ cert, label, onSubmitted, onWeakness }) {
  const key = `mock-exam:${cert}`;
  const [bank, setBank] = useState(null);
  const [selected, setSelected] = useState([]);
  const [count, setCount] = useState(20);
  const [exam, setExam] = useState(null);
  const [index, setIndex] = useState(0);
  const [loading, setLoading] = useState(true);
  const [requestBusy, setBusy] = useState(false);
  const [preparation, setPreparation] = useState(null);
  const busy = requestBusy || Boolean(preparation);
  const [error, setError] = useState('');
  const [reload, setReload] = useState(0);
  const [confirm, setConfirm] = useState(false);
  const lock = useRef(false);
  const mounted = useRef(false);

  useEffect(() => {
    mounted.current = true;
    const controller = new AbortController();
    async function load() {
      try {
        if (!cert) throw new Error('자격증 목록을 먼저 불러와 주세요.');
        const data = await apiRequest(`/api/question-bank/availability?cert=${encodeURIComponent(cert)}`, { signal: controller.signal });
        const saved = readSaved(key);
        const preparing = saved?.pending ? await apiRequest('/api/mock-preparations', { body: { cert, ...saved.pending }, signal: controller.signal }) : null;
        const restored = saved?.id ? await apiRequest(`/api/mock-exams/${saved.id}`, { signal: controller.signal }) : null;
        if (restored && restored.cert !== cert) throw new Error('저장된 시험의 자격증이 일치하지 않습니다.');
        if (!controller.signal.aborted) {
          setBank(data); setSelected(saved?.pending?.distribution?.map(p => p.topic) || data.subjects.map(s => s.topic));
          if (saved?.pending?.distribution) setCount(saved.pending.distribution.reduce((sum, p) => sum + p.count, 0));
          setExam(restored); setPreparation(preparing); setIndex(0); setError('');
        }
      } catch (e) { if (!controller.signal.aborted) setError(e.message); }
      finally { if (!controller.signal.aborted) setLoading(false); }
    }
    load();
    return () => { mounted.current = false; controller.abort(); };
  }, [cert, key, reload]);

  const preparationId = preparation?.request_id;
  useEffect(() => {
    if (!preparationId) return;
    const controller = new AbortController();
    let timer;
    async function poll() {
      try {
        const state = await apiRequest(`/api/mock-preparations/${preparationId}`, { signal: controller.signal });
        if (controller.signal.aborted) return;
        if (state.status === 'completed') {
          const result = await apiRequest(`/api/mock-exams/${state.exam_id}`, { signal: controller.signal });
          if (controller.signal.aborted) return;
          localStorage.setItem(key, JSON.stringify({ id: result.id }));
          setExam(result); setIndex(0); setPreparation(null); setError('');
          return;
        }
        if (state.status === 'failed') {
          localStorage.removeItem(key);
          setPreparation(null); setError(state.message);
          return;
        }
        setPreparation(state);
        setError('');
      } catch (e) {
        if (controller.signal.aborted) return;
        setError(e.message);
      }
      timer = setTimeout(poll, 2500);
    }
    poll();
    return () => { controller.abort(); clearTimeout(timer); };
  }, [preparationId, key]);

  async function action(fn) {
    if (lock.current) return;
    lock.current = true; setBusy(true); setError('');
    try { await fn(); } catch (e) {
      if (mounted.current) setError(e.code === 'INSUFFICIENT_QUESTIONS'
        ? '선택한 시험을 위한 문제가 아직 준비되지 않았습니다. 나중에 다시 확인해 주세요.'
        : e.message);
    }
    finally { lock.current = false; if (mounted.current) setBusy(false); }
  }
  const distribution = distribute((bank?.subjects || []).filter(s => selected.includes(s.topic)).map(s => s.topic), count);
  const validCount = Number.isInteger(count) && count >= 1 && count <= 100;
  const start = () => action(async () => {
    const saved = readSaved(key);
    const pending = saved?.pending && JSON.stringify(saved.pending.distribution) === JSON.stringify(distribution)
      ? saved.pending : { request_id: crypto.randomUUID(), distribution };
    // Persist the request before POST so a lost response can be retried safely.
    localStorage.setItem(key, JSON.stringify({ pending }));
    const result = await apiRequest('/api/mock-preparations', { body: { cert, ...pending } });
    if (mounted.current) setPreparation(result);
  });
  const answer = value => action(async () => {
    const item = exam.items[index];
    const result = await apiRequest(`/api/mock-exams/${exam.id}/answers/${item.id}`, { method: 'PUT', body: { selected_answer: value } });
    if (mounted.current) setExam(current => ({ ...current, items: current.items.map(q => q.id === item.id ? { ...q, selected_answer: result.selected_answer } : q) }));
  });
  const submit = () => action(async () => {
    const result = await apiRequest(`/api/mock-exams/${exam.id}/submit`, { body: { confirm_unanswered: true } });
    if (mounted.current) { setExam(result); setConfirm(false); onSubmitted(); }
  });
  const reset = () => {
    localStorage.removeItem(key); setLoading(true); setConfirm(false); setExam(null); setPreparation(null); setReload(n => n + 1);
  };
  const submitted = exam?.status === 'submitted';
  const unanswered = exam?.items.filter(q => q.selected_answer == null).length || 0;
  return <section className="mock-page"><div className="mock-content bank-exam">
    <h1>{label} 모의고사</h1>
    {error && <div role="alert"><p>{error}</p>{!busy && <button onClick={() => { setLoading(true); setReload(n => n + 1); }}>서버 상태 다시 확인</button>}
      {!bank && !busy && !loading && <button onClick={reset}>저장된 시험 연결 해제</button>}</div>}
    {loading ? <p role="status">시험 정보를 불러오는 중…</p> : !bank ? <button onClick={() => { setLoading(true); setReload(n => n + 1); }}>다시 불러오기</button> : preparation ? <PreparationProgress preparation={preparation} /> : !exam ? <>
      <p>과목과 문항 수를 선택하세요. 필요한 문제는 AI가 준비하며, 준비가 끝나면 시험이 시작됩니다.</p>
      <label>전체 문항 수 <input type="number" min="1" max="100" value={count} disabled={busy} onChange={e => setCount(Number(e.target.value))} /></label>
      <div className="bank-subjects">{bank.subjects.map(s => <label key={s.topic}>
        <input type="checkbox" checked={selected.includes(s.topic)} disabled={busy} onChange={() => setSelected(current => current.includes(s.topic) ? current.filter(t => t !== s.topic) : [...current, s.topic])} />
        {s.label} · 이번 시험 {distribution.find(p => p.topic === s.topic)?.count || 0}개
      </label>)}</div>
      <p>선택한 과목에 문항을 고르게 배분합니다. 같은 시험에는 같은 문제가 중복되지 않습니다.</p>
      <button disabled={busy || !validCount || !distribution.length} onClick={start}>{busy ? '시험 구성 중…' : '모의고사 시작'}</button>
    </> : <>
      {submitted ? <div className="bank-result"><h2>결과: {exam.result.score}점</h2>
        <p>{exam.result.total}문항 중 {exam.result.correct}개 정답 · 미응답 {exam.result.unanswered}개</p>
        <p>선택한 문제 기준 점수입니다. 학습 현황과 취약점 분석에 반영했습니다.</p>
        <ul>{exam.result.subjects.map(s => <li key={s.topic}>{s.label}: {s.correct}/{s.total} · 정답률 {s.accuracy}%</li>)}</ul>
        <button onClick={onWeakness}>취약점 학습</button><button onClick={reset}>새 모의고사</button>
      </div> : <p>총 {exam.items.length}문항 · 답안 저장 {exam.items.length - unanswered}개 · 미응답 {unanswered}개 {busy && '· 저장 중…'}</p>}
      <nav className="bank-map" aria-label="문제 이동">{exam.items.map((q, i) => <button key={q.id} disabled={busy} aria-current={index === i ? 'step' : undefined}
        className={q.selected_answer != null ? 'answered' : ''} onClick={() => setIndex(i)}>{i + 1}{submitted ? (q.is_correct ? ' ✓' : ' ×') : q.selected_answer != null ? ' •' : ''}</button>)}</nav>
      <ExamQuestion item={exam.items[index]} submitted={submitted} busy={busy} onAnswer={answer} />
      <div className="bank-actions"><button disabled={busy || index === 0} onClick={() => setIndex(i => i - 1)}>이전</button>
        <button disabled={busy || index === exam.items.length - 1} onClick={() => setIndex(i => i + 1)}>다음</button>
        {!submitted && <><button disabled={busy || exam.items[index].selected_answer == null} onClick={() => answer(null)}>답안 지우기</button>
          <button disabled={busy} onClick={() => setConfirm(true)}>시험 제출</button></>}
      </div>
      {confirm && !submitted && <div className="bank-confirm" role="alertdialog" aria-label="시험 제출 확인">
        <p>{unanswered ? `미응답 ${unanswered}문항은 오답으로 처리됩니다. ` : ''}제출 후에는 답안을 바꿀 수 없습니다.</p>
        <button disabled={busy} onClick={submit}>{busy ? '제출 중…' : '확인하고 제출'}</button><button disabled={busy} onClick={() => setConfirm(false)}>계속 풀기</button>
      </div>}
    </>}
  </div></section>;
}
