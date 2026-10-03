import { useEffect, useRef, useState } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { apiRequest } from './api';
import { downloadSummary } from './summaryDownload';

export function SummaryContent({ label, data, loading, error, busy, onReload, onGenerate, onDownload }) {
  const notes = data?.notes || [];
  const generating = busy || data?.generating;
  return <section className="wrongnote-page"><div className="wrongnote-content">
    <div className="wrongnote-header">
      <div><span className="wrongnote-page-label">CONCEPT SUMMARY</span><h1>취약점 개념 요약</h1>
        <p>풀이 기록에서 확인된 취약 과목의 핵심 개념을 정리합니다.</p></div>
      <div className="wrongnote-header-actions">
        <button type="button" className="wrongnote-download-button" disabled={!notes.length} onClick={onDownload}>요약 다운로드</button>
        <button type="button" className="wrongnote-download-button" disabled={loading || generating} onClick={onReload}>다시 조회</button>
      </div>
    </div>
    {loading && <p role="status">저장된 요약과 학습 기록을 불러오고 있습니다.</p>}
    {error && <p role="alert">{error}</p>}
    {data && <>
      <section className="wrongnote-focus-section">
        <div className="wrongnote-section-heading"><h2>현재 취약 과목</h2><span className="wrongnote-cert-chip">{label}</span></div>
        <p>{data.weakness.message}</p>
        <p>과목별 {data.weakness.min_attempts}문제 이상 · 정답률 {data.weakness.accuracy_threshold}% 미만 기준</p>
        <div className="wrongnote-top-grid">{data.weakness.ranking.map((item, index) =>
          <div className="wrongnote-top-item" key={item.topic}>
            <span className="wrongnote-top-rank">{index + 1}</span>
            <div className="wrongnote-top-info"><strong>{item.label}</strong><p>{item.correct}/{item.total}문제 정답</p></div>
            <div className="wrongnote-top-score">{item.accuracy}%</div>
          </div>)}</div>
        <button type="button" className="wrongnote-download-button" disabled={loading || generating || !data.can_generate || !data.needs_update} onClick={onGenerate}>
          {generating ? '요약 생성 중…' : notes.length ? '최신 기록으로 갱신' : '개념 요약 생성'}
        </button>
        {generating && <p role="status">과목별 개념 요약을 준비하고 있습니다. 저장된 요약은 계속 볼 수 있습니다.</p>}
        {data.needs_update && !generating && <p>새 학습 기록이나 자료를 반영해 요약을 생성·갱신할 수 있습니다.</p>}
        {!data.needs_update && data.can_generate && <p>현재 기준의 요약이 저장되어 있습니다.</p>}
      </section>
      {data.issues.map(issue => <p role="alert" key={issue.topic}>{issue.label}: {issue.message}</p>)}
      {!notes.length && !generating && <p>아직 저장된 개념 요약이 없습니다.</p>}
      <div className="wrongnote-note-grid">{notes.map(note => <article className="wrongnote-card" key={note.id}>
        <div className="wrongnote-card-header"><div><span className="wrongnote-card-label">CONCEPT REVIEW</span><h2>{note.label}</h2></div></div>
        <p>생성 시각: {new Date(note.created_at).toLocaleString('ko-KR', { timeZone: 'Asia/Seoul' })}</p>
        <p>생성 당시 {note.basis.correct}/{note.basis.total}문제 정답 · {note.basis.accuracy}%</p>
        {!note.is_current_topic && <p>이전 학습 기록으로 생성한 요약입니다. 현재 취약 과목에는 포함되지 않습니다.</p>}
        {note.stale && note.is_current_topic && <p>갱신이 필요한 이전 요약입니다.</p>}
        <div className="summary-markdown"><ReactMarkdown remarkPlugins={[remarkGfm]}>{note.markdown}</ReactMarkdown></div>
      </article>)}</div>
    </>}
  </div></section>;
}

export default function SummaryNotes({ cert, label, connectionError, onReconnect }) {
  const [data, setData] = useState(null);
  const [error, setError] = useState('');
  const [loaded, setLoaded] = useState(-1);
  const [revision, setRevision] = useState(0);
  const [busy, setBusy] = useState(false);
  const generation = useRef(null);
  useEffect(() => {
    if (!cert) return;
    const controller = new AbortController();
    apiRequest(`/api/summary-notes?cert=${encodeURIComponent(cert)}`, { signal: controller.signal })
      .then(result => {
        if (!controller.signal.aborted) { setData(result); setError(''); }
      }).catch(err => {
        if (!controller.signal.aborted) setError(err.message);
      }).finally(() => {
        if (!controller.signal.aborted) setLoaded(revision);
      });
    return () => controller.abort();
  }, [cert, revision]);
  useEffect(() => () => generation.current?.abort(), []);
  useEffect(() => {
    if (!data?.generating || busy) return;
    const timer = setTimeout(() => setRevision(value => value + 1), 4000);
    return () => clearTimeout(timer);
  }, [data, busy, revision]);
  const generate = async () => {
    if (!cert || generation.current || !data?.can_generate || !data.needs_update) return;
    const controller = new AbortController();
    generation.current = controller;
    setBusy(true);
    setError('');
    try {
      const result = await apiRequest('/api/summary-notes', { signal: controller.signal, body: { cert } });
      if (!controller.signal.aborted) setData(result);
    } catch (err) {
      if (!controller.signal.aborted) {
        setError(err.message);
        if (err.code === 'SUMMARY_BUSY' || err.code === 'NO_SUMMARY_TARGET') setRevision(value => value + 1);
      }
    } finally {
      if (!controller.signal.aborted) { generation.current = null; setBusy(false); }
    }
  };
  return <SummaryContent label={label} data={data}
    loading={cert ? loaded !== revision : !connectionError} error={error || connectionError} busy={busy}
    onGenerate={generate} onReload={() => { if (!cert) onReconnect(); setRevision(value => value + 1); }}
    onDownload={() => downloadSummary(label, data?.notes || [])} />;
}
