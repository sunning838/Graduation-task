import { useEffect, useRef, useState } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { apiRequest } from './api';
import { downloadSummary } from './summaryDownload';

export function SummaryContent({ label, data, loading, error, busy, onReload, onGenerate, onDownload }) {
  const notes = data?.notes || [];
  const ranking = data?.weakness?.ranking || [];
  const generating = busy || data?.generating;
  const [selectedTopic, setSelectedTopic] = useState('');

  useEffect(() => {
    if (!data) return;

    const currentTopics = ranking.map(item => item.topic);
    if (!selectedTopic || !currentTopics.includes(selectedTopic)) {
      setSelectedTopic(currentTopics[0] || notes[0]?.topic || notes[0]?.label || '');
    }
  }, [data, ranking, notes, selectedTopic]);

  const selectedItem = ranking.find(item => item.topic === selectedTopic) || ranking[0] || null;
  const selectedNote =
    notes.find(note => note.topic === selectedTopic) ||
    notes.find(note => selectedItem && note.label === selectedItem.label) ||
    null;

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
        <div className="wrongnote-top-grid">{ranking.map((item, index) =>
          <button
            type="button"
            className={`wrongnote-top-item ${selectedTopic === item.topic ? 'active' : ''}`}
            key={item.topic}
            onClick={() => setSelectedTopic(item.topic)}
            aria-pressed={selectedTopic === item.topic}
          >
            <span className="wrongnote-top-rank">{index + 1}</span>
            <div className="wrongnote-top-info">
              <strong>{item.label}</strong>
              <p>{item.correct}/{item.total}문제 정답</p>
            </div>
            <div className="wrongnote-top-score">{item.accuracy}%</div>
          </button>)}</div>
        <button type="button" className="wrongnote-download-button" disabled={loading || generating || !data.can_generate || !data.needs_update} onClick={onGenerate}>
          {generating ? '요약 생성 중…' : notes.length ? '최신 기록으로 갱신' : '개념 요약 생성'}
        </button>
        {generating && <p role="status">과목별 개념 요약을 준비하고 있습니다. 저장된 요약은 계속 볼 수 있습니다.</p>}
        {data.needs_update && !generating && <p>새 학습 기록이나 자료를 반영해 요약을 생성·갱신할 수 있습니다.</p>}
        {!data.needs_update && data.can_generate && <p>현재 기준의 요약이 저장되어 있습니다.</p>}
      </section>
      {data.issues.map(issue => <p role="alert" key={issue.topic}>{issue.label}: {issue.message}</p>)}
      {!notes.length && !generating && <p>아직 저장된 개념 요약이 없습니다.</p>}

      {selectedItem && <section className="wrongnote-selected-note">
        <div className="wrongnote-selected-note-heading">
          <div>
            <span className="wrongnote-card-label">SELECTED CONCEPT NOTE</span>
            <h2>{selectedItem.label}</h2>
          </div>
          <span className="wrongnote-selected-accuracy">
            현재 정답률 {selectedItem.accuracy}%
          </span>
        </div>

        {selectedNote ? <article className="wrongnote-card wrongnote-card-selected" key={selectedNote.id}>
          <div className="wrongnote-note-meta">
            <span>
              생성 시각 {new Date(selectedNote.created_at).toLocaleString('ko-KR', { timeZone: 'Asia/Seoul' })}
            </span>
            <span>
              생성 당시 {selectedNote.basis.correct}/{selectedNote.basis.total}문제 정답 · {selectedNote.basis.accuracy}%
            </span>
          </div>

          {!selectedNote.is_current_topic &&
            <p className="wrongnote-note-warning">이전 학습 기록으로 생성한 요약입니다. 현재 취약 과목에는 포함되지 않습니다.</p>}
          {selectedNote.stale && selectedNote.is_current_topic &&
            <p className="wrongnote-note-warning">새 학습 기록이 있어 갱신이 필요한 요약입니다.</p>}

          <div className="summary-markdown summary-markdown-selected">
            <ReactMarkdown remarkPlugins={[remarkGfm]}>{selectedNote.markdown}</ReactMarkdown>
          </div>
        </article> : <div className="wrongnote-empty-selected">
          <strong>{selectedItem.label} 개념 노트가 아직 없습니다.</strong>
          <p>위의 개념 요약 생성 또는 최신 기록으로 갱신 버튼을 사용해 요약을 준비할 수 있습니다.</p>
        </div>}
      </section>}
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
