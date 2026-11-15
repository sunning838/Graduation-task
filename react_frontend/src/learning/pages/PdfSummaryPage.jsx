import { useEffect, useRef, useState } from 'react';
import PdfResult from '../components/PdfResult.jsx';
import { summarizePdf } from '../api/api.js';
import { readPdfSummaryHistory, savePdfSummary } from '../api/pdfSummaryHistory.js';
import usePdfTutor from '../api/usePdfTutor.js';
import '../learning-soft.css';
import './pdf-summary.css';

// 답변 길이 (AI 튜터 화면의 설정과 같은 값)
const LENGTHS = [
  { id: 'short', label: '간단히' },
  { id: 'medium', label: '보통' },
  { id: 'long', label: '자세히' },
];
const summaryDate = new Intl.DateTimeFormat('ko-KR', {
  dateStyle: 'medium',
  timeStyle: 'short',
});

function PdfIcon() {
  return <svg viewBox="0 0 64 72" fill="none" aria-hidden="true"><path d="M12 3h27l15 15v51H12zM39 3v16h15M22 31h22M22 39h22M22 47h14" stroke="currentColor" strokeWidth="2.5" strokeLinejoin="round" /><rect x="3" y="52" width="39" height="17" rx="4" fill="currentColor" /><text x="22" y="64" textAnchor="middle" fill="var(--bg-main)" fontSize="11" fontWeight="700">PDF</text></svg>;
}

export default function PdfSummaryPage({ answerLength, onAnswerLengthChange }) {
  const inputRef = useRef(null);
  const summaryRequest = useRef(null);
  const [file, setFile] = useState(null);
  const [error, setError] = useState('');
  const [dragging, setDragging] = useState(false);
  const [result, setResult] = useState(null);
  const [recent, setRecent] = useState([]);
  const [historyError, setHistoryError] = useState('');
  const [loading, setLoading] = useState(false);
  const tutorChat = usePdfTutor(result, answerLength);

  useEffect(() => {
    let active = true;
    readPdfSummaryHistory().then(records => {
      if (active) setRecent(current => current.length ? current : records);
    }).catch(() => {
      if (active) setHistoryError('저장된 요약을 불러오지 못했습니다.');
    });
    return () => { active = false; };
  }, []);

  useEffect(() => () => summaryRequest.current?.abort(), []);

  const selectFile = files => {
    setDragging(false);
    if (summaryRequest.current || !files?.length) return;
    if (files.length !== 1 || !/\.pdf$/i.test(files[0].name)) {
      setFile(null); setError('PDF 파일 한 개를 선택해 주세요.'); return;
    }
    if (!files[0].size) { setFile(null); setError('비어 있는 파일입니다. 다른 PDF를 선택해 주세요.'); return; }
    setFile(files[0]); setError('');
  };
  const reset = () => {
    if (summaryRequest.current) return;
    setFile(null);
    setError('');

    if (inputRef.current) {
      inputRef.current.value = '';
    }
  };

  const extractSummary = async () => {
    if (!file || summaryRequest.current) return;
    const controller = new AbortController();
    summaryRequest.current = controller;
    setLoading(true);
    setError('');

    try {
      const data = await summarizePdf(file, answerLength, controller.signal);
      if (controller.signal.aborted) return;
      const record = {
        id: crypto.randomUUID(),
        file,
        summary: data.summary,
        sourcePages: data.source_pages,
        createdAt: Date.now(),
      };
      try {
        const records = await savePdfSummary(record);
        if (controller.signal.aborted) return;
        setRecent(records);
        setHistoryError('');
      } catch {
        if (controller.signal.aborted) return;
        setRecent(records => [record, ...records].slice(0, 3));
        setHistoryError('브라우저에 저장하지 못했습니다. 이 창을 열어 둔 동안 요약을 다시 볼 수 있습니다.');
      }
      setResult(record);
    } catch (err) {
      if (!controller.signal.aborted) setError(err.message || 'PDF 요약을 생성하지 못했습니다.');
    } finally {
      if (summaryRequest.current === controller) {
        summaryRequest.current = null;
        if (!controller.signal.aborted) setLoading(false);
      }
    }
  };

  const openSummary = record => {
    if (summaryRequest.current) return;
    setFile(record.file);
    setError('');
    setResult(record);
  };

  if (result) return <section className="pdf-summary-page is-result" aria-label="PDF 요약">
    <PdfResult key={result.id} file={result.file} summary={result.summary} onBack={() => setResult(null)} tutorChat={tutorChat} />
  </section>;

  // 처음 화면: PDF 올리기
  return <section className="pdf-summary-page" aria-labelledby="pdf-summary-title">
    <div className="pdf-summary-inner">
      <header className="pdf-summary-heading"><span className="pdf-demo-badge">미리보기</span><h1 id="pdf-summary-title">PDF 요약</h1><p>학습 자료의 핵심을 한눈에 살펴보세요.</p></header>
      <div className={`pdf-dropzone${dragging ? ' is-dragging' : ''}`} onDragOver={event => { event.preventDefault(); setDragging(true); }} onDragLeave={event => { if (!event.currentTarget.contains(event.relatedTarget)) setDragging(false); }} onDrop={event => { event.preventDefault(); selectFile(event.dataTransfer.files); }}>
        <PdfIcon />
        <input ref={inputRef} type="file" accept=".pdf,application/pdf" hidden aria-label="PDF 파일 선택" onChange={event => { selectFile(event.target.files); event.target.value = ''; }} />
        <button
          type="button"
          className="pdf-select-button"
          disabled={loading}
          onClick={() => inputRef.current?.click()}
        >
          {file ? '다른 파일 선택' : '파일 선택'}
          <span aria-hidden="true">＋</span>
        </button>
        <p className="pdf-drop-hint">또는 PDF 파일을 여기에 끌어 놓으세요</p>
        {file && <div className="pdf-file-selection"><div><strong>{file.name}</strong><span>{(file.size / 1024 / 1024).toFixed(2)} MB · PDF</span></div><button type="button" disabled={loading} onClick={reset} aria-label="선택한 파일 제거" title="파일 제거">×</button></div>}
      </div>
      {error && <p className="pdf-error" role="alert">{error}</p>}
      {/* 답변 길이: 올리기 상자 바로 아래 왼쪽 */}
      <div className="pdf-options">
        <div className="ps-length" role="radiogroup" aria-label="답변 길이">
          <span className="ps-length-label" aria-hidden="true">답변 길이</span>
          {LENGTHS.map(item => <button key={item.id} type="button" role="radio" aria-checked={answerLength === item.id}
            className={answerLength === item.id ? 'is-active' : ''} onClick={() => onAnswerLengthChange?.(item.id)}>{item.label}</button>)}
        </div>
      </div>
      <div className="pdf-summary-description"><h2>PDF를 요약해 드립니다.</h2><p>강의 자료의 텍스트와 핵심 개념을 추출하고, 복습하기 좋은 요약으로 정리합니다.</p><p className="pdf-demo-notice">현재 프로토타입은 텍스트가 포함된 PDF의 개념 요약을 지원합니다.
        스캔 이미지 PDF는 아직 지원하지 않습니다.</p></div>
      <div className="pdf-extract-actions">
        <button
          type="button"
          className="pdf-select-button"
          disabled={!file || loading}
          onClick={extractSummary}
        >
          {loading ? '개념 요약 생성 중…' : '추출하기'}
        </button>
      </div>
      {recent.length > 0 && <section className="pdf-recent" aria-labelledby="pdf-recent-title">
        <h2 id="pdf-recent-title">최근 요약</h2>
        <ul>{recent.map(record => <li key={record.id}>
          <button type="button" disabled={loading} onClick={() => openSummary(record)}>
            <strong>{record.file.name}</strong>
            <time dateTime={new Date(record.createdAt).toISOString()}>{summaryDate.format(record.createdAt)}</time>
          </button>
        </li>)}</ul>
      </section>}
      {historyError && <p className="pdf-error" role="status">{historyError}</p>}
    </div>
  </section>;
}
