import { useRef, useState } from 'react';
import PdfResult from '../components/PdfResult.jsx';
import '../learning-soft.css';
import './pdf-summary.css';

// 답변 길이 (AI 튜터 화면의 설정과 같은 값)
const LENGTHS = [
  { id: 'short', label: '간단히' },
  { id: 'medium', label: '보통' },
  { id: 'long', label: '자세히' },
];

function PdfIcon() {
  return <svg viewBox="0 0 64 72" fill="none" aria-hidden="true"><path d="M12 3h27l15 15v51H12zM39 3v16h15M22 31h22M22 39h22M22 47h14" stroke="currentColor" strokeWidth="2.5" strokeLinejoin="round"/><rect x="3" y="52" width="39" height="17" rx="4" fill="currentColor"/><text x="22" y="64" textAnchor="middle" fill="var(--bg-main)" fontSize="11" fontWeight="700">PDF</text></svg>;
}

// tutorChat / answerLength / onAnswerLengthChange: App.jsx의 AI 튜터 대화와 답변 길이
export default function PdfSummaryPage({ tutorChat, answerLength, onAnswerLengthChange }) {
  const inputRef = useRef(null);
  const [file, setFile] = useState(null);
  const [error, setError] = useState('');
  const [dragging, setDragging] = useState(false);
  const [result, setResult] = useState(false);
  const selectFile = files => {
    setDragging(false);
    if (!files?.length) return;
    setResult(false);
    if (files.length !== 1 || !/\.pdf$/i.test(files[0].name)) {
      setFile(null); setError('PDF 파일 한 개를 선택해 주세요.'); return;
    }
    if (!files[0].size) { setFile(null); setError('비어 있는 파일입니다. 다른 PDF를 선택해 주세요.'); return; }
    setFile(files[0]); setError('');
  };
  const reset = () => { setFile(null); setResult(false); setError(''); if (inputRef.current) inputRef.current.value = ''; };

  // 추출하기를 누르면: 원문 + 요약 화면 (이전으로 누르면 올리기 화면, 고른 파일은 유지)
  if (result && file) return <section className="pdf-summary-page is-result" aria-label="PDF 요약">
    <PdfResult file={file} onBack={() => setResult(false)} tutorChat={tutorChat} />
  </section>;

  // 처음 화면: PDF 올리기
  return <section className="pdf-summary-page" aria-labelledby="pdf-summary-title">
    <div className="pdf-summary-inner">
      <header className="pdf-summary-heading"><span className="pdf-demo-badge">미리보기</span><h1 id="pdf-summary-title">PDF 요약</h1><p>학습 자료의 핵심을 한눈에 살펴보세요.</p></header>
      <div className={`pdf-dropzone${dragging ? ' is-dragging' : ''}`} onDragOver={event => { event.preventDefault(); setDragging(true); }} onDragLeave={event => { if (!event.currentTarget.contains(event.relatedTarget)) setDragging(false); }} onDrop={event => { event.preventDefault(); selectFile(event.dataTransfer.files); }}>
        <PdfIcon />
        <input ref={inputRef} type="file" accept=".pdf,application/pdf" hidden aria-label="PDF 파일 선택" onChange={event => { selectFile(event.target.files); event.target.value = ''; }} />
        <button type="button" className="pdf-select-button" onClick={() => inputRef.current?.click()}>{file ? '다른 파일 선택' : '파일 선택'}<span aria-hidden="true">＋</span></button>
        <p className="pdf-drop-hint">또는 PDF 파일을 여기에 끌어 놓으세요</p>
        {file && <div className="pdf-file-selection"><div><strong>{file.name}</strong><span>{(file.size / 1024 / 1024).toFixed(2)} MB · PDF</span></div><button type="button" onClick={reset} aria-label="선택한 파일 제거" title="파일 제거">×</button></div>}
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
      <div className="pdf-summary-description"><h2>PDF를 요약해 드립니다.</h2><p>강의 자료의 텍스트와 핵심 개념을 추출하고, 복습하기 좋은 요약으로 정리합니다.</p><p className="pdf-demo-notice">현재는 목업 화면입니다. 원문은 내 컴퓨터에서만 열어 보여주고 서버로 보내지 않으며, 요약은 고정된 예시가 표시됩니다.</p></div>
      <div className="pdf-extract-actions"><button type="button" className="pdf-select-button" disabled={!file} onClick={() => setResult(true)}>추출하기</button></div>
    </div>
  </section>;
}