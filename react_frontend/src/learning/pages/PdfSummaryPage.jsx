import { useRef, useState } from 'react';
import './pdf-summary.css';

function PdfIcon() {
  return <svg viewBox="0 0 64 72" fill="none" aria-hidden="true"><path d="M12 3h27l15 15v51H12zM39 3v16h15M22 31h22M22 39h22M22 47h14" stroke="currentColor" strokeWidth="2.5" strokeLinejoin="round"/><rect x="3" y="52" width="39" height="17" rx="4" fill="currentColor"/><text x="22" y="64" textAnchor="middle" fill="var(--bg-main)" fontSize="11" fontWeight="700">PDF</text></svg>;
}

export default function PdfSummaryPage() {
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
      <div className="pdf-summary-description"><h2>PDF를 요약해 드립니다.</h2><p>강의 자료의 텍스트와 핵심 개념을 추출하고, 복습하기 좋은 요약으로 정리합니다.</p><p className="pdf-demo-notice">현재는 목업 화면입니다. 파일을 업로드하거나 내용을 분석하지 않으며, 추출하기를 누르면 고정된 예시가 표시됩니다.</p></div>
      <div className="pdf-extract-actions"><button type="button" className="pdf-select-button" disabled={!file} onClick={() => setResult(true)}>추출하기</button></div>
      {result && file && <section className="pdf-mock-result" aria-label="예시 요약 결과" aria-live="polite"><span className="pdf-demo-badge">예시 결과 · 실제 파일 내용 아님</span><h2>소프트웨어 생명 주기</h2><p>소프트웨어 개발은 요구사항 분석, 설계, 구현, 테스트, 유지보수 단계로 이루어집니다.</p><h3>핵심 개념</h3><ul><li>요구사항 분석: 사용자의 요구와 시스템의 범위를 정리합니다.</li><li>설계·구현: 구조를 설계하고 프로그램으로 구현합니다.</li><li>테스트·유지보수: 오류를 확인하고 변경 사항을 반영합니다.</li></ul></section>}
    </div>
  </section>;
}
