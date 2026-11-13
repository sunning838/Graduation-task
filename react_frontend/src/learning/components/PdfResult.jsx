import { useCallback, useEffect, useRef, useState } from 'react';
import * as pdfjsLib from 'pdfjs-dist';
import pdfWorker from 'pdfjs-dist/build/pdf.worker.min.mjs?url';
import TutorDrawer from './TutorDrawer.jsx';
import TutorPane from './TutorPane.jsx';
import { ChatIcon, ChevronLeft, ChevronRight } from './icons.jsx';

pdfjsLib.GlobalWorkerOptions.workerSrc = pdfWorker;

// 확대/축소: 일반 웹 브라우저처럼 10% 단위 (50% ~ 300%)
const MIN_ZOOM = 0.5;
const MAX_ZOOM = 3;

// 백엔드 연결 전까지 보여줄 예시 요약입니다. (실제 파일 내용 아님)
// page: 이 항목을 누르면 원문에서 이동할 쪽
const MOCK_SUMMARY = {
  lines: [
    '소프트웨어 개발은 요구사항 분석, 설계, 구현, 테스트, 유지보수 단계로 이루어진다.',
    '각 단계의 결과물은 다음 단계의 입력이 된다.',
    '유지보수는 개발 이후 가장 오래 걸리고 비용이 큰 단계다.',
  ],
  sections: [
    { title: '요구사항 분석', page: 1, text: '사용자의 요구와 시스템의 범위를 정리합니다.' },
    { title: '설계·구현', page: 2, text: '시스템 구조를 설계하고 프로그램으로 구현합니다.' },
    { title: '테스트·유지보수', page: 3, text: '오류를 확인하고, 운영 중 생기는 변경 사항을 반영합니다.' },
  ],
};

const clampZoom = value => Math.min(MAX_ZOOM, Math.max(MIN_ZOOM, Math.round(value * 100) / 100));
// 지금 배율에서 10% 단위로 한 칸 이동 (예: 맞춤 114% → 확대 120%, 축소 110%)
const stepZoom = (value, direction) => {
  const percent = Math.round(value * 100);
  const next = direction > 0 ? Math.floor(percent / 10) * 10 + 10 : Math.ceil(percent / 10) * 10 - 10;
  return clampZoom(next / 100);
};

function PdfBadge() {
  return <span className="ps-file-icon" aria-hidden="true">PDF</span>;
}

// file: 사용자가 고른 PDF(File), tutorChat: App.jsx의 AI 튜터 대화
// onBack: 올리기 화면으로 돌아가기 (고른 파일은 그대로 남음)
export default function PdfResult({ file, onBack, tutorChat }) {
  const [doc, setDoc] = useState(null);
  const [loadError, setLoadError] = useState('');
  const [page, setPage] = useState(1);
  const [zoom, setZoom] = useState(null); // null = 폭에 맞춤
  const [fitZoom, setFitZoom] = useState(1);
  const [rendering, setRendering] = useState(false);
  const [tutorOpen, setTutorOpen] = useState(false);
  const stageRef = useRef(null);
  const canvasRef = useRef(null);
  const renderTask = useRef(null);
  const pageCount = doc?.numPages || 0;
  const scale = zoom ?? fitZoom;

  // 1) PDF 파일 읽기
  useEffect(() => {
    let cancelled = false;
    let loadingTask = null; // PDF를 닫을 때는 이 작업(loadingTask)을 destroy 합니다.
    setDoc(null); setLoadError(''); setPage(1); setZoom(null);
    file.arrayBuffer().then(buffer => {
      if (cancelled) return null;
      loadingTask = pdfjsLib.getDocument({ data: buffer });
      return loadingTask.promise;
    }).then(pdf => {
      if (pdf && !cancelled) setDoc(pdf);
    }).catch(error => {
      if (cancelled) return;
      setLoadError(error?.name === 'PasswordException'
        ? '비밀번호가 걸린 PDF는 열 수 없어요.'
        : 'PDF를 열지 못했어요. 파일이 손상되지 않았는지 확인해 주세요.');
    });
    return () => {
      cancelled = true;
      // 화면을 떠날 때 PDF 정리 (실패해도 화면이 꺼지지 않도록 감쌈)
      try { renderTask.current?.cancel(); } catch { /* 무시 */ }
      try { Promise.resolve(loadingTask?.destroy?.()).catch(() => {}); } catch { /* 무시 */ }
    };
  }, [file]);

  // 2) '폭에 맞춤' 배율 계산 (화면 크기가 바뀌면 다시 계산)
  const measureFit = useCallback(async () => {
    if (!doc || !stageRef.current) return;
    const pdfPage = await doc.getPage(page);
    const base = pdfPage.getViewport({ scale: 1 });
    const width = stageRef.current.clientWidth - 48;
    const height = stageRef.current.clientHeight - 48;
    setFitZoom(clampZoom(Math.min(width / base.width, height / base.height)));
  }, [doc, page]);

  useEffect(() => {
    measureFit();
    const stage = stageRef.current;
    if (!stage || typeof ResizeObserver === 'undefined') return;
    const observer = new ResizeObserver(() => measureFit());
    observer.observe(stage);
    return () => observer.disconnect();
  }, [measureFit]);

  // 3) 현재 쪽을 캔버스에 그리기
  useEffect(() => {
    if (!doc || !canvasRef.current) return;
    let cancelled = false;
    (async () => {
      setRendering(true);
      try {
        const pdfPage = await doc.getPage(page);
        if (cancelled) return;
        const viewport = pdfPage.getViewport({ scale });
        const ratio = window.devicePixelRatio || 1;
        const canvas = canvasRef.current;
        canvas.width = Math.floor(viewport.width * ratio);
        canvas.height = Math.floor(viewport.height * ratio);
        canvas.style.width = `${Math.floor(viewport.width)}px`;
        canvas.style.height = `${Math.floor(viewport.height)}px`;
        try { renderTask.current?.cancel(); } catch { /* 무시 */ }
        const context = canvas.getContext('2d');
        const task = pdfPage.render({
          canvas, canvasContext: context, viewport,
          transform: ratio !== 1 ? [ratio, 0, 0, ratio, 0, 0] : undefined,
        });
        renderTask.current = task;
        await task.promise;
      } catch (error) {
        if (error?.name !== 'RenderingCancelledException' && !cancelled) setLoadError('이 쪽을 그리지 못했어요.');
      } finally {
        if (!cancelled) setRendering(false);
      }
    })();
    return () => { cancelled = true; try { renderTask.current?.cancel(); } catch { /* 무시 */ } };
  }, [doc, page, scale]);

  const goPage = next => setPage(value => Math.min(pageCount || 1, Math.max(1, typeof next === 'function' ? next(value) : next)));
  const zoomBy = direction => setZoom(stepZoom(scale, direction));

  // 원문 위에서 Ctrl + 휠로 확대/축소
  useEffect(() => {
    const stage = stageRef.current;
    if (!stage) return;
    const onWheel = event => {
      if (!event.ctrlKey) return;
      event.preventDefault();
      setZoom(stepZoom(zoom ?? fitZoom, event.deltaY < 0 ? 1 : -1));
    };
    stage.addEventListener('wheel', onWheel, { passive: false });
    return () => stage.removeEventListener('wheel', onWheel);
  }, [zoom, fitZoom]);

  const onStageKey = event => {
    if (event.key === 'ArrowRight' || event.key === 'PageDown') { event.preventDefault(); goPage(value => value + 1); }
    if (event.key === 'ArrowLeft' || event.key === 'PageUp') { event.preventDefault(); goPage(value => value - 1); }
    if (event.key === '+' || event.key === '=') { event.preventDefault(); zoomBy(1); }
    if (event.key === '-') { event.preventDefault(); zoomBy(-1); }
  };

  const sizeLabel = file.size < 1024 * 1024 ? `${Math.max(1, Math.round(file.size / 1024))} KB` : `${(file.size / 1024 / 1024).toFixed(2)} MB`;

  return <section className="ps-result" aria-label="PDF 요약 결과">
    <header className="ps-result-head">
      <div className="ps-file">
        <PdfBadge />
        <div>
          <h2>{file.name}</h2>
          <span>{pageCount ? `${pageCount}쪽 · ` : ''}{sizeLabel} · 예시 요약</span>
        </div>
      </div>
      <div className="ps-head-actions">
        <button type="button" className="ls-pill" onClick={() => setTutorOpen(true)}><ChatIcon />튜터에게 질문</button>
        <button type="button" className="ls-pill ps-back" onClick={onBack}><ChevronLeft />이전으로</button>
      </div>
    </header>

    <div className="ps-grid">
      <section className="ls-card ps-viewer" aria-label="원문">
        <div className="ps-card-head">
          <h3>원문</h3>
          <div className="ps-zoom" aria-label="확대/축소">
            <button type="button" className="ls-round" aria-label="축소" disabled={!doc || scale <= MIN_ZOOM} onClick={() => zoomBy(-1)}>−</button>
            <button type="button" className="ps-zoom-value" onClick={() => setZoom(null)} title="화면에 맞추기" aria-label={`현재 ${Math.round(scale * 100)}%, 눌러서 화면에 맞추기`}>
              {zoom === null ? '맞춤' : `${Math.round(scale * 100)}%`}
            </button>
            <button type="button" className="ls-round" aria-label="확대" disabled={!doc || scale >= MAX_ZOOM} onClick={() => zoomBy(1)}>+</button>
          </div>
        </div>

        <div className={`ps-stage${zoom !== null && zoom > fitZoom ? ' is-zoomed' : ''}`} ref={stageRef} tabIndex={0} onKeyDown={onStageKey}
          aria-label={pageCount ? `원문 ${page}쪽 / ${pageCount}쪽` : '원문'}>
          {loadError
            ? <div className="ps-stage-note" role="alert"><p>{loadError}</p><button type="button" className="ls-pill" onClick={onBack}><ChevronLeft />이전으로 돌아가기</button></div>
            : <>
              {!doc && <p className="ps-stage-note" role="status"><span className="presentation-spinner" />PDF를 여는 중…</p>}
              <canvas ref={canvasRef} className={`ps-canvas${rendering ? ' is-rendering' : ''}`} hidden={!doc} />
            </>}
        </div>

        <div className="ps-page-nav">
          <button type="button" className="ls-round" aria-label="이전 쪽" disabled={!doc || page <= 1} onClick={() => goPage(value => value - 1)}><ChevronLeft /></button>
          <label className="ps-page-input">
            <input type="number" min="1" max={pageCount || 1} value={page} disabled={!doc}
              aria-label="이동할 쪽" onChange={event => { const value = Number(event.target.value); if (value) goPage(value); }} />
            <span>/ {pageCount || '–'}쪽</span>
          </label>
          <button type="button" className="ls-round" aria-label="다음 쪽" disabled={!doc || page >= pageCount} onClick={() => goPage(value => value + 1)}><ChevronRight /></button>
        </div>
      </section>

      <section className="ls-card ps-summary" aria-label="요약">
        <div className="ps-card-head">
          <h3>요약</h3>
          <span className="pdf-demo-badge">예시 · 실제 파일 내용 아님</span>
        </div>
        <div className="ps-summary-scroll">
          <div className="ps-lines">
            <span>세 줄 요약</span>
            <ol>{MOCK_SUMMARY.lines.map(line => <li key={line}>{line}</li>)}</ol>
          </div>
          {MOCK_SUMMARY.sections.map(section => {
            const target = Math.min(section.page, pageCount || section.page);
            const current = target === page;
            return <button key={section.title} type="button" className={`ps-section${current ? ' is-current' : ''}`}
              aria-current={current ? 'true' : undefined} onClick={() => goPage(target)}>
              <strong>{section.title} <small>· {target}쪽</small></strong>
              <span>{section.text}</span>
            </button>;
          })}
        </div>
      </section>
    </div>

    <TutorDrawer open={tutorOpen} onClose={() => setTutorOpen(false)} heading="자료를 보다가 궁금한 점을 질문해 보세요.">
      <TutorPane key={file.name} chat={tutorChat}
        info={`보고 있는 자료: ${file.name}`}
        context={`[PDF 요약 중인 자료: ${file.name}]`} />
    </TutorDrawer>
  </section>;
}