import { useEffect, useRef, useState } from 'react';
import { apiRequest } from './api';

export default function LearningResetDialog({ cert, label, onClose, onComplete }) {
  const dialog = useRef(null);
  const pending = useRef(null);
  const lock = useRef(false);
  const [includeProgress, setIncludeProgress] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');

  useEffect(() => {
    const element = dialog.current;
    element.showModal();
    return () => element.close();
  }, []);

  async function submit() {
    if (lock.current || !cert) return;
    lock.current = true;
    setBusy(true); setError('');
    // A retry after a lost response must not delete records created afterwards.
    if (!pending.current || pending.current.includeProgress !== includeProgress) {
      pending.current = { requestId: crypto.randomUUID(), includeProgress };
    }
    try {
      const result = await apiRequest('/api/learning/reset', { body: {
        cert, learner: '내 학습', include_progress: includeProgress,
        confirm: true, request_id: pending.current.requestId,
      } });
      onComplete(result);
    } catch (e) {
      setError(e.message);
    } finally {
      lock.current = false; setBusy(false);
    }
  }

  return <dialog ref={dialog} className="learning-reset-dialog" aria-labelledby="learning-reset-title"
    aria-describedby="learning-reset-description" aria-busy={busy}
    onCancel={event => { event.preventDefault(); if (!lock.current) onClose(); }}>
    <h2 id="learning-reset-title">학습 데이터 초기화</h2>
    <p id="learning-reset-description"><strong>{label}</strong>의 문제풀이 기록과 관련 학습 결과를 삭제합니다. 삭제 후 복구할 수 없습니다.</p>
    <ul>
      <li>문제풀이·정답률·목표 달성 기록</li>
      <li>서술해보기·소크라테스 문답 기록</li>
      <li>모의고사 응시·답안·결과와 이어풀기 정보</li>
      <li>취약점 개념 요약</li>
    </ul>
    <label className="learning-reset-option">
      <input type="checkbox" checked={includeProgress} disabled={busy}
        onChange={event => setIncludeProgress(event.target.checked)} />
      <span>개념학습 진도도 초기화<span className="learning-reset-option-hint">단원별 시작·완료 기록과 이어서 공부할 단원을 함께 지웁니다.</span></span>
    </label>
    <p className="learning-reset-kept">참고 자료, 문제은행, 강의 이미지·음성, 일반 AI 튜터 대화는 유지합니다.</p>
    <p className="learning-reset-shared">같은 서버의 풀이 기록을 공유하는 이용자에게도 적용됩니다.</p>
    {error && <p role="alert" className="learning-reset-error">{error}</p>}
    {busy && <p role="status">학습 기록을 초기화하고 있습니다…</p>}
    <div className="learning-reset-actions">
      <button type="button" autoFocus disabled={busy} onClick={onClose}>취소</button>
      <button type="button" className="learning-reset-confirm" disabled={busy || !cert} onClick={submit}>
        {busy ? '초기화 중…' : '기록 삭제하기'}
      </button>
    </div>
  </dialog>;
}
