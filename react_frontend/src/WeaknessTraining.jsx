import { useEffect, useRef, useState } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { apiRequest } from './api';
import { StatsStatus } from './LearningDashboard';

export default function WeaknessTraining({ cert, label, stats, loading, error, onRefresh, onGeneralPractice }) {
  const [quiz, setQuiz] = useState(null);
  const [answer, setAnswer] = useState(null);
  const [result, setResult] = useState(null);
  const [phase, setPhase] = useState('idle');
  const [requestError, setRequestError] = useState('');
  const request = useRef(null);
  const analysis = stats?.weakness;
  const busy = phase !== 'idle';
  useEffect(() => () => request.current?.abort(), []);

  const generate = async () => {
    if (!cert || request.current || analysis?.status !== 'ready') return;
    const controller = new AbortController();
    request.current = controller;
    setPhase('generating');
    setRequestError('');
    setQuiz(null);
    setAnswer(null);
    setResult(null);
    try {
      const data = await apiRequest('/api/quiz', { signal: controller.signal, body: { cert, mode: 'weakness' } });
      if (!controller.signal.aborted) setQuiz(data);
    } catch (err) {
      if (!controller.signal.aborted) {
        setRequestError(err.message);
        if (['NO_WEAK_TOPIC', 'INSUFFICIENT_HISTORY'].includes(err.code)) onRefresh();
      }
    } finally {
      if (!controller.signal.aborted) {
        request.current = null;
        setPhase('idle');
      }
    }
  };

  const submit = async () => {
    if (!quiz || answer === null || result || request.current) return;
    const controller = new AbortController();
    request.current = controller;
    setPhase('submitting');
    setRequestError('');
    try {
      const data = await apiRequest('/api/quiz/submit', {
        signal: controller.signal, body: { attempt_id: quiz.attempt_id, selected_answer: answer },
      });
      if (!controller.signal.aborted) {
        setResult(data);
        onRefresh();
      }
    } catch (err) {
      if (!controller.signal.aborted) setRequestError(err.message);
    } finally {
      if (!controller.signal.aborted) {
        request.current = null;
        setPhase('idle');
      }
    }
  };

  return <section className="weakness-page"><div className="weakness-page-content">
    <div className="weakness-page-header"><div>
      <span className="weakness-page-label">WEAKNESS TRAINING</span><h1>취약점 집중 학습</h1>
      <p>실제 풀이 기록을 바탕으로 취약한 과목을 집중적으로 연습합니다.</p>
    </div><button type="button" onClick={onRefresh} disabled={loading}>분석 새로고침</button></div>
    <StatsStatus loading={loading} error={error} onRetry={onRefresh} />
    {analysis && <>
      <div className="weakness-overview-grid">
        <section className="weakness-panel weakness-focus-panel">
          <div className="weakness-panel-kicker">PRIORITY AREA</div>
          <h2>{analysis.focus?.label || (analysis.status === 'insufficient_data' ? '분석할 기록 부족' : '취약 기준 해당 과목 없음')}</h2>
          <p className="weakness-focus-description">{analysis.message}</p>
          {analysis.focus && <>
            <div className="weakness-score-row"><span>현재 정답률</span><strong>{analysis.focus.accuracy}%</strong></div>
            <div className="weakness-progress"><div className="weakness-progress-value" style={{ width: `${analysis.focus.accuracy}%` }} /></div>
          </>}
          <p>과목별 {analysis.min_attempts}문제 이상 · 정답률 {analysis.accuracy_threshold}% 미만 기준</p>
          {analysis.insufficient_subjects.length > 0 && <p>기록이 부족한 {analysis.insufficient_subjects.length}개 과목은 분석에서 제외됩니다.</p>}
          {analysis.status !== 'ready' && <button type="button" className="weakness-retry-button" onClick={onGeneralPractice}>일반 문제 풀기</button>}
        </section>
        <section className="weakness-panel weakness-ranking-panel">
          <h2>취약 과목 순위</h2>
          <div className="weakness-ranking-list">{analysis.ranking.map((item, index) =>
            <div className="weakness-ranking-item" key={item.topic}>
              <span className="weakness-rank-number">{index + 1}</span>
              <div className="weakness-rank-topic"><strong>{item.label}</strong><span>{item.correct}/{item.total}문제 정답 · {item.accuracy}%</span></div>
            </div>)}{!analysis.ranking.length && <p>현재 표시할 취약 과목이 없습니다.</p>}</div>
        </section>
      </div>
    </>}
    <section className="weakness-practice-card">
      <div className="weakness-practice-header"><div><h2>취약 과목 집중 문제</h2>
        <p>과목별 학습 자료를 바탕으로 문제를 생성합니다.</p></div>
        <div className="weakness-practice-tags"><span>{label}</span>{quiz && <span>{quiz.topic_label}</span>}</div>
      </div>
      {requestError && <p role="alert">{requestError}</p>}
      {phase === 'generating' && <p role="status">문제를 생성하고 검토하고 있습니다.</p>}
      {!quiz && <button type="button" className="weakness-submit-button" disabled={busy || loading || analysis?.status !== 'ready'} onClick={generate}>
        {phase === 'generating' ? '문제 생성 중…' : '취약점 문제 시작'}
      </button>}
      {quiz && <div className="weakness-question-area">
        <h3>Q. {quiz.question}</h3>
        {quiz.code_block && <pre className="quiz-code"><code>{quiz.code_block}</code></pre>}
        {quiz.table_data && <ReactMarkdown remarkPlugins={[remarkGfm]}>{quiz.table_data}</ReactMarkdown>}
        <div className="weakness-options">{quiz.options.map((option, index) => {
          const number = index + 1;
          const state = result ? (number === result.correct_answer ? ' correct' : number === answer ? ' incorrect' : '') : number === answer ? ' selected' : '';
          return <button type="button" key={number} className={`weakness-option${state}`} disabled={busy || !!result} onClick={() => setAnswer(number)}>
            <span className="weakness-option-number">{number}</span><span>{option.replace(/^\s*\d+\s*[).:-]\s*/, '')}</span>
          </button>;
        })}</div>
        {!result && <button type="button" className="weakness-submit-button" disabled={busy || answer === null} onClick={submit}>{phase === 'submitting' ? '채점 중…' : '정답 제출'}</button>}
        {result && <div className="weakness-result">
          <div className={`weakness-result-title ${result.is_correct ? 'success' : 'wrong'}`}>{result.is_correct ? '정답입니다!' : '오답입니다.'}</div>
          <p>정답: {result.correct_answer}번</p>
          <div className="weakness-explanation"><h3>핵심 해설</h3><ReactMarkdown remarkPlugins={[remarkGfm]}>{result.explanation}</ReactMarkdown></div>
          <button type="button" className="weakness-retry-button" disabled={busy || loading || analysis?.status !== 'ready'} onClick={generate}>다음 취약점 문제</button>
        </div>}
      </div>}
    </section>
  </div></section>;
}
