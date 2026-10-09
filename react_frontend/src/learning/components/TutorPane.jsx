import { useEffect, useRef, useState } from 'react';
import ChatMessage from './ChatMessage.jsx';

export default function TutorPane({ tab, messages, canAsk, loading, error, onAsk, onRetry }) {
  const [question, setQuestion] = useState('');
  const boxRef = useRef(null);

  // 새 답변이 오면 질문 목록 맨 아래로 스크롤
  useEffect(() => {
    if (boxRef.current) boxRef.current.scrollTop = boxRef.current.scrollHeight;
  }, [messages.length, loading]);

  const submit = () => {
    const q = question.trim();
    if (!q || !canAsk || loading) return;
    onAsk(q);
    setQuestion('');
  };

  return (
    <section className="pane">
      <h3 className="pane-title">튜터에게 질문</h3>
      <p className="muted small">현재 보고 있는 설명: {tab}</p>

      <div className="pane-scroll tutor-box" ref={boxRef}>
        {messages.length === 0 && !loading && (
          <p className="muted">직접 입력한 질문과 튜터의 답변이 여기에 쌓입니다.</p>
        )}
        {messages.map((m, i) => (
          <ChatMessage key={i} message={m} />
        ))}
        {loading && <p className="muted">답변을 준비하고 있어요…</p>}
      </div>

      {error && (
        <div className="notice">
          답변을 준비하지 못했습니다. 다시 시도해 주세요.
          <button type="button" className="btn btn-ghost" onClick={onRetry}>
            답변 다시 요청
          </button>
        </div>
      )}

      <div className="chat-input">
        <input
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          onKeyDown={(e) => {
            // 한글 입력 중 Enter가 두 번 처리되는 것 방지
            if (e.key === 'Enter' && !e.nativeEvent.isComposing) submit();
          }}
          placeholder={canAsk ? '이 부분이 궁금해요…' : '설명이 준비되면 질문할 수 있어요'}
          disabled={!canAsk}
        />
        <button
          type="button"
          className="btn btn-primary"
          onClick={submit}
          disabled={!canAsk || loading || !question.trim()}
        >
          질문하기
        </button>
      </div>
    </section>
  );
}