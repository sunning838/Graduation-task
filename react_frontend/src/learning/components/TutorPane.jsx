import { useEffect, useRef, useState } from 'react';
import ChatMessage from './ChatMessage.jsx';

// 대화는 사이드바 'AI 튜터'와 같은 대화입니다.
// chat: App.jsx에서 내려주는 { messages, loading, error, ready, send(text, context) }
// info: 안내 줄에 보여줄 문구, context: 질문과 함께 서버에만 보내는 정보 (없으면 개념 학습 기준)
export default function TutorPane({ tab, lessonTitle, chat, info, context }) {
  const [question, setQuestion] = useState('');
  const [lastQuestion, setLastQuestion] = useState('');
  const boxRef = useRef(null);
  const messages = chat?.messages || [];
  const loading = Boolean(chat?.loading);
  const canAsk = Boolean(chat?.ready);

  // 새 답변이 오면 질문 목록 맨 아래로 스크롤
  useEffect(() => {
    if (boxRef.current) boxRef.current.scrollTop = boxRef.current.scrollHeight;
  }, [messages.length, loading, chat?.error]);

  const ask = async text => {
    setLastQuestion(text);
    const ok = await chat.send(text, context ?? `[개념 학습 중인 단원: ${lessonTitle}]`);
    if (ok) setLastQuestion('');
    return ok;
  };

  const submit = async () => {
    const q = question.trim();
    if (!q || !canAsk || loading) return;
    setQuestion('');
    const ok = await ask(q);
    if (!ok) setQuestion(q);
  };

  return (
    <section className="pane">
      <h3 className="pane-title">튜터에게 질문</h3>

      {/* 왼쪽 탭 줄과 같은 높이로 맞춘 안내 줄 */}
      <div className="tutor-head">
        <span className="muted small">{info ?? `현재 보고 있는 설명: ${tab}`}</span>
      </div>

      <div className="pane-scroll tutor-box">
        {/* 대화 목록 (스크롤) */}
        <div className="tutor-messages" ref={boxRef}>
          {messages.length === 0 && !loading && (
            <p className="muted">직접 입력한 질문과 튜터의 답변이 여기에 쌓입니다.</p>
          )}
          {messages.map(m => (
            <ChatMessage key={m.id} message={{ role: m.role, text: m.content }} />
          ))}
          {loading && <p className="muted">답변을 준비하고 있어요…</p>}
          {chat?.error && !loading && (
            <div className="notice">
              답변을 준비하지 못했습니다. 다시 시도해 주세요.
              {lastQuestion && (
                <button type="button" className="btn btn-ghost" onClick={() => ask(lastQuestion)}>
                  답변 다시 요청
                </button>
              )}
            </div>
          )}
        </div>

        {/* 질문 입력바: 박스 안 맨 아래 */}
        <div className="chat-input">
          <input
            value={question}
            maxLength={2000}
            onChange={(e) => setQuestion(e.target.value)}
            onKeyDown={(e) => {
              // 한글 입력 중 Enter가 두 번 처리되는 것 방지
              if (e.key === 'Enter' && !e.nativeEvent.isComposing) submit();
            }}
            placeholder={canAsk ? '이 부분이 궁금해요…' : 'AI 튜터에 연결하는 중이에요'}
            disabled={!canAsk || loading}
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
      </div>
    </section>
  );
}