import ChatMessage from './ChatMessage.jsx';
import PresentationLecture from './PresentationLecture.jsx';
import { DEFAULT_VARIANT } from './lectureVariants.js';

// 설명 방식은 '기본 설명' 하나만 사용합니다.
export default function LecturePane({ message, error, onRetry }) {
  if (message?.text) {
    return <PresentationLecture
      key={`${message.lessonId}:${message.text}`}
      lessonId={message.lessonId}
      variant={DEFAULT_VARIANT}
      message={message}
    />;
  }

  if (message && (message.visual || message.research)) {
    return <div className="ls-card ls-fallback"><ChatMessage message={message} /></div>;
  }

  if (error) {
    return <div className="ls-card ls-state">
      <p>설명을 준비하지 못했어요. 다시 시도해 주세요.</p>
      <button type="button" className="ls-pill" onClick={onRetry}>설명 다시 준비하기</button>
    </div>;
  }

  return <div className="ls-card ls-state" role="status">
    <span className="presentation-spinner" />
    <p>강의를 준비하고 있어요…</p>
  </div>;
}