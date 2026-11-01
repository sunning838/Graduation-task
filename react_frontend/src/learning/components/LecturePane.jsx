import ChatMessage from './ChatMessage.jsx';
import PresentationLecture from './PresentationLecture.jsx';
import { VARIANT_LABELS } from './lectureVariants.js';

export default function LecturePane({ tab, onTabChange, variants, errors, loading, onRetry, onOpenTutor }) {
  const saved = variants[tab];
  const hasContent = saved && (saved.text || saved.visual || saved.research);

  return (
    <section className="pane presentation-pane">
      <div className="presentation-heading"><h3 className="pane-title">개념 강의</h3><span className="muted small">슬라이드로 듣고, 본문으로 다시 살펴보세요.</span></div>

      <div className="tabs" role="tablist">
        {VARIANT_LABELS.map((label) => (
          <button
            key={label}
            type="button"
            role="tab"
            aria-selected={tab === label}
            className={`tab${tab === label ? ' is-active' : ''}`}
            onClick={() => onTabChange(label)}
          >
            {label}
          </button>
        ))}
      </div>

      <div className="presentation-body">
        {saved ? (
          hasContent ? (
            saved.text ? (
              <PresentationLecture
                key={`${saved.lessonId}:${tab}:${saved.text}`}
                lessonId={saved.lessonId}
                variant={tab}
                message={saved}
                onOpenTutor={onOpenTutor}
              />
            ) : (
              <ChatMessage message={saved} />
            )
          ) : null
        ) : errors[tab] ? (
          <div className="notice">
            설명을 준비하지 못했습니다. 다시 시도해 주세요.
            <button type="button" className="btn btn-ghost" onClick={() => onRetry(tab)}>
              설명 다시 준비하기
            </button>
          </div>
        ) : loading[tab] ? (
          <p className="muted">'{tab}' 설명을 준비하고 있어요…</p>
        ) : null}
      </div>
    </section>
  );
}
