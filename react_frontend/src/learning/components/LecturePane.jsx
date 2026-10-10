import ChatMessage from './ChatMessage.jsx';
import AudioLecture from './AudioLecture.jsx';

// 탭별로 백엔드에 보내는 요청 문장 (파이썬 variant_prompts 그대로)
export const VARIANT_PROMPTS = {
  '기본 설명': '처음 배우는 학생에게 이 학습 항목의 첫 강의를 시작해 주세요.',
  '쉬운 설명':
    '현재 단원의 개념 전체를 쉬운 용어와 짧은 단계로 다시 설명하세요. 낯선 용어를 먼저 풀고 기본 설명의 정확한 조건은 유지하세요. 독립적으로 읽을 수 있는 강의로 작성하세요.',
  '예시로 이해하기':
    '현재 단원을 하나의 구체적인 사례로 단계별 설명하세요. 사례의 각 부분과 해당 개념을 명시적으로 연결하세요. 예시임을 표시하고 원문 조건을 유지하세요. 독립적인 사례 강의로 작성하세요.',
};
export const DEFAULT_VARIANT = '기본 설명';

export default function LecturePane({ tab, onTabChange, variants, errors, loading, onRetry }) {
  const saved = variants[tab];
  const hasContent = saved && (saved.text || saved.visual || saved.research);

  return (
    <section className="pane">
      <h3 className="pane-title">개념 강의</h3>

      <div className="tabs" role="tablist">
        {Object.keys(VARIANT_PROMPTS).map((label) => (
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

      <div className="pane-scroll lecture-box">
        {saved ? (
          hasContent ? (
            saved.text ? (
              // 강의 글이 있으면 음성 강의와 함께 표시
              <AudioLecture
                key={`${saved.lessonId}:${tab}`}
                lessonId={saved.lessonId}
                variant={tab}
                message={saved}
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