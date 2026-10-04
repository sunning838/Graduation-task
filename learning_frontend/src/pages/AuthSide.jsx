export default function AuthSide() {
  return (
    <aside className="auth-side">
      <div className="auth-brand">AI Tutor</div>
      <p className="auth-side-text">
        기출 데이터로 만든 문제를 풀고, 틀린 문제는 오답노트에 모아 다시 풀어보세요.
      </p>
      <ul className="auth-features">
        <li><span className="icon">◉</span>AI 튜터</li>
        <li><span className="icon">▫</span>문제 풀이</li>
        <li><span className="icon">◇</span>취약점 학습</li>
        <li><span className="icon">△</span>모의고사</li>
        <li><span className="icon">○</span>오답노트</li>
      </ul>
      <div className="auth-side-foot">안양대학교 AI Tutor</div>
    </aside>
  );
}
