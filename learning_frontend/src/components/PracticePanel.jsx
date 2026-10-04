const PRACTICE_URL = 'http://localhost:8512/'; // 기존 문제풀이 화면 주소

export default function PracticePanel({ cert, lesson, onBack }) {
  const params = new URLSearchParams({ cert });
  const inScope = lesson && lesson.cert === cert;
  if (inScope) params.set('lesson', lesson.id);

  return (
    <section className="practice">
      <h1 className="page-title">실전 문제풀이</h1>
      <p className="muted">기존 문제풀이·모의고사·오답노트 화면에서 실전 학습을 진행하세요.</p>
      {inScope && <div className="notice">학습 범위: {lesson.title}</div>}
      <div className="row">
        <a className="btn btn-primary" href={`${PRACTICE_URL}?${params}`} target="_blank" rel="noopener noreferrer">
          문제풀이 시작
        </a>
        <button type="button" className="btn btn-ghost" onClick={onBack}>
          개념 학습으로 돌아가기
        </button>
      </div>
    </section>
  );
}