// 취약점 문제 출제 요청 중, 일반 문제풀이의 로딩 스타일을 재사용합니다.
export default function WeaknessGenerationLoading() {
  return (
    <div className="quiz-loading" role="status" aria-live="polite">
      <div className="quiz-loading-spinner" aria-hidden="true" />
      <h3>AI가 문제를 생성하고 검토하고 있습니다.</h3>
      <p>출제 후 검수까지 진행하므로 잠시 시간이 걸릴 수 있습니다.</p>
    </div>
  );
}
