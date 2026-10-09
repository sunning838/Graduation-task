import ProgressBar from './ProgressBar.jsx';

export default function Sidebar({ config, certs, cert, onCertChange, done, total, onOpenPractice }) {
  return (
    <aside className="concept-sidebar">
      <label className="field">
        <span className="field-label">자격증</span>
        <select value={cert} onChange={(e) => onCertChange(e.target.value)}>
          {certs.map((c) => (
            <option key={c} value={c}>
              {config[c].label}
            </option>
          ))}
        </select>
      </label>

      <ProgressBar value={total ? done / total : 0} label={`전체 ${done} / ${total} 학습 완료`} />
      <p className="muted small">진행도는 학습한 분량이며 문제 정답률과 별개입니다.</p>

      <button type="button" className="btn btn-ghost btn-block" onClick={onOpenPractice}>
        실전 문제풀이 열기
      </button>
    </aside>
  );
}