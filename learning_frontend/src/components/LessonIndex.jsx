import { useState } from 'react';
import ProgressBar from './ProgressBar.jsx';

export default function LessonIndex({ config, cert, lessons, states, onOpen }) {
  const subjects = [...new Set(lessons.map((l) => l.subject))];

  return (
    <section>
      <h2 className="section-title">학습 목차</h2>
      {subjects.map((subject) => (
        <SubjectGroup
          key={`${cert}-${subject}`}
          label={config[cert].topics[subject] ?? subject}
          group={lessons.filter((l) => l.subject === subject)}
          states={states}
          onOpen={onOpen}
          defaultOpen={subjects.length === 1}
        />
      ))}
    </section>
  );
}

function SubjectGroup({ label, group, states, onOpen, defaultOpen }) {
  const [selected, setSelected] = useState(group[0].id);
  const complete = group.filter((l) => states[l.id]?.status === 'complete').length;

  return (
    <details className="subject" open={defaultOpen}>
      <summary>
        {label} <span className="muted small">{complete}/{group.length} 완료</span>
      </summary>
      <div className="subject-body">
        <ProgressBar value={complete / group.length} />
        <label className="field">
          <span className="field-label">학습 항목</span>
          <select value={selected} onChange={(e) => setSelected(e.target.value)}>
            {group.map((l) => (
              <option key={l.id} value={l.id}>
                {`${states[l.id]?.status === 'complete' ? '✓' : '◌'} ${l.title}`}
              </option>
            ))}
          </select>
        </label>
        <button type="button" className="btn btn-primary" onClick={() => onOpen(group.find((l) => l.id === selected))}>
          선택한 단원 공부하기
        </button>
      </div>
    </details>
  );
}