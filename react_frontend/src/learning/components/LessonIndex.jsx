import { useState } from 'react';

export default function LessonIndex({ config, cert, lessons, states, onOpen }) {
  const subjects = [...new Set(lessons.map((l) => l.subject))];

  return (
    <section>
      <h2 id="learning-index-heading" tabIndex={-1} className="section-title">학습 목차</h2>
      {subjects.map((subject) => (
        <SubjectGroup
          key={`${cert}-${subject}`}
          label={config[cert].topics[subject] ?? subject}
          group={lessons.filter((l) => l.subject === subject)}
          states={states}
          onOpen={onOpen}
        />
      ))}
    </section>
  );
}

function SubjectGroup({ label, group, states, onOpen }) {
  const [selected, setSelected] = useState(group[0].id);
  const complete = group.filter((l) => states[l.id]?.status === 'complete').length;

  return (
    <details className="subject">
      <summary>
        {label}
      </summary>
      <div className="subject-body">
        <div className="subject-progress">
          <span className="subject-progress-count" aria-hidden="true">{complete} / {group.length} 완료</span>
          <div className="subject-progress-segments" role="progressbar" aria-label={`${label} 학습 진도`} aria-valuemin={0} aria-valuemax={group.length} aria-valuenow={complete} aria-valuetext={`${group.length}개 단원 중 ${complete}개 완료`} style={{ gridTemplateColumns: `repeat(${group.length}, minmax(0, 1fr))` }}>
            {group.map(lesson => <span key={lesson.id} className={`subject-progress-segment${states[lesson.id]?.status === 'complete' ? ' is-complete' : ''}`} title={`${lesson.title} · ${states[lesson.id]?.status === 'complete' ? '완료' : '미완료'}`} aria-hidden="true" />)}
          </div>
        </div>
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