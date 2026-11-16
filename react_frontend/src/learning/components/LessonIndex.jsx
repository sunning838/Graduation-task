import { useState } from 'react';
import './lesson-index.css';

const isDone = (states, lesson) => states[lesson.id]?.status === 'complete';
// 제목 앞 번호("3. ")는 동그라미에 따로 보여주므로 글자에서는 뺍니다.
const plainTitle = title => title.replace(/^\s*\d+\.\s*/, '');

// 과목 안에서 이어서 볼 단원: 마지막으로 본 단원 → (끝냈으면) 그다음 안 끝낸 단원 → 처음 안 끝낸 단원 → 첫 단원
function resumeLesson(group, states) {
  const recent = group.reduce((best, l) => states[l.id] && (!best || states[l.id].updatedAt > states[best.id].updatedAt) ? l : best, null);
  if (recent && !isDone(states, recent)) return recent;
  const from = recent ? group.indexOf(recent) + 1 : 0;
  return group.slice(from).find(l => !isDone(states, l)) || group.find(l => !isDone(states, l)) || group[0];
}

// 개념 학습 목차 (B2): 왼쪽 과목 목록 + 오른쪽 '이어서 학습' 카드와 단원 2열 목록
// last: 전체에서 마지막으로 본 단원, onOpen(lesson): 그 단원 열기
export default function LessonIndex({ config, cert, lessons, states, last, onOpen }) {
  const subjects = [...new Set(lessons.map(l => l.subject))];
  const [selected, setSelected] = useState(() => last?.subject ?? subjects[0]);
  const [picked, setPicked] = useState(null); // 목록에서 고른 단원 (바로 열지 않고 아래 '학습하기'로 엶)
  const subject = subjects.includes(selected) ? selected : subjects[0];
  const label = key => config[cert].topics[key] ?? key;

  const group = lessons.filter(l => l.subject === subject);
  const complete = group.filter(l => isDone(states, l)).length;
  const resume = resumeLesson(group, states);
  const allDone = complete === group.length;
  const percent = Math.round(complete / group.length * 100);
  const pickedLesson = group.find(l => l.id === picked) || null;
  const pickedNo = pickedLesson ? group.indexOf(pickedLesson) + 1 : 0;

  return <section className="li-index" aria-labelledby="learning-index-heading">
    <h2 id="learning-index-heading" tabIndex={-1} className="li-visually-hidden">학습 목차</h2>

    {/* 왼쪽: 과목 목록 */}
    <nav className="ls-card li-subjects" aria-label="과목">
      <h3>과목</h3>
      {subjects.map(key => {
        const items = lessons.filter(l => l.subject === key);
        const count = items.filter(l => isDone(states, l)).length;
        const on = key === subject;
        return <button key={key} type="button" className={`li-subject${on ? ' is-active' : ''}`} aria-pressed={on} onClick={() => { setSelected(key); setPicked(null); }}>
          <span className="li-subject-top"><strong>{label(key)}</strong><span>{count}/{items.length}</span></span>
          <span className="li-bar" aria-hidden="true"><span style={{ width: `${Math.round(count / items.length * 100)}%` }} /></span>
        </button>;
      })}
    </nav>

    {/* 오른쪽: 이어서 학습 + 단원 목록 */}
    <div className="ls-card li-main">
      <div className="li-hero">
        <div className="li-hero-text">
          <div className="ls-ring li-ring" style={{ '--ls-progress': `${percent}%` }} role="img" aria-label={`${label(subject)} ${group.length}개 단원 중 ${complete}개 완료`}>
            <span>{complete}/{group.length}</span>
          </div>
          <div>
            <span className="li-hero-sub">{label(subject)} · {allDone ? '모두 완료' : '다음 단원'}</span>
            <h3 className="li-hero-title">{resume.title}</h3>
          </div>
        </div>
        <div className="li-hero-actions">
          <button type="button" className="li-btn li-btn-primary" onClick={() => onOpen(resume)}>
            {allDone ? '다시 보기' : '이어서 학습'}
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true"><path d="M9 6l6 6-6 6" /></svg>
          </button>
        </div>
      </div>

      <h4 className="li-list-title">전체 단원 {group.length}개</h4>
      <ul className="li-lessons">
        {group.map((lesson, i) => {
          const finished = isDone(states, lesson);
          const chosen = lesson.id === picked;
          return <li key={lesson.id}>
            <button type="button" className={`li-lesson${finished ? ' is-done' : ''}${chosen ? ' is-picked' : ''}`}
              aria-pressed={chosen} title={lesson.title}
              onClick={() => setPicked(chosen ? null : lesson.id)}>
              <span className="li-dot" aria-hidden="true">{finished ? '✓' : i + 1}</span>
              <span className="li-lesson-title">{plainTitle(lesson.title)}</span>
              <span className="li-visually-hidden">{finished ? ' (완료)' : ''}</span>
            </button>
          </li>;
        })}
      </ul>

      {/* 단원을 고르면 아래에 '학습하기' 버튼이 나타남 (잘못 눌러도 바로 넘어가지 않도록) */}
      {pickedLesson && <div className="li-pick" role="region" aria-label="선택한 단원">
        <div className="li-pick-text">
          <span>선택한 단원 · {isDone(states, pickedLesson) ? '완료한 단원' : `${pickedNo} / ${group.length}`}</span>
          <strong>{pickedNo}. {plainTitle(pickedLesson.title)}</strong>
        </div>
        <div className="li-pick-actions">
          <button type="button" className="li-btn" onClick={() => setPicked(null)}>취소</button>
          <button type="button" className="li-btn li-btn-primary" onClick={() => onOpen(pickedLesson)}>
            학습하기
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true"><path d="M9 6l6 6-6 6" /></svg>
          </button>
        </div>
      </div>}
    </div>
  </section>;
}