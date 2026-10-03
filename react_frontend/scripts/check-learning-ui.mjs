import assert from 'node:assert/strict';
import { createServer } from 'vite';
import { createElement } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';

// Offline component rendering checks; never calls Gemini or the project database.
const server = await createServer({ server: { middlewareMode: true } });
try {
  const { default: Dashboard } = await server.ssrLoadModule('/src/LearningDashboard.jsx');
  const { default: Weakness } = await server.ssrLoadModule('/src/WeaknessTraining.jsx');
  const noop = () => {};
  const stats = {
    has_records: false, total_solved: 0, correct_solved: 0, accuracy: null,
    today_solved: 0, daily_goal: 50, goal_rate: 0, unclassified_solved: 0,
    subjects: [{ topic: 'database', label: '데이터베이스 구축', total: 0, correct: 0, accuracy: null, status: 'unlearned' }],
    weakness: { status: 'insufficient_data', message: '분석할 기록 부족', focus: null, ranking: [],
      min_attempts: 5, accuracy_threshold: 60, insufficient_subjects: ['database'] },
  };
  const empty = renderToStaticMarkup(createElement(Dashboard, { stats, onRetry: noop }));
  assert.match(empty, /아직 풀이 기록이 없습니다/);
  assert.match(empty, /미학습/);
  assert.doesNotMatch(empty, /0문제 정답 · 0%/);
  const missing = renderToStaticMarkup(createElement(Weakness, { cert: 'EIP', label: '정보처리기사', stats, onRefresh: noop, onGeneralPractice: noop }));
  assert.match(missing, /분석할 기록 부족/);
  assert.match(missing, /일반 문제 풀기/);
  assert.match(missing, /disabled=""[^>]*>취약점 문제 시작/);
  assert.doesNotMatch(missing, /FRONTEND SAMPLE|트랜잭션의 ACID/);

  const focus = { topic: 'database', label: '데이터베이스 구축', total: 5, correct: 2, accuracy: 40, status: 'analyzed' };
  stats.has_records = true;
  stats.subjects = [focus];
  stats.weakness = { ...stats.weakness, status: 'ready', focus, ranking: [focus], message: '5문제 중 2문제 정답', insufficient_subjects: [] };
  const ready = renderToStaticMarkup(createElement(Weakness, { cert: 'EIP', label: '정보처리기사', stats, onRefresh: noop }));
  assert.match(ready, /5문제 중 2문제 정답/);
  assert.doesNotMatch(ready, /disabled=""[^>]*>취약점 문제 시작/);

  const error = renderToStaticMarkup(createElement(Dashboard, { stats: null, error: '조회 실패', onRetry: noop }));
  assert.match(error, /role="alert"/);
  assert.match(error, /다시 조회/);
  const loading = renderToStaticMarkup(createElement(Dashboard, { stats: null, loading: true, onRetry: noop }));
  assert.match(loading, /role="status"/);
  console.log('Learning UI render checks passed (empty, ready, loading, error).');
} finally {
  await server.close();
}
