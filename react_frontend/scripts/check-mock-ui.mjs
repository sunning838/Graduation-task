import assert from 'node:assert/strict';
import { createServer } from 'vite';
import { createElement } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';

const server = await createServer({ server: { middlewareMode: true } });
try {
  const { ExamQuestion, PreparationProgress, default: MockExam } = await server.ssrLoadModule('/src/MockExam.jsx');
  const progress = renderToStaticMarkup(createElement(PreparationProgress, { preparation: {
    message: '시험 문제를 준비하고 있습니다.', progress: { total: 20, ready: 7, remaining: 13, percent: 35 }
  } }));
  assert.match(progress, /총 20문항 중 7문항 준비 완료/);
  assert.match(progress, /13문항 남음/);
  assert.match(progress, /35% 완료/);
  assert.match(progress, /max="20" value="7"/);
  assert.doesNotMatch(progress, /출제 가능/);
  const item = { id: 'q', position: 0, topic_label: '데이터베이스 구축', question: '**다음 SQL** 결과는?',
    options: ['1) Alpha', 'Beta', 'Gamma', 'Delta'], code_block: 'SELECT 1;', table_data: '| A |\n|---|\n| 1 |',
    selected_answer: 2, correct_answer: 2, explanation: 'SECRET_EXPLANATION', is_correct: true };
  const render = extra => renderToStaticMarkup(createElement(ExamQuestion, { item, onAnswer() {}, ...extra }));
  const taking = render({ submitted: false });
  assert.match(taking, /<strong>다음 SQL<\/strong>/);
  assert.match(taking, /SELECT 1;/);
  assert.match(taking, /<table>/);
  assert.match(taking, /aria-pressed="true"/);
  assert.doesNotMatch(taking, /SECRET_EXPLANATION|1\) Alpha/);
  assert.match(render({ submitted: true }), /SECRET_EXPLANATION/);
  assert.match(render({ submitted: true }), /정답: 2/);
  assert.equal((render({ busy: true }).match(/disabled=""/g) || []).length, 4);
  assert.match(renderToStaticMarkup(createElement(MockExam, { cert: 'EIP', label: '정보처리기사' })), /시험 정보를 불러오는 중/);
  console.log('Mock UI checks passed: Markdown, table, code, saved answer, hidden explanations, result, busy, loading.');
} finally { await server.close(); }
