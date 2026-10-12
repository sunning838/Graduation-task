import assert from 'node:assert/strict';
import { createRequire } from 'node:module';
import { fileURLToPath, pathToFileURL } from 'node:url';
const require = createRequire(new URL('../../react_frontend/package.json', import.meta.url));
const { createServer } = await import(pathToFileURL(require.resolve('vite')));
const { createElement } = require('react');
const { renderToStaticMarkup } = require('react-dom/server');
const server = await createServer({ root: fileURLToPath(new URL('../../react_frontend/', import.meta.url)), server: { middlewareMode: true } });
try {
  const { default: ChatMessage, Visual } = await server.ssrLoadModule('/src/learning/components/ChatMessage.jsx');
  const render = (Component, props) => renderToStaticMarkup(createElement(Component, props));
  const visual = { kind: 'graph', title: '생명 주기', purpose: '모형 분류', layout: 'hierarchy', nodes: ['생명 주기', '폭포수', '프로토타입'],
    edges: [{ from: 0, to: 1, relation: '하위 유형', label: '모형' }, { from: 0, to: 2, relation: '하위 유형', label: '모형' }], dot: 'digraph G { PRIVATE_CODE }' };
  const graph = render(Visual, { visual });
  assert.match(graph, /<svg/); assert.match(graph, /폭포수/); assert.match(graph, /프로토타입/);
  assert.doesNotMatch(graph, /PRIVATE_CODE|digraph/);
  assert.match(render(Visual, { visual: { kind: 'graph' } }), /도식을 표시하지 못했습니다/);
  assert.match(render(ChatMessage, { message: { role: 'assistant', text: '| A | B |\n|---|---|\n| 1 | 2 |' } }), /<table>/);
  assert.doesNotMatch(render(Visual, { visual: { ...visual, nodes: ['<script>alert(1)</script>'], edges: [] } }), /<script>/);
  const { default: LearningPage } = await server.ssrLoadModule('/src/learning/pages/LearningPage.jsx');
  assert.match(render(LearningPage, {}), /학습 목록을 불러오고/);
  console.log('Concept UI checks passed: SVG hierarchy, text escaping, Markdown table, fallback, initial loading.');
} finally { await server.close(); }
