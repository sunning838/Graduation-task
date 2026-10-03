import assert from 'node:assert/strict';
import { createServer } from 'vite';
import { createElement } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';

const server = await createServer({ server: { middlewareMode: true } });
try {
  const { SummaryContent } = await server.ssrLoadModule('/src/SummaryNotes.jsx');
  const { summaryMarkdown } = await server.ssrLoadModule('/src/summaryDownload.js');
  const props = { label: '정보처리기사', onGenerate() {}, onReload() {}, onDownload() {} };
  const data = { weakness: { message: '기록 부족', ranking: [], min_attempts: 5, accuracy_threshold: 60 },
    notes: [], can_generate: false, needs_update: false, generating: false, issues: [] };
  const render = extra => renderToStaticMarkup(createElement(SummaryContent, { ...props, data, ...extra }));
  const empty = render();
  assert.match(empty, /취약점 개념 요약/);
  assert.match(empty, /기록 부족/);
  assert.match(empty, /disabled=""[^>]*>개념 요약 생성/);
  assert.match(empty, /disabled=""[^>]*>요약 다운로드/);
  assert.doesNotMatch(empty, /오답노트|FRONTEND SAMPLE/);

  const note = { id: 'a', topic: 'database', label: '데이터베이스 구축', markdown: '### 핵심 개념\n\n**정규화**를 복습합니다.',
    created_at: '2026-10-03T00:00:00+00:00', basis: { correct: 1, total: 5, accuracy: 20 }, stale: true, is_current_topic: true };
  data.notes = [note];
  data.can_generate = true;
  data.needs_update = true;
  data.issues = [{ topic: 'database', label: '데이터베이스 구축', message: '생성 실패. 기존 요약은 유지됩니다.' }];
  const failed = render({ error: '서버 오류' });
  assert.match(failed, /서버 오류/);
  assert.match(failed, /정규화/);
  assert.match(failed, /기존 요약은 유지됩니다/);
  assert.match(failed, /최신 기록으로 갱신/);
  assert.doesNotMatch(failed, /disabled=""[^>]*>요약 다운로드/);
  assert.match(render({ busy: true }), /disabled=""[^>]*>요약 생성 중…/);
  assert.match(render({ busy: true }), /정규화/);

  const exported = summaryMarkdown(props.label, data.notes);
  assert.ok(exported.includes(note.markdown));
  assert.ok(exported.includes(note.created_at));
  assert.match(exported, /정보처리기사 취약점 개념 요약/);
  assert.match(exported, /갱신이 필요한/);
  note.is_current_topic = false;
  data.can_generate = false;
  assert.match(render(), /현재 취약 과목에는 포함되지 않습니다/);
  assert.match(render({ loading: true }), /role="status"/);
  console.log('Summary UI/download checks passed (empty, saved, stale, failure, busy, previous topic).');
} finally {
  await server.close();
}
