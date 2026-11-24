// Browser-only reset checks. Every API response is a fixture; real records are never reset.
import assert from 'node:assert/strict';
import http from 'node:http';
import fs from 'node:fs/promises';
import path from 'node:path';
import { createRequire } from 'node:module';
import { fileURLToPath, pathToFileURL } from 'node:url';

const require = createRequire(process.env.PRESENTATION_TEST_NODE_MODULES
  ? pathToFileURL(path.join(process.env.PRESENTATION_TEST_NODE_MODULES, '_entry.cjs'))
  : new URL('../../react_frontend/package.json', import.meta.url));
const { chromium } = require('playwright');
const root = fileURLToPath(new URL('../../react_frontend/dist/', import.meta.url));
const output = new URL('./.learning-reset-ui/', import.meta.url);
await fs.mkdir(output, { recursive: true });
const server = http.createServer(async (req, res) => {
  try {
    const pathname = new URL(req.url, 'http://127.0.0.1').pathname;
    const filename = path.resolve(root, '.' + (pathname === '/' ? '/index.html' : pathname));
    if (!filename.startsWith(root)) throw new Error('Invalid asset');
    const data = await fs.readFile(filename);
    res.setHeader('Content-Type', /\.(m?js)$/.test(filename) ? 'text/javascript' : filename.endsWith('.css') ? 'text/css' : filename.endsWith('.svg') ? 'image/svg+xml' : 'text/html');
    res.end(data);
  } catch { res.writeHead(404); res.end(); }
});
await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
const base = 'http://127.0.0.1:' + server.address().port;
const browser = await chromium.launch({ headless: true, executablePath: process.env.CHROME_PATH || 'C:/Program Files/Google/Chrome/Application/chrome.exe' });
const topics = { software_design: '소프트웨어 설계', software_development: '소프트웨어 개발', database: '데이터베이스 구축', programming_language: '프로그래밍 언어 활용', info_system: '정보시스템 구축 관리' };
const lessons = [
  { id: 'a', version: 'v1', cert: 'EIP', subject: 'software_design', title: '1. 첫 단원' },
  { id: 'b', version: 'v1', cert: 'EIP', subject: 'software_design', title: '2. 다음 단원' },
];
const json = (route, value, status = 200) => route.fulfill({ status, contentType: 'application/json', body: JSON.stringify(value) });
const results = [];
let current;

async function setup(options = {}) {
  const page = await browser.newPage({ viewport: { width: 1440, height: 1000 } });
  page.setDefaultTimeout(10000);
  const state = { calls: [], errors: [], unexpected: [], progressCalls: [], teachCalls: [],
    progress: { a: { status: 'complete', updatedAt: 1 } }, revision: 0, clock: 10, reset: false };
  current = { page, state };
  page.on('pageerror', error => state.errors.push(error.message));
  await page.route(url => url.pathname.startsWith('/api/'), async route => {
    const request = route.request(), p = new URL(request.url()).pathname;
    const body = request.method() === 'POST' ? request.postDataJSON() : null;
    if (p === '/api/certifications') return json(route, { certifications: [{ id: 'EIP', label: '정보처리기사', option_count: 4 }] });
    if (p === '/api/learning/catalog') return json(route, { config: { EIP: { label: '정보처리기사', topics } }, lessons });
    if (p === '/api/learning/progress') {
      if (body) {
        assert.equal(body.reset_revision, state.revision);
        assert.ok(lessons.some(lesson => lesson.id === body.lesson_id));
        state.progressCalls.push(structuredClone(body));
        const complete = body.complete || state.progress[body.lesson_id]?.status === 'complete';
        state.progress[body.lesson_id] = { status: complete ? 'complete' : 'studying', updatedAt: ++state.clock };
      }
      return json(route, { states: state.progress, reset_revision: state.revision });
    }
    if (p === '/api/stats') {
      const subjects = Object.entries(topics).map(([topic, label]) => ({
        topic, label, total: state.reset ? 0 : 5, correct: state.reset ? 0 : 1,
        accuracy: state.reset ? null : 20, status: state.reset ? 'unlearned' : 'analyzed',
      }));
      return json(route, {
        subjects, has_records: !state.reset, total_solved: state.reset ? 0 : 25, correct_solved: state.reset ? 0 : 5,
        today_solved: state.reset ? 0 : 5, daily_goal: 50, goal_rate: state.reset ? 0 : 10, accuracy: state.reset ? null : 20, unclassified_solved: 0,
        weakness: { status: state.reset ? 'insufficient_data' : 'ready', message: state.reset ? '취약점 분석을 위해 문제를 풀어 주세요.' : '데이터베이스 구축이 취약합니다.',
          ranking: state.reset ? [] : [subjects[2]], min_attempts: 5, accuracy_threshold: 60 },
      });
    }
    if (p === '/api/learning/reset') {
      assert.equal(body.confirm, true); assert.equal(body.cert, 'EIP');
      assert.ok(body.request_id);
      state.calls.push(structuredClone(body));
      if (options.failFirst && state.calls.length === 1) return json(route, { detail: { message: '초기화하지 못했습니다. 다시 시도해 주세요.' } }, 500);
      if (options.hold) await new Promise(resolve => { state.release = resolve; });
      state.reset = true;
      if (body.include_progress) { state.progress = {}; state.revision++; }
      return json(route, { ...body, deleted: {}, message: '학습 기록을 초기화했습니다.' });
    }
    if (p === '/api/learning/teach') {
      state.teachCalls.push(structuredClone(body));
      return json(route, { text: '### ' + lessons.find(lesson => lesson.id === body.lesson_id).title + '\n\n원문에 근거한 설명입니다.', visual: null });
    }
    if (p === '/api/learning/presentations/lookup') return json(route, { status: 'idle', slides: [], segments: [], images: { total: 1, ready: 0, failed: 0 } });
    if (p === '/api/chat') return json(route, { conversation_id: 'chat-1', answer: '기존 대화 답변입니다.' });
    if (p === '/api/quiz') return json(route, { attempt_id: 'quiz-a', question: '문제입니다.', topic: 'database', topic_label: '데이터베이스 구축', options: ['a', 'b', 'c', 'd'] });
    if (p === '/api/question-bank/availability') return json(route, { cert: 'EIP', subjects: Object.entries(topics).map(([topic, label]) => ({ topic, label, ready: 10 })) });
    if (p.startsWith('/api/mock-exams/')) return json(route, { detail: { code: 'EXAM_NOT_FOUND', message: '삭제된 시험입니다.' } }, 404);
    if (p === '/api/mock-preparations') return json(route, { detail: { code: 'LEARNING_RESET', message: '학습 기록이 초기화되었습니다.' } }, 409);
    state.unexpected.push(request.method() + ' ' + p);
    return json(route, { detail: { message: 'Unexpected test route: ' + p } }, 404);
  });
  await page.goto(base);
  await page.getByRole('button', { name: '학습 시작하기', exact: true }).click();
  await page.getByRole('img', { name: '소프트웨어 설계 2개 단원 중 1개 완료', exact: true }).waitFor();
  await page.locator('.li-hero').getByRole('heading', { name: lessons[1].title, exact: true }).waitFor();
  return { page, state };
}

const profile = page => page.getByTitle('나의 학습 현황', { exact: true }).click();
const open = page => page.getByRole('button', { name: '학습 데이터 초기화', exact: true }).click();
const dialog = page => page.getByRole('dialog', { name: '학습 데이터 초기화', exact: true });
const confirm = page => dialog(page).getByRole('button', { name: '기록 삭제하기', exact: true }).click();
const clean = state => { assert.deepEqual(state.errors, []); assert.deepEqual(state.unexpected, []); };
async function practiceSpace(page) {
  await page.getByTitle('문제풀이 화면으로 가기', { exact: true }).click();
  if (await page.locator('.sidebar.collapsed').count()) await page.locator('.sidebar-toggle').click();
}
async function firstLesson(page, state) {
  await page.getByTitle(lessons[0].title, { exact: true }).click();
  const choice = page.getByRole('region', { name: '선택한 단원', exact: true });
  await choice.getByRole('button', { name: '학습하기', exact: true }).click();
  await page.locator('.ls-title').getByRole('heading', { name: lessons[0].title, exact: true }).waitFor();
  await page.getByRole('region', { name: '강의 본문', exact: true }).waitFor();
  assert.equal(state.progressCalls.at(-1).lesson_id, lessons[0].id);
  assert.equal(state.teachCalls.at(-1).lesson_id, lessons[0].id);
}

try {
  {
    const { page, state } = await setup();
    await profile(page); await open(page);
    assert.equal(await dialog(page).getByRole('checkbox').isChecked(), false);
    await page.screenshot({ path: fileURLToPath(new URL('confirm-dark.png', output)), fullPage: true });
    await dialog(page).getByRole('button', { name: '취소', exact: true }).click();
    await dialog(page).waitFor({ state: 'hidden' }); assert.equal(state.calls.length, 0);
    await page.locator('.theme-toggle').click(); await open(page);
    await page.screenshot({ path: fileURLToPath(new URL('confirm-light.png', output)), fullPage: true });
    await page.setViewportSize({ width: 390, height: 844 });
    await page.screenshot({ path: fileURLToPath(new URL('confirm-mobile.png', output)), fullPage: true });
    assert.ok(await dialog(page).evaluate(element => element.getBoundingClientRect().width <= innerWidth));
    await page.keyboard.press('Escape'); await dialog(page).waitFor({ state: 'hidden' });
    assert.equal(state.calls.length, 0); clean(state);
    results.push('Confirmation default, cancel/Escape, dark/light/mobile layout');
    await page.close();
  }
  {
    const { page, state } = await setup({ hold: true });
    await practiceSpace(page);
    await page.locator('.menu-item').filter({ hasText: 'AI 튜터' }).click();
    await page.getByPlaceholder('AI 튜터에게 질문해보세요').fill('기존 질문');
    await page.locator('.send-button').click();
    await page.getByText('기존 대화 답변입니다.', { exact: true }).waitFor();
    await page.evaluate(() => {
      localStorage.setItem('mock-exam:EIP', '{"id":"old"}');
      localStorage.setItem('mock-exam:LREA_1', 'other-exam'); localStorage.setItem('unrelated', 'keep');
    });
    await profile(page); await open(page); await confirm(page);
    await dialog(page).getByRole('button', { name: '초기화 중…', exact: true }).waitFor();
    assert.ok(await dialog(page).getByRole('button', { name: '초기화 중…', exact: true }).isDisabled());
    assert.ok(await dialog(page).getByRole('checkbox').isDisabled());
    assert.ok(await dialog(page).getByRole('button', { name: '취소', exact: true }).isDisabled());
    await page.keyboard.press('Escape'); assert.ok(await dialog(page).isVisible());
    assert.equal(state.calls.length, 1); state.release();
    await page.getByText('문제풀이 기록을 초기화했습니다. 개념학습 진도는 유지했습니다.', { exact: true }).waitFor();
    assert.equal(state.calls[0].include_progress, false);
    assert.match(await page.locator('.total-stats').innerText(), /0/);
    assert.equal(await page.evaluate(() => localStorage.getItem('mock-exam:EIP')), null);
    assert.equal(await page.evaluate(() => localStorage.getItem('mock-exam:LREA_1')), 'other-exam');
    assert.equal(await page.evaluate(() => localStorage.getItem('unrelated')), 'keep');
    await page.locator('.menu-item').filter({ hasText: 'AI 튜터' }).click();
    await page.getByText('기존 대화 답변입니다.', { exact: true }).waitFor();
    await page.getByTitle('개념 학습으로 돌아가기', { exact: true }).click();
    await page.getByRole('img', { name: '소프트웨어 설계 2개 단원 중 1개 완료', exact: true }).waitFor();
    assert.equal(state.progress.a.status, 'complete');
    assert.equal(state.progressCalls.length, 0);
    clean(state); results.push('Default reset, busy lock, scoped storage cleanup, progress and chat retention');
    await page.close();
  }
  {
    const { page, state } = await setup();
    await page.getByRole('button', { name: '이어서 학습', exact: true }).click();
    await page.locator('.ls-title').getByRole('heading', { name: lessons[1].title, exact: true }).waitFor();
    await page.getByRole('region', { name: '강의 본문', exact: true }).waitFor();
    assert.equal(state.teachCalls.at(-1).lesson_id, lessons[1].id);
    await profile(page); await open(page); await dialog(page).getByRole('checkbox').check(); await confirm(page);
    await page.getByText('문제풀이 기록과 개념학습 진도를 초기화했습니다.', { exact: true }).waitFor();
    assert.equal(state.calls[0].include_progress, true); assert.deepEqual(state.progress, {}); assert.equal(state.revision, 1);
    await page.getByRole('button', { name: '← 이전 화면', exact: true }).click();
    await page.getByRole('heading', { name: '학습 목차', exact: true }).waitFor();
    await page.getByRole('img', { name: '소프트웨어 설계 2개 단원 중 0개 완료', exact: true }).waitFor();
    assert.equal(await page.locator('.ls-title').count(), 0, 'Reset clears the old active lesson');
    await firstLesson(page, state);
    assert.equal(state.progressCalls.at(-1).reset_revision, 1);
    assert.equal(state.progressCalls.at(-1).complete, false);
    assert.equal(state.progress.a.status, 'studying'); assert.equal(state.progress.b, undefined);
    clean(state); results.push('Optional reset clears progress and active lesson; explicit first lesson uses the new reset revision');
    await page.close();
  }
  {
    const { page, state } = await setup({ failFirst: true });
    await profile(page); await open(page); await confirm(page);
    await dialog(page).getByRole('alert').waitFor(); assert.equal(state.reset, false);
    assert.match(await page.locator('.total-stats').innerText(), /25/);
    assert.equal(state.progress.a.status, 'complete');
    const requestId = state.calls[0].request_id;
    await confirm(page); await dialog(page).waitFor({ state: 'hidden' });
    assert.equal(state.calls.length, 2); assert.equal(state.calls[1].request_id, requestId);
    clean(state); results.push('Failed reset preserves records and retries with the same request ID');
    await page.close();
  }
  for (const saved of [{ id: 'discarded-exam' }, { pending: { request_id: 'discarded-preparation', distribution: [{ topic: 'database', count: 1 }] } }]) {
    const { page, state } = await setup();
    await page.evaluate(value => localStorage.setItem('mock-exam:EIP', JSON.stringify(value)), saved);
    await practiceSpace(page);
    await page.locator('.menu-item').filter({ hasText: '모의고사' }).click();
    await page.getByRole('button', { name: '모의고사 시작', exact: true }).waitFor();
    assert.equal(await page.evaluate(() => localStorage.getItem('mock-exam:EIP')), null);
    clean(state); await page.close();
  }
  results.push('Discarded exam/preparation recovery');
  console.log(JSON.stringify({ passed: results, screenshots: fileURLToPath(output) }, null, 2));
} catch (error) {
  if (current && !current.page.isClosed()) {
    await current.page.screenshot({ path: fileURLToPath(new URL('failure.png', output)), fullPage: true });
    console.error('Browser errors:', current.state.errors, 'Unexpected requests:', current.state.unexpected);
    console.error((await current.page.locator('body').innerText()).slice(0, 1500));
  }
  throw error;
} finally {
  await browser.close(); server.closeAllConnections(); await new Promise(resolve => server.close(resolve));
}
