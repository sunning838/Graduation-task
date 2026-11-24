// Current UI browser regressions. All normal-mode API, image and audio responses are fixtures.
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import { createRequire } from 'node:module';
import { fileURLToPath, pathToFileURL } from 'node:url';

const require = createRequire(process.env.PRESENTATION_TEST_NODE_MODULES
  ? pathToFileURL(path.join(process.env.PRESENTATION_TEST_NODE_MODULES, '_entry.cjs'))
  : new URL('../../react_frontend/package.json', import.meta.url));
const { chromium } = require('playwright');
const { lectureFromSegments, MAX_SLIDES } = await import(new URL('../../react_frontend/src/learning/components/lectureSlides.js', import.meta.url));
const longSegments = Array.from({ length: 18 }, (_, index) => ({ index, markdown: '### 개념 ' + index + '\n\n원문 설명입니다.' }));
const longLecture = lectureFromSegments(longSegments);
assert.equal(MAX_SLIDES, 16);
assert.equal(longLecture.slides.length, 16);
assert.deepEqual(longLecture.slides.flatMap(slide => slide.segment_indices), longSegments.map(segment => segment.index));
assert.ok(longLecture.slides.every(slide => slide.image_status === 'pending'));

const base = process.env.PRESENTATION_TEST_URL || 'http://127.0.0.1:5188';
const backend = process.env.PRESENTATION_TEST_API_URL || 'http://127.0.0.1:8018';
const output = new URL('./.presentation-ui/', import.meta.url);
await fs.mkdir(output, { recursive: true });
const liveMode = process.argv.includes('--live-cache');
const live = liveMode ? JSON.parse(await fs.readFile(new URL('./.presentation-live.json', import.meta.url), 'utf8')) : null;
const paragraphs = [
  '### 스택의 구조\n\n스택은 마지막에 들어온 데이터가 먼저 나오는 자료 구조입니다. 데이터를 넣은 순서와 꺼내는 순서를 구분해 생각해야 합니다. 접시를 위로 쌓았다가 맨 위의 접시부터 꺼내는 모습을 떠올리면 이해할 수 있습니다.',
  '- 한쪽 끝에서 데이터를 넣고 꺼냅니다. 이 위치를 스택의 맨 위라고 부릅니다. 먼저 들어온 값은 그 위의 값들이 모두 빠진 뒤에 꺼낼 수 있습니다. 스택의 예시를 살펴볼 때는 현재 맨 위에 어떤 값이 놓여 있는지 확인합니다.',
  '### PUSH 과정\n\nPUSH는 스택의 맨 위에 데이터를 추가하는 연산입니다. A와 B를 차례로 넣으면 A 위에 B가 놓입니다. 이때 새 값을 넣을 때마다 맨 위의 위치도 함께 달라진다는 점을 기억합니다.',
  '- 마지막으로 넣은 데이터가 맨 위에 있습니다. 추가된 값은 다음에 꺼낼 첫 번째 후보가 됩니다. 현재 저장된 값들의 순서를 아래에서 위로 기록하면서 PUSH 전후의 상태를 비교해 볼 수 있습니다.',
  '### POP 과정\n\nPOP은 스택의 맨 위에서 데이터를 꺼내는 연산입니다. A 다음에 B를 넣었다면 먼저 꺼내는 값은 B입니다. 값을 꺼낸 뒤에는 그 아래에 있던 값이 새로운 맨 위가 됩니다.',
  '- POP 결과와 남은 스택을 구분합니다. 꺼낸 값은 반환 결과이고 아래에 남은 값들은 다음 연산에서 사용합니다. 연산이 반복될 때마다 현재 맨 위를 확인하면 후입선출 원리를 일관되게 적용할 수 있습니다.',
];
const segments = paragraphs.map((markdown, index) => ({ index, markdown, start: index * 5, end: (index + 1) * 5 }));
const text = paragraphs.join('\n\n');
const slides = ['스택의 구조', 'PUSH 과정', 'POP 과정'].map((title, index) => ({
  id: 'slide-' + index, title, segment_indices: [index * 2, index * 2 + 1],
  bullets: [['마지막에 들어온 데이터가 먼저 나옵니다.', '입출력 위치는 스택의 맨 위입니다.'], ['맨 위에 데이터를 추가합니다.'], ['맨 위의 데이터를 꺼냅니다.']][index],
  image_status: 'pending', image_url: null, caption: '',
}));
const lessons = [
  { id: live?.lesson_id || 'test-lesson', version: 'v1', cert: 'EIP', subject: 'software_design', title: live ? '소프트웨어 생명 주기' : '1. 자료 구조의 스택' },
  { id: 'test-next', version: 'v1', cert: 'EIP', subject: 'software_design', title: '2. 자료 구조의 큐' },
  { id: 'test-selected', version: 'v1', cert: 'EIP', subject: 'database', title: '1. 정규화' },
];
const topics = { software_design: '소프트웨어 설계', database: '데이터베이스 구축' };
const initialProgress = { [lessons[0].id]: { status: 'complete', updatedAt: 1 }, [lessons[1].id]: { status: 'studying', updatedAt: 2 } };
const packageId = 'a'.repeat(64);
const json = (route, data, status = 200) => route.fulfill({ status, contentType: 'application/json', body: JSON.stringify(data) });
const browser = await chromium.launch({ headless: true, executablePath: process.env.CHROME_PATH || 'C:/Program Files/Google/Chrome/Application/chrome.exe' });
const results = [];
let current;

function presentation(status) {
  const ready = status === 'ready' ? slides.length : ['generating', 'partial'].includes(status) ? 1 : 0;
  return {
    package_id: packageId, status, segments, images: { total: slides.length, ready, failed: status === 'partial' ? slides.length - ready : 0 },
    retryable: status === 'partial',
    slides: slides.map((slide, index) => ({
      ...slide, image_status: index < ready ? 'ready' : status === 'partial' ? 'failed' : 'pending',
      image_url: index < ready ? '/api/learning/presentations/' + packageId + '/images/' + slide.id : null,
    })),
  };
}

async function setup(progress = initialProgress) {
  const page = await browser.newPage({ viewport: { width: 1600, height: 1000 } });
  page.setDefaultTimeout(10000);
  const state = { progress: structuredClone(progress), clock: 10, phase: 'idle', starts: [], lookups: [], polls: 0, retries: 0,
    progressCalls: [], teachCalls: [], chats: [], tts: [], errors: [], unexpected: [] };
  current = { page, state };
  page.on('pageerror', error => state.errors.push(error.message));
  await page.route(url => url.pathname.startsWith('/api/'), async route => {
    const request = route.request(), p = new URL(request.url()).pathname;
    const body = request.method() === 'POST' ? request.postDataJSON() : null;
    if (p === '/api/certifications') return json(route, { certifications: [{ id: 'EIP', label: '정보처리기사', option_count: 4 }] });
    if (p === '/api/stats') return json(route, {});
    if (p === '/api/learning/catalog') return json(route, { config: { EIP: { label: '정보처리기사', topics } }, lessons });
    if (p === '/api/learning/progress') {
      if (body) {
        assert.equal(body.reset_revision, 0);
        assert.ok(lessons.some(lesson => lesson.id === body.lesson_id));
        state.progressCalls.push(structuredClone(body));
        const complete = body.complete || state.progress[body.lesson_id]?.status === 'complete';
        state.progress[body.lesson_id] = { status: complete ? 'complete' : 'studying', updatedAt: ++state.clock };
      }
      return json(route, { states: state.progress, reset_revision: 0 });
    }
    if (p === '/api/learning/teach') {
      state.teachCalls.push(structuredClone(body));
      const lesson = lessons.find(item => item.id === body.lesson_id);
      assert.ok(lesson);
      assert.equal(body.variant, 'basic');
      return json(route, { text: body.lesson_id === lessons[0].id ? live?.text || text : '### ' + lesson.title + '\n\n선택한 단원의 개념 설명입니다.', visual: null });
    }
    if (p === '/api/chat') {
      state.chats.push(structuredClone(body));
      return json(route, { conversation_id: 'study-chat', answer: '후입선출은 나중에 넣은 데이터를 먼저 꺼내는 방식입니다.' });
    }
    if (live && p.startsWith('/api/learning/presentations')) {
      // Cache mode may only look up or read already generated files.
      assert.ok(request.method() === 'GET' || p === '/api/learning/presentations/lookup', 'Live cache checks cannot generate or retry images');
      return route.fulfill({ response: await route.fetch({ url: backend + p }) });
    }
    if (p === '/api/learning/presentations/lookup') {
      state.lookups.push(structuredClone(body));
      if (body.lesson_id !== lessons[0].id) return json(route, { status: 'idle', slides: [], segments: [], images: { total: 1, ready: 0, failed: 0 } });
      return json(route, presentation(state.phase));
    }
    if (p === '/api/learning/presentations') {
      state.starts.push(structuredClone(body)); state.phase = 'planning';
      return json(route, presentation(state.phase));
    }
    if (p === '/api/learning/presentations/' + packageId + '/retry') {
      state.retries++; state.phase = 'ready'; return json(route, presentation(state.phase));
    }
    if (p === '/api/learning/presentations/' + packageId) {
      state.polls++;
      if (state.phase === 'planning') state.phase = 'generating';
      return json(route, presentation(state.phase));
    }
    if (p.startsWith('/api/learning/presentations/' + packageId + '/images/')) return route.fulfill({
      contentType: 'image/svg+xml', body: '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 160 90"><rect width="160" height="90" fill="lightblue"/></svg>',
    });
    if (p === '/api/learning/tts') {
      state.tts.push(structuredClone(body));
      return json(route, { audio_url: '/api/learning/tts/offline/audio', timeline: segments });
    }
    if (p === '/api/learning/tts/offline/audio') {
      const rate = 8000, wav = Buffer.alloc(44 + 30 * rate * 2);
      wav.write('RIFF', 0); wav.writeUInt32LE(wav.length - 8, 4); wav.write('WAVEfmt ', 8);
      wav.writeUInt32LE(16, 16); wav.writeUInt16LE(1, 20); wav.writeUInt16LE(1, 22);
      wav.writeUInt32LE(rate, 24); wav.writeUInt32LE(rate * 2, 28); wav.writeUInt16LE(2, 32); wav.writeUInt16LE(16, 34);
      wav.write('data', 36); wav.writeUInt32LE(wav.length - 44, 40);
      const match = request.headers().range?.match(/bytes=(\d+)-(\d*)/);
      if (match) {
        const start = Number(match[1]), end = match[2] ? Math.min(Number(match[2]), wav.length - 1) : wav.length - 1;
        return route.fulfill({ status: 206, contentType: 'audio/wav', headers: { 'Accept-Ranges': 'bytes', 'Content-Range': 'bytes ' + start + '-' + end + '/' + wav.length }, body: wav.subarray(start, end + 1) });
      }
      return route.fulfill({ contentType: 'audio/wav', headers: { 'Accept-Ranges': 'bytes' }, body: wav });
    }
    state.unexpected.push(request.method() + ' ' + p);
    return json(route, { detail: { message: 'Unexpected test route: ' + p } }, 404);
  });
  await page.goto(base);
  await page.getByRole('button', { name: '학습 시작하기', exact: true }).click();
  await page.getByRole('heading', { name: '학습 목차', exact: true }).waitFor();
  return { page, state };
}

async function assertOpened(page, state, lesson) {
  await page.locator('.ls-title').getByRole('heading', { name: lesson.title, exact: true }).waitFor();
  await page.getByRole('region', { name: '강의 본문', exact: true }).waitFor();
  assert.equal(state.progressCalls.at(-1).lesson_id, lesson.id);
  assert.ok(state.teachCalls.some(request => request.lesson_id === lesson.id));
  assert.deepEqual(state.errors, []);
  assert.deepEqual(state.unexpected, []);
}

async function chooseLesson(page, state, lesson) {
  await page.getByRole('navigation', { name: '과목', exact: true }).getByRole('button', { name: new RegExp(topics[lesson.subject]) }).click();
  const before = state.progressCalls.length;
  await page.getByTitle(lesson.title, { exact: true }).click();
  const choice = page.getByRole('region', { name: '선택한 단원', exact: true });
  await choice.waitFor();
  assert.equal(state.progressCalls.length, before, 'Selecting a lesson must not start it');
  await choice.getByRole('button', { name: '학습하기', exact: true }).click();
  await assertOpened(page, state, lesson);
}

async function showSlide(page, index) {
  const region = page.getByRole('region', { name: '강의 슬라이드', exact: true });
  await region.getByRole('button', { name: (index + 1) + '번 슬라이드 ' + slides[index].title, exact: true }).click();
  await region.getByRole('heading', { name: slides[index].title, exact: true }).waitFor();
}

async function decodeImage(page) {
  const image = page.getByRole('region', { name: '강의 슬라이드', exact: true }).locator('img');
  await image.waitFor(); await image.evaluate(element => element.decode());
  return image.getAttribute('src');
}

try {
  if (!live) {
    {
      const { page, state } = await setup({});
      await chooseLesson(page, state, lessons[0]);
      assert.equal(state.progress[lessons[0].id].status, 'studying');
      results.push('No progress: select the first lesson and use 학습하기');
      await page.close();
    }
    {
      const { page, state } = await setup();
      await page.locator('.li-hero').getByRole('heading', { name: lessons[1].title, exact: true }).waitFor();
      await page.getByRole('button', { name: '이어서 학습', exact: true }).click();
      await assertOpened(page, state, lessons[1]);
      assert.notEqual(lessons[0].id, lessons[1].id);
      results.push('Resume: opens the in-progress second lesson');
      await page.close();
    }
    {
      const { page, state } = await setup();
      await chooseLesson(page, state, lessons[2]);
      results.push('Subject/lesson selection: opens the chosen database lesson');
      await page.close();
    }
    {
      const completed = Object.fromEntries(lessons.slice(0, 2).map((lesson, index) => [lesson.id, { status: 'complete', updatedAt: index + 1 }]));
      const { page, state } = await setup(completed);
      await page.getByRole('button', { name: '다시 보기', exact: true }).click();
      await assertOpened(page, state, lessons[0]);
      assert.ok(lessons.slice(0, 2).every(lesson => state.progress[lesson.id].status === 'complete'));
      results.push('Completed subject: review preserves completion');
      await page.close();
    }
  }

  const { page, state } = await setup();
  await page.locator('.li-hero').getByRole('heading', { name: lessons[1].title, exact: true }).waitFor();
  await chooseLesson(page, state, lessons[0]);
  assert.equal(state.progress[lessons[0].id].status, 'complete', 'Opening the first lesson for review must not reset progress');
  results.push('Existing progress: explicit first-lesson selection is distinct from resume');
  const transcript = page.locator('.ls-reference-scroll');
  assert.ok(await transcript.isVisible());
  assert.equal(await page.getByRole('dialog', { name: '튜터에게 질문', exact: true }).isVisible(), false);

  if (live) {
    assert.equal(live.result.status, 'ready');
    const sources = new Set();
    for (let index = 0; index < live.result.slides.length; index++) {
      await page.getByRole('region', { name: '강의 슬라이드', exact: true }).getByRole('button', { name: new RegExp('^' + (index + 1) + '번 슬라이드 ') }).click();
      sources.add(await decodeImage(page));
      await page.screenshot({ path: fileURLToPath(new URL('live-slide-' + (index + 1) + '.png', output)) });
    }
    assert.equal(sources.size, live.result.slides.length);
    results.push('Live cache: each slide displays an existing image without generation');
  } else {
    await page.getByRole('button', { name: '설명 그림 만들기', exact: true }).waitFor();
    assert.equal(state.starts.length, 0); assert.equal(state.polls, 0);
    await page.getByRole('button', { name: '다음 슬라이드', exact: true }).click();
    await page.getByRole('region', { name: '강의 슬라이드', exact: true }).getByRole('heading', { name: 'PUSH 과정', exact: true }).waitFor();
    await page.getByRole('button', { name: '이전 슬라이드', exact: true }).click();
    await page.getByRole('button', { name: '음성으로 듣기', exact: true }).click();
    await page.waitForFunction(() => document.querySelector('audio')?.readyState >= 3);
    const audio = page.locator('audio');
    await audio.evaluate(element => element.pause());
    await page.getByLabel('강의 재생 위치', { exact: true }).press('End');
    await page.getByRole('region', { name: '강의 슬라이드', exact: true }).getByRole('heading', { name: 'POP 과정', exact: true }).waitFor();
    assert.ok(await audio.evaluate(element => element.currentTime >= 29.9));
    await page.getByLabel('강의 재생 속도', { exact: true }).selectOption('1.5');
    assert.equal(await audio.evaluate(element => element.playbackRate), 1.5);
    assert.deepEqual(state.tts, [{ lesson_id: lessons[0].id, variant: 'basic', text }]);
    await audio.evaluate(element => { element.currentTime = 21; element.dispatchEvent(new Event('timeupdate')); });
    await page.waitForFunction(() => document.querySelectorAll('.reference-paragraph')[4]?.getAttribute('aria-current') === 'true');
    await page.waitForFunction(() => document.querySelector('.ls-reference-scroll').scrollTop > 20);
    await transcript.hover(); await page.mouse.wheel(0, -3000);
    await page.waitForFunction(() => document.querySelector('.ls-reference-scroll').scrollTop < 5);
    await audio.evaluate(element => { element.currentTime = 26; element.dispatchEvent(new Event('timeupdate')); });
    await page.waitForFunction(() => document.querySelectorAll('.reference-paragraph')[5]?.getAttribute('aria-current') === 'true');
    // Let any smooth scroll settle before checking that manual reading remains at the top.
    await page.waitForTimeout(350);
    assert.ok(await transcript.evaluate(element => element.scrollTop < 5));
    await page.getByRole('button', { name: '3번 문단부터 듣기', exact: true }).click();
    await page.waitForFunction(() => document.querySelectorAll('.reference-paragraph')[2]?.getAttribute('aria-current') === 'true');
    await page.waitForFunction(() => document.querySelector('.ls-reference-scroll').scrollTop > 20);
    await audio.evaluate(element => element.pause());
    results.push('Transcript, audio seek/speed/highlight, manual reading and paragraph playback');

    await page.getByRole('button', { name: '튜터에게 질문', exact: true }).click();
    const drawer = page.getByRole('dialog', { name: '튜터에게 질문', exact: true });
    await drawer.getByPlaceholder('이 부분이 궁금해요…').fill('후입선출이 무슨 뜻인가요?');
    await drawer.getByRole('button', { name: '질문하기', exact: true }).click();
    await drawer.getByText('후입선출은 나중에 넣은 데이터를 먼저 꺼내는 방식입니다.', { exact: true }).waitFor();
    assert.equal(state.chats.length, 1);
    assert.equal(state.chats[0].cert, 'EIP');
    assert.ok(state.chats[0].message.includes(lessons[0].title));
    await drawer.getByRole('button', { name: '닫기', exact: true }).click();
    assert.equal(state.starts.length, 0, 'Navigation, audio and questions cannot generate images');
    assert.equal(state.polls, 0);

    await showSlide(page, 1);
    await page.getByRole('button', { name: '설명 그림 만들기', exact: true }).click();
    await showSlide(page, 0);
    const firstSource = await decodeImage(page);
    assert.deepEqual(state.starts, [{ lesson_id: lessons[0].id, variant: 'basic', text }]);
    state.phase = 'partial';
    await showSlide(page, 1);
    await page.getByRole('button', { name: '그림 다시 만들기', exact: true }).click();
    await decodeImage(page);
    assert.equal(state.retries, 1);
    const sources = new Set();
    for (let index = 0; index < slides.length; index++) {
      await showSlide(page, index); sources.add(await decodeImage(page));
    }
    assert.equal(sources.size, slides.length);
    await showSlide(page, 0);
    assert.equal(await decodeImage(page), firstSource, 'Retry preserves the completed first image');
    const lookups = state.lookups.length;
    await page.getByRole('button', { name: '목차', exact: true }).click();
    await chooseLesson(page, state, lessons[0]);
    await decodeImage(page);
    assert.ok(state.lookups.length > lookups);
    assert.equal(state.starts.length, 1, 'Reopening a cached lecture cannot generate images again');
    assert.equal(await page.getByRole('button', { name: '설명 그림 만들기', exact: true }).count(), 0);
    results.push('Manual generation for all three slides, failed-slide retry, completed images and cache reuse');
  }

  for (const [name, width, height] of [['desktop', 1600, 1000], ['laptop', 1366, 768], ['mobile', 390, 844]]) {
    await page.setViewportSize({ width, height });
    await page.locator('.ls-header').scrollIntoViewIfNeeded();
    assert.ok(await page.getByRole('region', { name: '강의 본문', exact: true }).isVisible());
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth + 2), false, name + ' horizontal overflow');
    await page.screenshot({ path: fileURLToPath(new URL((live ? 'live-' : '') + name + '.png', output)), fullPage: true });
  }
  assert.deepEqual(state.errors, []); assert.deepEqual(state.unexpected, []);
  results.push('Desktop/laptop/mobile layout and no browser errors');
  console.log(JSON.stringify({ passed: results, liveImage: liveMode, screenshots: fileURLToPath(output) }, null, 2));
} catch (error) {
  if (current && !current.page.isClosed()) {
    await current.page.screenshot({ path: fileURLToPath(new URL('failure.png', output)), fullPage: true });
    console.error('Browser errors:', current.state.errors, 'Unexpected requests:', current.state.unexpected);
    console.error((await current.page.locator('body').innerText()).slice(0, 1500));
  }
  throw error;
} finally { await browser.close(); }
