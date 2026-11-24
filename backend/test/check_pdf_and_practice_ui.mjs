// Offline browser checks against the Vite dev/preview server on port 5188.
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';
import path from 'node:path';

const require = createRequire(process.env.PRESENTATION_TEST_NODE_MODULES
  ? path.join(process.env.PRESENTATION_TEST_NODE_MODULES, '_entry.cjs')
  : new URL('../../react_frontend/package.json', import.meta.url));
const { chromium } = require('playwright');

function pdfFixture(label) {
  const stream = `BT /F1 16 Tf 30 240 Td (${label}) Tj ET\n`;
  const objects = [
    '<< /Type /Catalog /Pages 2 0 R >>',
    '<< /Type /Pages /Kids [3 0 R] /Count 1 >>',
    '<< /Type /Page /Parent 2 0 R /MediaBox [0 0 300 300] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>',
    '<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>',
    `<< /Length ${stream.length} >>\nstream\n${stream}endstream`,
  ];
  let contents = '%PDF-1.4\n';
  const offsets = objects.map((object, index) => {
    const offset = contents.length;
    contents += `${index + 1} 0 obj\n${object}\nendobj\n`;
    return offset;
  });
  const xref = contents.length;
  return Buffer.from(contents + `xref\n0 6\n0000000000 65535 f \n${offsets.map(offset =>
    `${String(offset).padStart(10, '0')} 00000 n \n`).join('')}trailer\n<< /Size 6 /Root 1 0 R >>\nstartxref\n${xref}\n%%EOF`);
}

const browser = await chromium.launch({ headless: true,
  executablePath: process.env.CHROME_PATH || 'C:/Program Files/Google/Chrome/Application/chrome.exe' });
try {
  const page = await browser.newPage({ viewport: { width: 1440, height: 1000 } });
  page.setDefaultTimeout(10000);
  const errors = [], questions = [], quizRequests = [], submissions = [];
  let summaries = 0;
  let releaseSubmission;
  page.on('pageerror', error => errors.push(error.message));
  await page.route(url => url.pathname.startsWith('/api/'), async route => {
    const request = route.request();
    const pathname = new URL(request.url()).pathname;
    const json = data => route.fulfill({ json: data });
    if (pathname === '/api/certifications') return json({ certifications: [{ id: 'EIP', label: '정보처리기사' }] });
    if (pathname === '/api/learning/catalog') return json({
      config: { EIP: { label: '정보처리기사', topics: { database: '데이터베이스' } } },
      lessons: [{ id: 'lesson-1', version: '1', cert: 'EIP', subject: 'database', title: '테스트 단원' }],
    });
    if (pathname === '/api/learning/progress') return json({ states: {}, reset_revision: 0 });
    if (pathname === '/api/learning/teach') return json({ text: '## 테스트 단원\n\n개념 설명입니다.', visual: null });
    if (pathname === '/api/learning/presentations/lookup') return json({ status: 'idle' });
    if (pathname === '/api/stats') return json({});
    if (pathname === '/api/pdf-summary') {
      const filename = request.postDataBuffer().toString().match(/filename="(test-[1-4]\.pdf)"/)[1];
      summaries += 1;
      return json({ summary: `요약 ${filename}`, source_pages: [{ page: 1, text: `원문 ${filename}` }] });
    }
    if (pathname === '/api/pdf-summary/question') {
      const data = request.postDataJSON();
      questions.push(data);
      const answer = `문서 답변 ${data.source_pages[0].text}`;
      if (data.source_pages[0].text.endsWith('test-4.pdf')) {
        return json({
          answer: `${answer}\n\n**보충 설명**\n\n공식 자료에서 확인한 정의입니다.`,
          research: { status: 'supplemented', suggestions_html: '<p>검색 참고</p>',
            sources: [{ title: '공식 출처', url: 'https://developers.google.com/machine-learning' }] },
        });
      }
      return json({ answer });
    }
    if (pathname === '/api/chat') throw new Error('PDF questions must use the PDF endpoint');
    if (pathname === '/api/quiz') {
      const data = request.postDataJSON();
      quizRequests.push(data);
      return json({ attempt_id: `q-${quizRequests.length}`, question: data.lesson_id ? '단원 문제' : `일반 문제 ${quizRequests.length}`,
        topic_label: '데이터베이스', options: ['첫째', '둘째', '셋째', '넷째'] });
    }
    if (pathname === '/api/quiz/submit') {
      submissions.push(request.postDataJSON());
      await new Promise(resolve => { releaseSubmission = resolve; });
      return json({ is_correct: false, correct_answer: 1, explanation: '저장된 해설입니다.' });
    }
    return route.fulfill({ status: 404, json: { detail: 'Unexpected test route' } });
  });

  const button = name => page.getByRole('button', { name, exact: true });
  const menu = text => page.locator('.menu-item').filter({ hasText: text });
  const pdfResult = page.getByRole('region', { name: 'PDF 요약 결과' });
  async function summarize(n) {
    const filename = `test-${n}.pdf`;
    await page.locator('input[type=file]').setInputFiles({ name: filename, mimeType: 'application/pdf', buffer: pdfFixture(filename) });
    await button('추출하기').click();
    await pdfResult.getByText(`요약 ${filename}`, { exact: true }).waitFor();
    await pdfResult.getByLabel('원문 1쪽 / 1쪽', { exact: true }).waitFor();
  }
  async function ask(filename) {
    await button('튜터에게 질문').click();
    await page.getByPlaceholder('이 부분이 궁금해요…').fill('이 문서 내용은 무엇인가요?');
    await button('질문하기').click();
    await page.getByText(`문서 답변 원문 ${filename}`, { exact: true }).waitFor();
    const source = page.getByRole('link', { name: '공식 출처', exact: true });
    if (filename === 'test-4.pdf') {
      await page.getByText('보충 설명', { exact: true }).waitFor();
      assert.equal(await source.getAttribute('href'), 'https://developers.google.com/machine-learning');
    } else {
      assert.equal(await source.count(), 0);
    }
    await button('닫기').click();
  }

  await page.goto(process.env.PRESENTATION_TEST_URL || 'http://127.0.0.1:5188');
  await button('학습 시작하기').click();
  await page.locator('.sidebar-toggle').click();
  await button('PDF 요약').click();
  await summarize(1);
  await menu('개념 학습').click();
  await button('PDF 요약').click();
  await pdfResult.getByText('요약 test-1.pdf', { exact: true }).waitFor();
  assert.equal(summaries, 1);
  for (const n of [2, 3, 4]) {
    await button('이전으로').click();
    await summarize(n);
  }

  await ask('test-4.pdf');
  await button('이전으로').click();
  const recent = page.getByRole('region', { name: '최근 요약' });
  assert.deepEqual(await recent.locator('strong').allTextContents(), ['test-4.pdf', 'test-3.pdf', 'test-2.pdf']);
  await recent.getByRole('button', { name: /test-3.pdf/ }).click();
  await ask('test-3.pdf');
  assert.deepEqual(questions.map(data => data.source_pages[0].text), ['원문 test-4.pdf', '원문 test-3.pdf']);
  assert.deepEqual(questions.map(data => data.summary), ['요약 test-4.pdf', '요약 test-3.pdf']);
  assert.ok(questions.every(data => data.history.length === 0));
  assert.equal(summaries, 4);

  await page.reload();
  await button('학습 시작하기').click();
  await button('PDF 요약').click();
  await recent.getByRole('button', { name: /test-2.pdf/ }).click();
  await pdfResult.getByText('요약 test-2.pdf', { exact: true }).waitFor();
  await pdfResult.getByLabel('원문 1쪽 / 1쪽', { exact: true }).waitFor();
  assert.equal(summaries, 4);

  await page.locator('.sidebar-toggle').click();
  await page.getByTitle('문제풀이 화면으로 가기').click();
  await page.getByText('Q. 일반 문제 1', { exact: true }).waitFor();
  await page.locator('.quiz-option').nth(1).click();
  await menu('AI 튜터').click();
  await menu('문제 풀이').click();
  assert.match(await page.locator('.quiz-option.selected').innerText(), /둘째/);
  assert.equal(quizRequests.length, 1);

  await button('정답 제출').click();
  await menu('AI 튜터').click();
  assert.equal(submissions[0].attempt_id, 'q-1');
  releaseSubmission();
  await menu('문제 풀이').click();
  await page.getByText('저장된 해설입니다.', { exact: true }).waitFor();
  await page.getByTitle('개념 학습으로 돌아가기').click();
  await button('이어서 학습').click();
  await button('문제 풀기').click();
  await page.getByText('Q. 단원 문제', { exact: true }).waitFor();
  assert.equal(quizRequests[1].lesson_id, 'lesson-1');
  await menu('문제 풀이').click();
  await page.getByText('Q. 일반 문제 1', { exact: true }).waitFor();
  await page.getByText('저장된 해설입니다.', { exact: true }).waitFor();
  assert.equal(quizRequests.length, 2);
  await button('새 문제').click();
  await page.getByText('Q. 일반 문제 3', { exact: true }).waitFor();
  assert.equal(quizRequests.length, 3);
  assert.deepEqual(errors, []);
  console.log('Passed: PDF result restoration, persistent latest three, document-specific chat, and scoped practice restoration.');
} finally {
  await browser.close();
}
