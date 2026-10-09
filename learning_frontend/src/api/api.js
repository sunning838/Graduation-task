import catalogData from '../data/catalog.json';

const USE_MOCK = true;
const BASE_URL = 'http://localhost:8000'; // 백엔드 주소 

const wait = (ms) => new Promise((r) => setTimeout(r, ms));

async function request(path, { method = 'GET', body } = {}) {
  const res = await fetch(BASE_URL + path, {
    method,
    headers: { 'Content-Type': 'application/json' },
    body: body ? JSON.stringify(body) : undefined,
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.message || '요청을 처리하지 못했습니다.');
  return data;
}

export async function getCatalog() {
  if (!USE_MOCK) return request('/learning/catalog');
  await wait(200);
  return catalogData;
}

const progressKey = (learner) => `mock_progress:${learner}`;

export async function getProgress(learner) {
  if (!USE_MOCK) return request(`/learning/progress?learner=${encodeURIComponent(learner)}`);
  try {
    return JSON.parse(localStorage.getItem(progressKey(learner))) || {};
  } catch {
    return {};
  }
}

export async function saveProgress(learner, lessonId, complete = false) {
  if (!USE_MOCK) {
    return request('/learning/progress', {
      method: 'POST',
      body: { learner, lesson_id: lessonId, complete },
    });
  }
  const all = await getProgress(learner);
  const wasComplete = all[lessonId]?.status === 'complete';
  all[lessonId] = { status: complete || wasComplete ? 'complete' : 'started', updatedAt: Date.now() };
  localStorage.setItem(progressKey(learner), JSON.stringify(all));
  return all;
}

export async function teach(lessonId, prompt) {
  if (!USE_MOCK) return request('/learning/teach', { method: 'POST', body: { lesson_id: lessonId, prompt } });

  // 튜터 질문: 임시 답변
  if (prompt.includes('[현재 선택한 설명 방식]')) {
    await wait(600);
    const question = prompt.split('\n')[0];
    return {
      text: `"${question}"에 대한 답변입니다.\n\n지금은 가짜(mock) 답변이에요. 백엔드가 연결되면 이 자리에 실제 튜터 답변이 표시됩니다.`,
      visual: null,
      research: null,
    };
  }


  return { text: '', visual: null, research: null };
}