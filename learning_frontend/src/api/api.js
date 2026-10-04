// ─────────────────────────────────────────────────────────────
// 백엔드 연결 지점
// 지금은 백엔드가 없어서 가짜(mock) 데이터로 동작함.
// 백엔드가 준비되면 USE_MOCK을 false로 바꾸고 BASE_URL과 경로만 맞추면 됨.
// ─────────────────────────────────────────────────────────────
import catalogData from '../data/catalog.json';
// ↑ scripts/export_catalog.py로 백엔드 단원 목록을 뽑아 만든 파일
//   단원(.md)이 바뀌면 스크립트를 다시 실행하면 됨

const USE_MOCK = true;
const BASE_URL = 'http://localhost:8000'; // 백엔드 주소 (백엔드 담당자에게 확인)

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

// ── 로그인 / 회원가입 ─────────────────────────────────────────
function readMockUsers() {
  try {
    return JSON.parse(localStorage.getItem('mock_users')) || [];
  } catch {
    return [];
  }
}

/** 로그인: 성공하면 { name, email } 반환 */
export async function login(email, password) {
  if (!USE_MOCK) return request('/auth/login', { method: 'POST', body: { email, password } });

  // mock: 아무 아이디/비밀번호나 입력하면 로그인 성공
  await wait(300);
  const saved = readMockUsers().find((u) => u.email === email);
  return { name: saved?.name ?? email.split('@')[0], email };
}

/** 회원가입: 성공하면 { name, email } 반환 */
export async function signup(name, email, password) {
  if (!USE_MOCK) return request('/auth/signup', { method: 'POST', body: { name, email, password } });

  await wait(500);
  const users = readMockUsers();
  if (users.some((u) => u.email === email)) {
    throw new Error('이미 가입된 이메일입니다.');
  }
  // 주의: mock 전용. 실제 서비스에서는 비밀번호를 프론트에 저장하지 않음
  users.push({ name, email, password });
  localStorage.setItem('mock_users', JSON.stringify(users));
  return { name, email };
}

// ── 학습 (backend/learning.py의 catalog, Progress, teach 대신) ──
// 백엔드 응답 형식 약속:
//   getCatalog  → { config: { [cert]: { label, topics: { [subject]: 라벨 } } },
//                   lessons: [{ id, cert, subject, title }] }   ※ title은 display_title 적용된 값
//   getProgress → { [lessonId]: { status: 'started' | 'complete', updatedAt: 숫자 } }
//   teach       → { text, visual, research }                    ※ text는 learner_text 적용된 값
//                 visual: { kind: 'table', title, purpose, columns, rows }
//                       | { kind: 'graph', title, purpose, dot } | null
//                 research: { status, suggestions_html, sources: [{ title, url }] } | null

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

/** 단원을 열었을 때(complete=false), 완료했을 때(complete=true) 호출. 저장 후 전체 진도 반환 */
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

/** 강의 설명 / 튜터 답변 요청 (파이썬 teach(active, all_lessons, prompt)와 같은 역할) */
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

  // 기본 설명 / 쉬운 설명 / 예시로 이해하기: 백엔드 연결 전까지 빈칸
  return { text: '', visual: null, research: null };
}