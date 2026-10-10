// ─────────────────────────────────────────────────────────────
// 학습실 백엔드 연결 (backend/learning_routes.py)
// USE_MOCK = true  → 백엔드/키 없이 화면 확인용 (예시 강의 + 소리 없는 재생바)
// USE_MOCK = false → 실제 백엔드 연결 (Gemini 키 필요)
// ─────────────────────────────────────────────────────────────
import catalogData from '../data/catalog.json';

const USE_MOCK = false;
const BASE_URL = 'http://localhost:8000'; // 백엔드 주소

// 학습실에 보여줄 자격증
const VISIBLE_CERT = '정보처리기사';

const wait = (ms) => new Promise((r) => setTimeout(r, ms));

async function request(path, { method = 'GET', body } = {}) {
  const res = await fetch(BASE_URL + path, {
    method,
    headers: { 'Content-Type': 'application/json' },
    body: body ? JSON.stringify(body) : undefined,
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.detail?.message || data.message || '요청을 처리하지 못했습니다.');
  return data;
}

// 백엔드가 준 음성 주소를 실제 재생 주소로 바꿈 (가짜 모드 주소는 그대로)
export const audioUrl = (path) => (/^(blob:|https?:)/.test(path) ? path : BASE_URL + path);

export async function getCatalog() {
  if (USE_MOCK) {
    await wait(200);
    return catalogData;
  }
  const data = await request('/api/learning/catalog');
  const config = Object.fromEntries(
    Object.entries(data.config).filter(([, cert]) => cert.label === VISIBLE_CERT)
  );
  return { config, lessons: data.lessons.filter((l) => l.cert in config) };
}

const progressKey = (learner) => `mock_progress:${learner}`;

export async function getProgress(learner) {
  if (!USE_MOCK) return request(`/api/learning/progress?learner=${encodeURIComponent(learner)}`);
  try {
    return JSON.parse(localStorage.getItem(progressKey(learner))) || {};
  } catch {
    return {};
  }
}

export async function saveProgress(learner, lessonId, complete = false) {
  if (!USE_MOCK) {
    return request('/api/learning/progress', {
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
  if (!USE_MOCK) {
    const data = await request('/api/learning/teach', { method: 'POST', body: { lesson_id: lessonId, prompt } });
    return { ...data, lessonId };
  }

  // 튜터 질문: 임시 답변
  if (prompt.includes('[현재 선택한 설명 방식]')) {
    await wait(600);
    const question = prompt.split('\n')[0];
    return {
      text: `"${question}"에 대한 답변입니다.\n\n지금은 가짜(mock) 답변이에요. 백엔드가 연결되면 이 자리에 실제 튜터 답변이 표시됩니다.`,
      visual: null,
      research: null,
      lessonId,
    };
  }

  // 강의 탭: 화면 확인용 예시 강의
  await wait(400);
  const title = catalogData.lessons.find((l) => l.id === lessonId)?.title ?? '이 단원';
  return {
    text: [
      `### ${title}`,
      '이 단원의 핵심 개념을 정리합니다. 지금은 화면 확인용 예시 강의예요.',
      '- 첫 번째 핵심 특징\n- 두 번째 핵심 특징\n- 세 번째 핵심 특징',
      '> **이해를 위한 예시** 백엔드가 연결되면 이 자리에 실제 AI 강의가 표시됩니다.',
      '### 핵심 정리',
      '- 예시 정리 문장입니다.\n- 음성을 재생하면 이 문단까지 강조가 내려옵니다.',
    ].join('\n\n'),
    visual: null,
    research: null,
    lessonId,
  };
}

// ── 가짜 모드용: 소리 없는 WAV 만들기 ──
function silentWavUrl(seconds) {
  const rate = 8000;
  const samples = Math.ceil(rate * seconds);
  const buffer = new ArrayBuffer(44 + samples * 2);
  const view = new DataView(buffer);
  const write = (offset, s) => {
    for (let i = 0; i < s.length; i += 1) view.setUint8(offset + i, s.charCodeAt(i));
  };
  write(0, 'RIFF');
  view.setUint32(4, 36 + samples * 2, true);
  write(8, 'WAVE');
  write(12, 'fmt ');
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, rate, true);
  view.setUint32(28, rate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  write(36, 'data');
  view.setUint32(40, samples * 2, true);
  return URL.createObjectURL(new Blob([buffer], { type: 'audio/wav' }));
}

// ── 가짜 모드용: 백엔드처럼 문단을 나누고 제목은 다음 문단과 묶기 ──
function mockSegments(text) {
  const blocks = text.split(/\n\s*\n/).map((b) => b.trim()).filter(Boolean);
  const grouped = [];
  for (let i = 0; i < blocks.length; i += 1) {
    if (/^#{1,6}\s/.test(blocks[i]) && i + 1 < blocks.length) {
      grouped.push(`${blocks[i]}\n\n${blocks[i + 1]}`);
      i += 1;
    } else {
      grouped.push(blocks[i]);
    }
  }
  return grouped;
}

// AI 음성 강의 만들기 → { audio_url, timeline: [{ index, start, end, markdown }] }
export async function createAudioLecture(lessonId, variant, text) {
  if (USE_MOCK) {
    await wait(800);
    const PER_SEGMENT = 4; // 문단마다 4초
    const segments = mockSegments(text);
    return {
      audio_url: silentWavUrl(segments.length * PER_SEGMENT),
      timeline: segments.map((markdown, index) => ({
        index,
        start: index * PER_SEGMENT,
        end: (index + 1) * PER_SEGMENT,
        markdown,
      })),
    };
  }
  return request('/api/learning/tts', {
    method: 'POST',
    body: { lesson_id: lessonId, variant, text },
  });
}