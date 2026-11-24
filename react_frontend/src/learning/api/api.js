// PDF requests and compatibility helpers for the existing audio component.
import { apiRequest } from '../../api.js';

const BASE_URL = 'http://localhost:8000';

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

// Keep existing audio URL handling.
export const audioUrl = (path) => (/^(blob:|https?:)/.test(path) ? path : BASE_URL + path);

export async function createAudioLecture(lessonId, variant, text) {
  return request('/api/learning/tts', {
    method: 'POST',
    body: { lesson_id: lessonId, variant, text },
  });
}

export function summarizePdf(file, answerLength, signal) {
  const formData = new FormData();
  formData.append('file', file);
  formData.append('answer_length', answerLength);
  return apiRequest('/api/pdf-summary', { body: formData, signal });
}

export function askPdf(record, question, history, answerLength, signal) {
  return apiRequest('/api/pdf-summary/question', {
    body: {
      source_pages: record.sourcePages,
      summary: (record.summary ?? '').slice(0, 3500),
      question,
      history,
      answer_length: answerLength,
    },
    signal,
  });
}
