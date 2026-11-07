// Text learning uses the shared API client; the team's audio client is unchanged.
import { apiRequest, apiUrl } from '../../api.js';

export const variants = { '기본 설명': 'basic', '쉬운 설명': 'simple', '예시로 이해하기': 'example' };
export async function getCatalog(signal) {
  const data = await apiRequest('/api/learning/catalog', { signal });
  return { config: { EIP: data.config.EIP }, lessons: data.lessons.filter(l => l.cert === 'EIP') };
}
const progressResult = data => ({ states: data.states ?? data, resetRevision: data.reset_revision ?? 0 });
export const getProgress = (learner, signal) => apiRequest(`/api/learning/progress?learner=${encodeURIComponent(learner)}&cert=EIP`, { signal }).then(progressResult);
export const saveProgress = (learner, lessonId, complete, resetRevision, signal) => apiRequest('/api/learning/progress', {
  signal, body: { learner, lesson_id: lessonId, complete, reset_revision: resetRevision },
}).then(progressResult);
export async function teach(lessonId, variant, signal, question = null) {
  const body = { lesson_id: lessonId, variant: variants[variant] };
  if (question) Object.assign(body, {
    question: question.question,
    explanation: question.explanation.slice(0, 12000),
    history: question.history.slice(-6).map(m => ({ role: m.role, text: m.text.slice(0, 2000) })),
  });
  const data = await apiRequest('/api/learning/teach', { signal, body });
  if (typeof data.text !== 'string' || !data.text.trim()) throw new Error('설명을 준비하지 못했습니다.');
  return { ...data, lessonId };
}

export const lookupPresentation = (lessonId, variant, text, signal) => apiRequest('/api/learning/presentations/lookup', {
  signal, body: { lesson_id: lessonId, variant: variants[variant], text },
});
export const startPresentation = (lessonId, variant, text, signal) => apiRequest('/api/learning/presentations', {
  signal, body: { lesson_id: lessonId, variant: variants[variant], text },
});
export const getPresentation = (id, signal) => apiRequest(`/api/learning/presentations/${id}`, { signal });
export const retryPresentation = (id, signal) => apiRequest(`/api/learning/presentations/${id}/retry`, { signal, body: {} });
export const createLectureAudio = (lessonId, variant, text, signal) => apiRequest('/api/learning/tts', {
  signal, body: { lesson_id: lessonId, variant: variants[variant], text },
});
export const lectureMediaUrl = path => path?.startsWith('/api/') ? apiUrl(path) : '';
