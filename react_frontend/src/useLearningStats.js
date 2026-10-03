import { useCallback, useEffect, useState } from 'react';
import { apiRequest } from './api';

export function useLearningStats(cert, page) {
  const [revision, setRevision] = useState(0);
  const [response, setResponse] = useState(null);
  const view = page === 'profile' || page === 'weakness' ? page : null;
  const key = `${cert}:${view}:${revision}`;
  const refresh = useCallback(() => setRevision(value => value + 1), []);
  useEffect(() => {
    if (!cert || !view) return;
    const controller = new AbortController();
    apiRequest(`/api/stats?cert=${encodeURIComponent(cert)}`, { signal: controller.signal })
      .then(data => {
        if (!controller.signal.aborted) setResponse({ key, data, error: '' });
      })
      .catch(error => {
        if (!controller.signal.aborted) setResponse({ key, data: null, error: error.message });
      });
    return () => controller.abort();
  }, [cert, view, key]);
  const current = response?.key === key ? response : null;
  return { stats: current?.data ?? null, error: current?.error ?? '',
           loading: Boolean(view && cert && !current), refresh };
}
