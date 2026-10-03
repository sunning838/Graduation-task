const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL || "http://localhost:8000").replace(/\/$/, "");

export async function apiRequest(path, { body, signal, method } = {}) {
  let response;
  try {
    response = await fetch(`${API_BASE_URL}${path}`, {
      method: method || (body === undefined ? "GET" : "POST"),
      headers: body === undefined ? {} : { "Content-Type": "application/json" },
      body: body === undefined ? undefined : JSON.stringify(body),
      signal,
    });
  } catch (error) {
    if (error.name === "AbortError") throw error;
    throw new Error("서버에 연결하지 못했습니다. 연결 상태를 확인하고 다시 시도해 주세요.", { cause: error });
  }
  const data = await response.json().catch(() => null);
  if (!response.ok) {
    const detail = data?.detail;
    const error = new Error(detail?.message || (typeof detail === "string" ? detail : "요청을 처리하지 못했습니다."));
    error.code = detail?.code;
    error.retryable = detail?.retryable ?? false;
    throw error;
  }
  if (!data) throw new Error("서버 응답을 읽지 못했습니다. 다시 시도해 주세요.");
  return data;
}
