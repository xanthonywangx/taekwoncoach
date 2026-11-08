// Everything the app needs from the Python API (uvicorn api:app --port 8000).
// Change the address in frontend/.env: VITE_API_URL=http://localhost:8000

export const API: string = (import.meta.env.VITE_API_URL as string | undefined) ?? 'http://localhost:8000';

export type CheckName = 'chamber_tightness' | 'chamber_height' | 'kick_straightness' | 'kick_angle';

export const CHECK_LABELS: Record<CheckName, string> = {
  chamber_tightness: 'Chamber tightness',
  chamber_height: 'Chamber height',
  kick_straightness: 'Kick straightness',
  kick_angle: 'Kick angle',
};

/** One kick as stored in SpacetimeDB (GET /kicks, GET /users/{name}/kicks). */
export interface Kick {
  result_id: string;
  username: string;
  recorded_time: string; // ISO
  status: 'completed' | 'processing' | 'uploaded' | 'failed' | 'unknown';
  fail_reason: string;
  passed: number | null; // 0-4, null if not graded
  tier: 'strong' | 'even' | 'building' | '' | null;
  feedback: string | null;
  feedback_source: string | null;
  checks: Record<CheckName, { pass: number; [measure: string]: number }> | null;
  images: { chamber?: string; extension?: string };
  clip_url: string | null;
}

/** What POST /grade returns. We only need a few fields; the feed has the rest. */
export interface GradeResult {
  id: string;
  status: 'graded' | 'retake';
  retake_reason?: string;
  db_saved?: boolean;
  notes?: string[];
}

export class ApiError extends Error {
  status: number;
  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
}

async function request<T>(path: string, init?: RequestInit, timeoutMs = 15000): Promise<T> {
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), timeoutMs);
  let res: Response;
  try {
    res = await fetch(`${API}${path}`, { ...init, signal: ctrl.signal });
  } catch (e) {
    if (ctrl.signal.aborted) throw new ApiError(0, 'The server took too long to answer. Try again.');
    throw new ApiError(0, `Can't reach the server at ${API}. Is uvicorn running?`);
  } finally {
    clearTimeout(timer);
  }
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    const detail = typeof body.detail === 'string' ? body.detail : `Request failed (${res.status})`;
    throw new ApiError(res.status, detail);
  }
  return res.json() as Promise<T>;
}

const json = (body: unknown): RequestInit => ({
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify(body),
});

/** Full URL for a path the API returns, e.g. /results/<id>/clip */
export const mediaUrl = (path?: string | null) => (path ? `${API}${path}` : undefined);

export const getRecentKicks = (limit = 50) => request<Kick[]>(`/kicks?limit=${limit}`);

export const getUserKicks = (username: string, limit = 50) =>
  request<Kick[]>(`/users/${encodeURIComponent(username)}/kicks?limit=${limit}`);

export const searchUsers = (query: string, limit = 8) =>
  request<{ username: string }[]>(`/users?search=${encodeURIComponent(query)}&limit=${limit}`);

/** Resolves with the user if the name exists, rejects with ApiError 404 if not. */
export const checkUsername = (username: string) =>
  request<{ username: string; created_at: string }>('/login', json({ username }));

export const createPlayer = (username: string) => request<{ username: string }>('/users', json({ username }));

export const deleteKick = (resultId: string, username: string) =>
  request<{ ok: boolean }>(
    `/results/${encodeURIComponent(resultId)}?username=${encodeURIComponent(username)}`,
    { method: 'DELETE' },
  );

/** Uploads the clip and waits for the grade (Nemotron can take 10-30 s, so allow 2 minutes). */
export function gradeKick(clip: Blob, ext: string, username: string): Promise<GradeResult> {
  const form = new FormData();
  form.append('video', clip, `kick.${ext}`);
  form.append('username', username);
  return request<GradeResult>('/grade', { method: 'POST', body: form }, 120_000);
}

export const USERNAME_RE = /^[a-z0-9_]{3,20}$/;
export const normalizeName = (raw: string) => raw.trim().toLowerCase();
