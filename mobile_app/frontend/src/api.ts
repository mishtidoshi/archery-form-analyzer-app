import axios from 'axios'
import type { Session, SessionCard, TrendData, Job, Archer, AuthResult } from './types'

const api = axios.create({ baseURL: '/api' })

// Attach the current archer's bearer token, if any, to every request. Set here rather
// than in auth.tsx so every api.ts call (including ones fired outside a component) picks
// it up automatically — auth.tsx only owns the localStorage read/write around login.
api.interceptors.request.use(config => {
  const token = localStorage.getItem('archer_token')
  if (token) {
    config.headers.Authorization = `Bearer ${token}`
  }
  return config
})

// A 401 on any archer-scoped call means the stored token is missing/expired — except on
// the auth endpoints themselves, where 401 just means "wrong password" and the caller
// (ProfilesPage) needs to show that inline rather than bounce the whole app. Everywhere
// else, drop the stale credentials and hard-reload into the picker so auth.tsx re-reads
// a clean localStorage state.
api.interceptors.response.use(
  r => r,
  error => {
    const status = error?.response?.status
    const url: string = error?.config?.url ?? ''
    const isAuthRoute = url.includes('/auth/login') || url.includes('/set-password')
    if (status === 401 && !isAuthRoute) {
      localStorage.removeItem('archer_auth')
      localStorage.removeItem('archer_token')
      if (window.location.pathname !== '/profiles') {
        window.location.href = '/profiles'
      }
    }
    return Promise.reject(error)
  },
)

// ── accounts ──────────────────────────────────────────────────────────────────

export const getArchers = () =>
  api.get<Archer[]>('/archers').then(r => r.data)

export const createArcher = (archer_id: string, name: string, password: string) =>
  api.post<AuthResult>('/archers', { archer_id, name, password }).then(r => r.data)

export const setArcherPassword = (archerId: string, password: string) =>
  api.post<AuthResult>(`/archers/${archerId}/set-password`, { password }).then(r => r.data)

export const login = (archer_id: string, password: string) =>
  api.post<AuthResult>('/auth/login', { archer_id, password }).then(r => r.data)

export const logout = () =>
  api.post('/auth/logout').then(r => r.data).catch(() => {})

export const updateArcherName = (name: string) =>
  api.put<{ archer_id: string; name: string }>('/archers/me', { name }).then(r => r.data)

export const changeArcherPassword = (current_password: string, new_password: string) =>
  api.post<AuthResult>('/archers/me/change-password', { current_password, new_password }).then(r => r.data)

export const deleteArcherAccount = (password: string) =>
  api.post<{ status: string; sessions_removed: number }>('/archers/me/delete', { password }).then(r => r.data)

export const getSessions = () =>
  api.get<SessionCard[]>('/sessions').then(r => r.data)

export const getSession = (id: string) =>
  api.get<Session>(`/sessions/${id}`).then(r => r.data)

export const getTrends = (view?: string) =>
  api.get<TrendData>('/trends', { params: view ? { view } : {} }).then(r => r.data)

export const getSessionFeedback = (id: string) =>
  api.post<{ feedback: string; findings_count: number }>(
    `/sessions/${id}/feedback`
  ).then(r => r.data)

export const getCoachNotes = () =>
  api.get<{ coaches: string[] }>('/coach-notes').then(r => r.data)

export const addCoachNoteEntry = (entry: {
  coach: string
  date: string
  cue: string
  context?: string
  category?: string
}) => api.post('/coach-notes/entry', entry).then(r => r.data)

export const uploadVideo = (formData: FormData) =>
  api.post<{ job_id: string; status: string }>('/upload', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  }).then(r => r.data)

export const getJob = (jobId: string) =>
  api.get<Job>(`/jobs/${jobId}`).then(r => r.data)
