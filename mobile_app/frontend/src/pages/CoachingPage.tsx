import { useState, useEffect } from 'react'
import { useNavigate, useSearchParams } from 'react-router-dom'
import { getSessions, getSessionFeedback } from '../api'
import type { SessionCard, ViewType } from '../types'
import ReactMarkdown from 'react-markdown'

const VIEW_ICON: Record<ViewType, string> = {
  face: '👤', back: '🏹', target: '🎯',
}

export default function CoachingPage() {
  const navigate = useNavigate()
  const [searchParams] = useSearchParams()
  const initId = searchParams.get('session')

  const [sessions, setSessions] = useState<SessionCard[]>([])
  const [selectedId, setSelectedId] = useState<string | null>(initId)
  const [feedback, setFeedback] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [sessionsLoading, setSessionsLoading] = useState(true)

  useEffect(() => {
    getSessions().then(s => {
      // Only sessions with computed metrics
      const scorable = s.filter(x => x.has_metrics)
      setSessions(scorable)
      setSessionsLoading(false)
    }).catch(() => setSessionsLoading(false))
  }, [])

  const handleGenerate = async () => {
    if (!selectedId) return
    setLoading(true)
    setError(null)
    setFeedback(null)
    try {
      const { feedback: text } = await getSessionFeedback(selectedId)
      setFeedback(text)
    } catch (err: unknown) {
      const msg = (err as { response?: { data?: { detail?: string } } })
        ?.response?.data?.detail
      setError(msg ?? 'Failed to generate feedback.')
    } finally {
      setLoading(false)
    }
  }

  const selectedSession = sessions.find(s => s.id === selectedId)

  return (
    <div className="flex flex-col h-full">
      <div className="bg-blue-600 px-5 pt-5 pb-5 flex-shrink-0">
        <h1 className="text-xl font-bold text-white">Coaching</h1>
        <p className="text-blue-200 text-xs mt-0.5">Post-session feedback from your measured metrics</p>
      </div>

      <div className="flex-1 overflow-y-auto flex flex-col">
      <div className="p-4 space-y-4 flex-1 flex flex-col">
        {/* Session picker */}
        <div className="card p-4 space-y-2 flex-1 flex flex-col">
          <label className="text-xs font-semibold text-gray-500 uppercase tracking-wide">
            Select Session
          </label>

          {sessionsLoading && (
            <div className="text-sm text-gray-400">Loading sessions…</div>
          )}

          {!sessionsLoading && sessions.length === 0 && (
            <div className="text-sm text-gray-500">
              No sessions with metrics yet.{' '}
              <button className="text-blue-600 underline" onClick={() => navigate('/upload')}>
                Upload a video
              </button>
            </div>
          )}

          {sessions.length > 0 && (
            <div className="space-y-2 flex-1 overflow-y-auto">
              {sessions.map(s => (
                <button
                  key={s.id}
                  onClick={() => {
                    setSelectedId(s.id)
                    setFeedback(null)
                    setError(null)
                  }}
                  className={`w-full text-left px-3 py-3 rounded-xl border-2 transition-colors ${
                    selectedId === s.id
                      ? 'border-blue-500 bg-blue-50'
                      : 'border-gray-100 bg-gray-50'
                  }`}
                >
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <span className="text-sm">{VIEW_ICON[s.view]}</span>
                      <div>
                        <div className="text-sm font-medium text-gray-800">
                          {formatDate(s.date)}
                        </div>
                        <div className="text-xs text-gray-400">
                          {s.arrows} arrows · {capitalize(s.view)} view
                          {s.distance_m ? ` · ${s.distance_m}m` : ''}
                        </div>
                      </div>
                    </div>
                    {selectedId === s.id && (
                      <svg className="w-4 h-4 text-blue-500" fill="currentColor" viewBox="0 0 20 20">
                        <path fillRule="evenodd"
                          d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z"
                          clipRule="evenodd" />
                      </svg>
                    )}
                  </div>
                  {(s.metrics_flagged > 0 || s.metrics_good > 0) && (
                    <div className="flex gap-2 mt-1.5">
                      {s.metrics_flagged > 0 && (
                        <span className="text-[10px] bg-red-100 text-red-600 px-1.5 py-0.5 rounded-full">
                          {s.metrics_flagged} flagged
                        </span>
                      )}
                      {s.metrics_good > 0 && (
                        <span className="text-[10px] bg-green-100 text-green-600 px-1.5 py-0.5 rounded-full">
                          {s.metrics_good} good
                        </span>
                      )}
                    </div>
                  )}
                </button>
              ))}
            </div>
          )}
        </div>

        {/* Generate button */}
        {selectedId && !feedback && !loading && (
          <button className="btn-primary w-full" onClick={handleGenerate}>
            Generate Coaching Feedback
          </button>
        )}

        {/* Loading */}
        {loading && (
          <div className="card p-5 flex flex-col items-center gap-3 border border-blue-200 bg-blue-50">
            <div className="w-8 h-8 rounded-full border-3 border-blue-500 border-t-transparent animate-spin" />
            <div className="text-sm font-medium text-blue-700 text-center">
              Analyzing {selectedSession ? `${selectedSession.arrows} arrows` : 'your session'}…
            </div>
            <div className="text-xs text-blue-400 text-center">
              Comparing your metrics against coach-validated thresholds.
            </div>
          </div>
        )}

        {/* Error */}
        {error && !loading && (
          <div className="card p-4 border border-red-200 bg-red-50 space-y-2">
            <div className="text-sm font-semibold text-red-700">Feedback unavailable</div>
            <div className="text-xs text-red-600">{error}</div>
            <button className="btn-secondary text-sm py-2 w-full" onClick={handleGenerate}>
              Try Again
            </button>
          </div>
        )}

        {/* Feedback */}
        {feedback && (
          <div className="space-y-3">
            <div className="card border border-blue-200 bg-gradient-to-br from-blue-50 to-indigo-50">
              <div className="px-4 py-3 border-b border-blue-100 flex items-center justify-between">
                <div>
                  <div className="text-xs font-semibold text-blue-700 uppercase tracking-wide">
                    Coach Analysis
                  </div>
                  {selectedSession && (
                    <div className="text-xs text-blue-400 mt-0.5">
                      {formatDate(selectedSession.date)} · {capitalize(selectedSession.view)} view
                    </div>
                  )}
                </div>
                <button
                  onClick={handleGenerate}
                  className="text-xs text-blue-500 bg-blue-100 hover:bg-blue-200 px-2 py-1 rounded-lg transition-colors"
                >
                  Regenerate
                </button>
              </div>
              <div className="px-4 py-4 prose prose-sm max-w-none text-gray-800 leading-relaxed">
                <ReactMarkdown>{feedback}</ReactMarkdown>
              </div>
            </div>

            <button
              className="btn-secondary w-full text-sm"
              onClick={() => navigate(`/sessions/${selectedId}`)}
            >
              View Session Metrics
            </button>
          </div>
        )}
      </div>
      </div>
    </div>
  )
}

function formatDate(iso: string): string {
  try {
    return new Date(iso + 'T12:00:00').toLocaleDateString('en-US', {
      month: 'short', day: 'numeric', year: 'numeric',
    })
  } catch { return iso }
}

function capitalize(s: string): string {
  return s.charAt(0).toUpperCase() + s.slice(1)
}
