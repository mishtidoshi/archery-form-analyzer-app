import { useState, useEffect } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import { getSession, getSessionFeedback } from '../api'
import type { Session } from '../types'
import MetricBar from '../components/MetricBar'
import ReactMarkdown from 'react-markdown'

export default function SessionDetailPage() {
  const { id } = useParams<{ id: string }>()
  const navigate = useNavigate()

  const [session, setSession] = useState<Session | null>(null)
  const [loading, setLoading] = useState(true)
  const [feedback, setFeedback] = useState<string | null>(null)
  const [feedbackLoading, setFeedbackLoading] = useState(false)
  const [feedbackError, setFeedbackError] = useState<string | null>(null)

  useEffect(() => {
    if (!id) return
    getSession(id).then(s => {
      setSession(s)
      setLoading(false)
    }).catch(() => setLoading(false))
  }, [id])

  const handleGetFeedback = async () => {
    if (!id) return
    setFeedbackLoading(true)
    setFeedbackError(null)
    try {
      const { feedback: text } = await getSessionFeedback(id)
      setFeedback(text)
    } catch (err: unknown) {
      const msg = (err as { response?: { data?: { detail?: string } } })
        ?.response?.data?.detail
      setFeedbackError(msg ?? 'Could not generate feedback.')
    } finally {
      setFeedbackLoading(false)
    }
  }

  if (loading) {
    return (
      <div className="page items-center justify-center">
        <div className="text-gray-400">Loading…</div>
      </div>
    )
  }

  if (!session) {
    return (
      <div className="page items-center justify-center">
        <div className="text-gray-500">Session not found</div>
        <button className="btn-secondary mt-4" onClick={() => navigate('/sessions')}>
          Back
        </button>
      </div>
    )
  }

  const good = session.enriched_metrics.filter(m => m.status === 'good').length
  const flagged = session.enriched_metrics.filter(m => m.status === 'low' || m.status === 'high').length

  return (
    <div className="page">
      {/* Header */}
      <div className="bg-blue-600 px-4 pt-4 pb-5">
        <div className="flex items-center gap-2">
          <button
            onClick={() => navigate('/sessions')}
            className="text-blue-200 -ml-1 p-1 hover:text-white transition-colors"
          >
            <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
            </svg>
          </button>
          <div>
            <h1 className="text-base font-bold text-white leading-tight">
              {formatDate(session.date)} · {capitalize(session.view)} View
            </h1>
            <p className="text-blue-200 text-xs mt-0.5">
              {session.metrics?.arrows_with_metrics ?? session.metrics?.arrows_shot ?? '?'} arrows
              {session.distance_m ? ` · ${session.distance_m}m` : ''}
              {session.location ? ` · ${session.location}` : ''}
            </p>
          </div>
        </div>
      </div>

      <div className="p-4 space-y-4">
        {/* Summary strip */}
        {(good > 0 || flagged > 0) && (
          <div className="flex gap-3">
            {good > 0 && (
              <div className="flex-1 card p-3 text-center border border-green-200 bg-green-50">
                <div className="text-xl font-bold text-green-700">{good}</div>
                <div className="text-xs text-green-600">In range</div>
              </div>
            )}
            {flagged > 0 && (
              <div className="flex-1 card p-3 text-center border border-red-200 bg-red-50">
                <div className="text-xl font-bold text-red-600">{flagged}</div>
                <div className="text-xs text-red-500">Flagged</div>
              </div>
            )}
            <div className="flex-1 card p-3 text-center">
              <div className="text-xl font-bold text-gray-800">
                {session.metrics?.hold_time_s_avg != null
                  ? `${session.metrics.hold_time_s_avg.toFixed(2)}s`
                  : '—'}
              </div>
              <div className="text-xs text-gray-400">Avg hold</div>
            </div>
          </div>
        )}

        {/* Metrics */}
        {session.enriched_metrics.length > 0 ? (
          <div className="space-y-2">
            <h2 className="text-xs font-semibold text-gray-500 uppercase tracking-wide">
              Metrics
            </h2>
            {session.enriched_metrics.map(m => (
              <MetricBar key={m.field} metric={m} />
            ))}
          </div>
        ) : (
          <div className="card p-6 text-center text-sm text-gray-500">
            <div className="text-2xl mb-2">📊</div>
            No computed metrics for this session.
            <div className="text-xs text-gray-400 mt-1">
              Use CLI tools to run YOLO analysis.
            </div>
          </div>
        )}

        {/* AI coaching feedback */}
        <div className="space-y-2">
          <h2 className="text-xs font-semibold text-gray-500 uppercase tracking-wide">
            Coach Feedback
          </h2>

          {!feedback && !feedbackLoading && (
            <button
              className="btn-primary w-full"
              onClick={handleGetFeedback}
              disabled={session.enriched_metrics.length === 0}
            >
              Generate Coaching Feedback
            </button>
          )}

          {feedbackLoading && (
            <div className="card p-4 flex items-center gap-3 border border-blue-200 bg-blue-50">
              <div className="w-5 h-5 rounded-full border-2 border-blue-500 border-t-transparent animate-spin" />
              <span className="text-sm text-blue-700">Generating feedback…</span>
            </div>
          )}

          {feedbackError && (
            <div className="card p-4 border border-red-200 bg-red-50 text-sm text-red-700">
              {feedbackError}
            </div>
          )}

          {feedback && (
            <div className="card p-4 space-y-3 border border-blue-100 bg-blue-50">
              <div className="flex items-center justify-between">
                <span className="text-xs font-semibold text-blue-700 uppercase tracking-wide">
                  Coach Analysis
                </span>
                <button
                  onClick={handleGetFeedback}
                  className="text-xs text-blue-500 underline"
                >
                  Regenerate
                </button>
              </div>
              <div className="prose prose-sm max-w-none text-gray-800 text-sm leading-relaxed">
                <ReactMarkdown>{feedback}</ReactMarkdown>
              </div>
            </div>
          )}
        </div>

        {/* Detection notes */}
        {session.detection_notes && (
          <div className="card p-4 border border-gray-100">
            <div className="text-xs font-semibold text-gray-400 mb-1">Detection notes</div>
            <div className="text-xs text-gray-500">{session.detection_notes}</div>
          </div>
        )}
      </div>
    </div>
  )
}

function formatDate(iso: string): string {
  try {
    return new Date(iso + 'T12:00:00').toLocaleDateString('en-US', {
      month: 'short', day: 'numeric', year: 'numeric',
    })
  } catch {
    return iso
  }
}

function capitalize(s: string): string {
  return s.charAt(0).toUpperCase() + s.slice(1)
}
