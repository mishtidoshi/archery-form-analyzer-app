import { useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import { getSessions } from '../api'
import type { SessionCard, ViewType } from '../types'

const VIEW_ICON: Record<ViewType, string> = {
  face:   '👤',
  back:   '🏹',
  target: '🎯',
}

const VIEW_COLOR: Record<ViewType, string> = {
  face:   'bg-purple-100 text-purple-700',
  back:   'bg-indigo-100 text-indigo-700',
  target: 'bg-orange-100 text-orange-700',
}

type Filter = 'all' | ViewType

function groupByDate(sessions: SessionCard[]): Record<string, SessionCard[]> {
  return sessions.reduce<Record<string, SessionCard[]>>((acc, s) => {
    const key = s.date || 'Unknown date'
    ;(acc[key] = acc[key] || []).push(s)
    return acc
  }, {})
}

export default function SessionsPage() {
  const navigate = useNavigate()
  const [sessions, setSessions] = useState<SessionCard[]>([])
  const [filter, setFilter] = useState<Filter>('all')
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    getSessions().then(s => {
      setSessions(s)
      setLoading(false)
    }).catch(() => setLoading(false))
  }, [])

  const filtered = filter === 'all'
    ? sessions
    : sessions.filter(s => s.view === filter)

  const grouped = groupByDate(filtered)
  const dates = Object.keys(grouped).sort().reverse()

  return (
    <div className="page">
      <div className="bg-blue-600 px-5 pt-5 pb-4 space-y-3">
        <h1 className="text-xl font-bold text-white">Sessions</h1>

        {/* Filter chips */}
        <div className="flex gap-2 overflow-x-auto pb-1 -mx-1 px-1">
          {(['all', 'face', 'back', 'target'] as Filter[]).map(f => (
            <button
              key={f}
              onClick={() => setFilter(f)}
              className={`flex-shrink-0 px-3 py-1 rounded-full text-xs font-medium capitalize transition-colors ${
                filter === f
                  ? 'bg-white text-blue-700'
                  : 'bg-blue-500 text-blue-100'
              }`}
            >
              {f === 'all' ? 'All views' : `${VIEW_ICON[f as ViewType]} ${f}`}
            </button>
          ))}
        </div>
      </div>

      <div className="p-4 space-y-6">
        {loading && (
          <div className="text-center py-12 text-gray-400 text-sm">Loading…</div>
        )}

        {!loading && filtered.length === 0 && (
          <div className="text-center py-12">
            <div className="text-4xl mb-3">🎯</div>
            <div className="text-gray-500 text-sm">No sessions yet</div>
            <div className="text-gray-400 text-xs mt-1">Upload a video to get started</div>
          </div>
        )}

        {dates.map(date => (
          <div key={date}>
            <div className="text-xs font-semibold text-gray-400 uppercase tracking-wide mb-2">
              {formatDate(date)}
            </div>
            <div className="space-y-2">
              {grouped[date].map(s => (
                <button
                  key={s.id}
                  className="card w-full text-left p-4 hover:shadow-md transition-shadow active:scale-[0.98]"
                  onClick={() => navigate(`/sessions/${s.id}`)}
                >
                  <div className="flex items-start justify-between gap-2">
                    <div className="flex items-center gap-2">
                      <span className={`text-xs px-2 py-0.5 rounded-full font-medium capitalize ${VIEW_COLOR[s.view]}`}>
                        {VIEW_ICON[s.view]} {s.view}
                      </span>
                      {s.distance_m && (
                        <span className="text-xs text-gray-400">{s.distance_m}m</span>
                      )}
                    </div>
                    <div className="flex items-center gap-1.5">
                      {s.metrics_flagged > 0 && (
                        <span className="text-xs bg-red-100 text-red-600 px-2 py-0.5 rounded-full font-medium">
                          {s.metrics_flagged} flagged
                        </span>
                      )}
                      {s.metrics_good > 0 && (
                        <span className="text-xs bg-green-100 text-green-600 px-2 py-0.5 rounded-full font-medium">
                          {s.metrics_good} good
                        </span>
                      )}
                    </div>
                  </div>

                  <div className="mt-2 flex items-center justify-between">
                    <div className="text-sm text-gray-600">
                      {s.arrows > 0 ? `${s.arrows} arrows` : 'No metrics'}
                      {s.location ? ` · ${s.location}` : ''}
                    </div>
                    <svg className="w-4 h-4 text-gray-300" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                    </svg>
                  </div>

                  {s.notes && (
                    <div className="mt-1 text-xs text-gray-400 truncate">{s.notes}</div>
                  )}

                  {!s.has_metrics && (
                    <div className="mt-2 text-xs text-amber-600 bg-amber-50 rounded-lg px-2 py-1">
                      Pending YOLO analysis — use CLI tools to finalize
                    </div>
                  )}
                </button>
              ))}
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}

function formatDate(iso: string): string {
  if (!iso || iso === 'Unknown date') return iso
  try {
    return new Date(iso + 'T12:00:00').toLocaleDateString('en-US', {
      weekday: 'long', month: 'short', day: 'numeric', year: 'numeric',
    })
  } catch {
    return iso
  }
}
