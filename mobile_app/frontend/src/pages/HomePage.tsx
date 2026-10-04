import { useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import { getSessions, getTrends } from '../api'
import { useAuth } from '../auth'
import type { SessionCard, TrendData, ViewType } from '../types'
import {
  LineChart, Line, ResponsiveContainer, Tooltip,
} from 'recharts'

// ── key metrics to surface on home (one per view, most coaching-relevant) ──
const SPOTLIGHT: Record<ViewType, { field: string; label: string; unit: string; good_range: [number | null, number | null] | null }> = {
  face:   { field: 'hold_time_s_avg',       label: 'Hold Time',       unit: 's',  good_range: [0.5, 1.5] },
  back:   { field: 'shoulder_level_avg',    label: 'Shoulder Level',  unit: '°',  good_range: [-3, 3]    },
  target: { field: 'draw_elbow_height_th_avg', label: 'Elbow Height', unit: ' th', good_range: [null, 0]  },
}

function isGood(value: number, gr: [number | null, number | null] | null): boolean {
  if (!gr) return true
  const [lo, hi] = gr
  return (lo == null || value >= lo) && (hi == null || value <= hi)
}

function daysAgo(dateStr: string): number {
  const d = new Date(dateStr + 'T12:00:00')
  const now = new Date()
  return Math.round((now.getTime() - d.getTime()) / (1000 * 60 * 60 * 24))
}

function formatShort(iso: string): string {
  try {
    return new Date(iso + 'T12:00:00').toLocaleDateString('en-US', {
      month: 'short', day: 'numeric',
    })
  } catch { return iso }
}

// ── greeting based on time of day ─────────────────────────────────────────────
function greeting(): string {
  const h = new Date().getHours()
  if (h < 12) return 'Good morning'
  if (h < 17) return 'Good afternoon'
  return 'Good evening'
}

export default function HomePage() {
  const navigate = useNavigate()
  const { archerName, logout } = useAuth()
  const [sessions, setSessions] = useState<SessionCard[]>([])
  const [trends, setTrends] = useState<TrendData>({})
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    Promise.all([getSessions(), getTrends()])
      .then(([s, t]) => {
        setSessions(s)
        setTrends(t)
      })
      .finally(() => setLoading(false))
  }, [])

  const latestSession = sessions[0] ?? null
  const practiceSessions = sessions.filter(s => s.has_metrics)
  const totalArrows = sessions.reduce((sum, s) => sum + s.arrows, 0)
  const totalSessions = sessions.length
  const daysSince = latestSession ? daysAgo(latestSession.date) : null

  // Streak: consecutive practice days (any session per day)
  const streak = calcStreak(sessions)

  // Spotlight metrics: last value for each view
  const spotlights = computeSpotlights(sessions, trends)

  return (
    <div className="page">
      {/* ── header ── */}
      <div className="bg-blue-600 px-5 pt-6 pb-8">
        <div className="flex items-start justify-between">
          <div>
            <p className="text-blue-200 text-sm">{greeting()},</p>
            <h1 className="text-2xl font-bold text-white mt-0.5">{archerName ?? 'Archer'}</h1>
          </div>
          <div className="flex items-center gap-2">
            <button
              onClick={() => navigate('/settings')}
              aria-label="Settings"
              className="text-blue-200 p-1.5 border border-white/30 rounded-full active:bg-white/10"
            >
              <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M10.343 3.94c.09-.542.56-.94 1.11-.94h1.093c.55 0 1.02.398 1.11.94l.149.894c.07.424.384.764.78.93.398.164.855.142 1.205-.108l.737-.527a1.125 1.125 0 011.45.12l.773.774c.39.389.44 1.002.12 1.45l-.527.737c-.25.35-.272.806-.107 1.204.165.397.505.71.93.78l.893.15c.543.09.94.559.94 1.109v1.094c0 .55-.397 1.02-.94 1.11l-.893.149c-.425.07-.765.383-.93.78-.165.398-.143.854.107 1.204l.527.738c.32.447.269 1.06-.12 1.45l-.774.773a1.125 1.125 0 01-1.449.12l-.738-.527c-.35-.25-.806-.272-1.203-.107-.397.165-.71.505-.781.929l-.149.894c-.09.542-.56.94-1.11.94h-1.094c-.55 0-1.019-.398-1.11-.94l-.148-.894c-.071-.424-.384-.764-.781-.93-.398-.164-.854-.142-1.204.108l-.738.527c-.447.32-1.06.269-1.45-.12l-.773-.774a1.125 1.125 0 01-.12-1.45l.527-.737c.25-.35.273-.806.108-1.204-.165-.397-.505-.71-.93-.78l-.894-.15c-.542-.09-.94-.558-.94-1.109v-1.094c0-.55.398-1.02.94-1.11l.894-.149c.424-.07.765-.383.93-.78.165-.398.143-.854-.108-1.204l-.526-.738a1.125 1.125 0 01.12-1.45l.773-.773a1.125 1.125 0 011.45-.12l.737.527c.35.25.807.272 1.204.107.397-.165.71-.505.78-.929l.15-.894z" />
                <path strokeLinecap="round" strokeLinejoin="round" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
              </svg>
            </button>
            <button
              onClick={() => {
                if (window.confirm('Switch profile?')) {
                  logout()
                  navigate('/profiles')
                }
              }}
              className="text-blue-200 text-xs font-medium border border-white/30 rounded-full px-3 py-1 active:bg-white/10"
            >
              Switch
            </button>
          </div>
        </div>

        {daysSince != null && (
          <p className="text-blue-200 text-xs mt-1">
            {daysSince === 0
              ? 'You practiced today'
              : daysSince === 1
              ? 'Last practice yesterday'
              : `Last practice ${daysSince} days ago`}
          </p>
        )}
      </div>

      <div className="px-4 -mt-4 space-y-4 pb-6">

        {/* ── upload CTA (primary action card) ─────────────── */}
        <button
          onClick={() => navigate('/upload')}
          className="w-full card p-4 flex items-center gap-4 border-2 border-blue-200 bg-white
                     active:scale-[0.98] transition-transform shadow-md"
        >
          <div className="w-12 h-12 rounded-2xl bg-blue-600 flex items-center justify-center flex-shrink-0">
            <svg className="w-6 h-6 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2.5}>
              <path strokeLinecap="round" strokeLinejoin="round"
                d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5m-13.5-9L12 3m0 0l4.5 4.5M12 3v13.5" />
            </svg>
          </div>
          <div className="text-left">
            <div className="font-semibold text-gray-900">Upload Practice Video</div>
            <div className="text-xs text-gray-400 mt-0.5">
              Add a video to analyze your form
            </div>
          </div>
          <svg className="w-4 h-4 text-gray-300 ml-auto flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
          </svg>
        </button>

        {loading && (
          <div className="text-center py-8 text-gray-400 text-sm">Loading your data…</div>
        )}

        {!loading && (
          <>
            {/* ── at-a-glance stats ─────────────── */}
            <div className="grid grid-cols-3 gap-3">
              <StatCard value={totalArrows.toLocaleString()} label="Total Arrows" color="blue" />
              <StatCard value={String(totalSessions)} label="Sessions" color="indigo" />
              <StatCard
                value={streak > 0 ? `${streak}d` : '—'}
                label="Streak"
                color={streak >= 3 ? 'green' : 'gray'}
              />
            </div>

            {/* ── last session card ─────────────── */}
            {latestSession && (
              <div>
                <SectionHeader>Last Session</SectionHeader>
                <button
                  className="card w-full text-left p-4 active:scale-[0.98] transition-transform"
                  onClick={() => navigate(`/sessions/${latestSession.id}`)}
                >
                  <div className="flex items-start justify-between gap-2">
                    <div>
                      <div className="font-semibold text-gray-900 text-sm">
                        {formatShort(latestSession.date)} · {capitalize(latestSession.view)} View
                      </div>
                      <div className="text-xs text-gray-400 mt-0.5">
                        {latestSession.arrows} arrows
                        {latestSession.distance_m ? ` · ${latestSession.distance_m}m` : ''}
                        {latestSession.location ? ` · ${latestSession.location}` : ''}
                      </div>
                    </div>
                    <div className="flex gap-1.5 flex-shrink-0">
                      {latestSession.metrics_flagged > 0 && (
                        <span className="text-xs bg-red-100 text-red-600 px-2 py-0.5 rounded-full font-medium">
                          {latestSession.metrics_flagged} flagged
                        </span>
                      )}
                      {latestSession.metrics_good > 0 && (
                        <span className="text-xs bg-green-100 text-green-600 px-2 py-0.5 rounded-full font-medium">
                          {latestSession.metrics_good} good
                        </span>
                      )}
                    </div>
                  </div>

                  {latestSession.notes && (
                    <div className="text-xs text-gray-400 mt-2 truncate">{latestSession.notes}</div>
                  )}

                  <div className="mt-3 flex gap-2">
                    <span className="text-xs text-blue-600 font-medium">View metrics →</span>
                    <span className="text-xs text-gray-400">·</span>
                    <button
                      className="text-xs text-blue-600 font-medium"
                      onClick={e => {
                        e.stopPropagation()
                        navigate(`/coaching?session=${latestSession.id}`)
                      }}
                    >
                      Get feedback →
                    </button>
                  </div>
                </button>
              </div>
            )}

            {/* ── form spotlight ─────────────── */}
            {spotlights.length > 0 && (
              <div>
                <SectionHeader>Current Form</SectionHeader>
                <div className="grid grid-cols-1 gap-2">
                  {spotlights.map(sp => (
                    <SpotlightCard key={sp.field} spotlight={sp} onClick={() => navigate('/trends')} />
                  ))}
                </div>
              </div>
            )}

            {/* ── quick actions ─────────────── */}
            <div>
              <SectionHeader>Quick Actions</SectionHeader>
              <div className="grid grid-cols-2 gap-3">
                <QuickAction
                  label="All Sessions"
                  sub={`${practiceSessions.length} with metrics`}
                  icon={<ListIconSmall />}
                  onClick={() => navigate('/sessions')}
                />
                <QuickAction
                  label="Trends"
                  sub="Form over time"
                  icon={<ChartIconSmall />}
                  onClick={() => navigate('/trends')}
                />
                <QuickAction
                  label="Coaching"
                  sub="Session feedback"
                  icon={<BrainIconSmall />}
                  onClick={() => navigate('/coaching')}
                />
                <QuickAction
                  label="Coach Notes"
                  sub="Build coach voice"
                  icon={<PenIconSmall />}
                  onClick={() => navigate('/coach-notes')}
                />
              </div>
            </div>
          </>
        )}

        {!loading && sessions.length === 0 && (
          <div className="text-center py-8">
            <div className="text-4xl mb-3">🎯</div>
            <div className="text-gray-600 font-medium">No sessions yet</div>
            <div className="text-gray-400 text-sm mt-1">Upload your first practice video above</div>
          </div>
        )}
      </div>
    </div>
  )
}

// ── sub-components ─────────────────────────────────────────────────────────────

function SectionHeader({ children }: { children: React.ReactNode }) {
  return (
    <h2 className="text-xs font-semibold text-gray-400 uppercase tracking-wide mb-2">
      {children}
    </h2>
  )
}

const STAT_COLORS: Record<string, string> = {
  blue:  'bg-blue-50 text-blue-700',
  indigo:'bg-indigo-50 text-indigo-700',
  green: 'bg-green-50 text-green-700',
  gray:  'bg-gray-50 text-gray-500',
}

function StatCard({ value, label, color }: { value: string; label: string; color: string }) {
  return (
    <div className={`card p-3 text-center ${STAT_COLORS[color] ?? STAT_COLORS.gray}`}>
      <div className="text-xl font-bold">{value}</div>
      <div className="text-[10px] mt-0.5 opacity-75">{label}</div>
    </div>
  )
}

interface SpotlightData {
  field: string
  view: ViewType
  label: string
  unit: string
  current: number
  good_range: [number | null, number | null] | null
  history: { date: string; value: number }[]
}

function SpotlightCard({ spotlight: sp, onClick }: { spotlight: SpotlightData; onClick: () => void }) {
  const good = isGood(sp.current, sp.good_range)
  const displayVal = Math.abs(sp.current) < 1
    ? sp.current.toFixed(3)
    : Math.abs(sp.current) < 10
    ? sp.current.toFixed(1)
    : sp.current.toFixed(0)

  return (
    <button
      onClick={onClick}
      className={`card w-full text-left p-4 flex items-center gap-4 border
                  ${good ? 'border-green-200 bg-green-50' : 'border-red-200 bg-red-50'}
                  active:scale-[0.98] transition-transform`}
    >
      {/* status dot */}
      <div className={`w-2.5 h-2.5 rounded-full flex-shrink-0 ${good ? 'bg-green-500' : 'bg-red-500'}`} />

      <div className="flex-1 min-w-0">
        <div className="text-xs text-gray-500 capitalize">{sp.view} view · {sp.label}</div>
        <div className={`text-lg font-bold mt-0.5 ${good ? 'text-green-700' : 'text-red-700'}`}>
          {displayVal}{sp.unit}
          <span className="text-xs font-normal ml-1.5 text-gray-400">
            {good ? 'in range' : 'flagged'}
          </span>
        </div>
      </div>

      {/* mini sparkline */}
      {sp.history.length >= 3 && (
        <div className="w-16 h-8 flex-shrink-0">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={sp.history}>
              <Line
                dataKey="value"
                stroke={good ? '#16a34a' : '#dc2626'}
                strokeWidth={1.5}
                dot={false}
              />
              <Tooltip content={() => null} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}
    </button>
  )
}

function QuickAction({
  label, sub, icon, onClick,
}: {
  label: string; sub: string; icon: React.ReactNode; onClick: () => void
}) {
  return (
    <button
      onClick={onClick}
      className="card p-4 text-left active:scale-[0.97] transition-transform"
    >
      <div className="text-gray-600 mb-2">{icon}</div>
      <div className="font-semibold text-sm text-gray-800">{label}</div>
      <div className="text-xs text-gray-400 mt-0.5">{sub}</div>
    </button>
  )
}

// ── helpers ────────────────────────────────────────────────────────────────────

function capitalize(s: string) {
  return s.charAt(0).toUpperCase() + s.slice(1)
}

function calcStreak(sessions: SessionCard[]): number {
  if (sessions.length === 0) return 0
  const dates = [...new Set(sessions.map(s => s.date))].sort().reverse()
  let streak = 0
  let expected = new Date()
  expected.setHours(0, 0, 0, 0)

  for (const d of dates) {
    const dt = new Date(d + 'T00:00:00')
    const diff = Math.round((expected.getTime() - dt.getTime()) / (1000 * 60 * 60 * 24))
    if (diff <= 1) {
      streak++
      expected = dt
    } else {
      break
    }
  }
  return streak
}

function computeSpotlights(sessions: SessionCard[], trends: TrendData): SpotlightData[] {
  const results: SpotlightData[] = []
  const seenViews = new Set<ViewType>()

  for (const session of sessions) {
    const view = session.view as ViewType
    if (seenViews.has(view) || !session.has_metrics) continue
    seenViews.add(view)

    const spot = SPOTLIGHT[view]
    if (!spot) continue

    // Get current value from trend data
    const viewTrend = trends[view]
    const series = viewTrend?.series?.[spot.field] ?? []
    if (series.length === 0) continue

    const latest = series[series.length - 1]
    results.push({
      field:      spot.field,
      view,
      label:      spot.label,
      unit:       spot.unit,
      current:    latest.value,
      good_range: spot.good_range,
      history:    series.slice(-8).map(p => ({ date: p.date, value: p.value })),
    })
  }
  return results
}

// ── inline icons ───────────────────────────────────────────────────────────────

function ListIconSmall() {
  return (
    <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
      <path strokeLinecap="round" strokeLinejoin="round"
        d="M8.25 6.75h12M8.25 12h12m-12 5.25h12M3.75 6.75h.007v.008H3.75V6.75zm.375 0a.375.375 0 11-.75 0 .375.375 0 01.75 0zM3.75 12h.007v.008H3.75V12zm.375 0a.375.375 0 11-.75 0 .375.375 0 01.75 0zm-.375 5.25h.007v.008H3.75v-.008zm.375 0a.375.375 0 11-.75 0 .375.375 0 01.75 0z" />
    </svg>
  )
}

function ChartIconSmall() {
  return (
    <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
      <path strokeLinecap="round" strokeLinejoin="round"
        d="M2.25 18L9 11.25l4.306 4.307a11.95 11.95 0 015.814-5.519l2.74-1.22m0 0l-5.94-2.28m5.94 2.28l-2.28 5.941" />
    </svg>
  )
}

function BrainIconSmall() {
  return (
    <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
      <path strokeLinecap="round" strokeLinejoin="round"
        d="M8.625 9.75a.375.375 0 11-.75 0 .375.375 0 01.75 0zm0 0H8.25m4.125 0a.375.375 0 11-.75 0 .375.375 0 01.75 0zm0 0H12m4.125 0a.375.375 0 11-.75 0 .375.375 0 01.75 0zm0 0h-.375m-13.5 3.01c0 1.6 1.123 2.994 2.707 3.227 1.087.16 2.185.283 3.293.369V21l4.184-4.183a1.14 1.14 0 01.778-.332 48.294 48.294 0 005.83-.498c1.585-.233 2.708-1.626 2.708-3.228V6.741c0-1.602-1.123-2.995-2.707-3.228A48.394 48.394 0 0012 3c-2.392 0-4.744.175-7.043.513C3.373 3.746 2.25 5.14 2.25 6.741v6.018z" />
    </svg>
  )
}

function PenIconSmall() {
  return (
    <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
      <path strokeLinecap="round" strokeLinejoin="round"
        d="M16.862 4.487l1.687-1.688a1.875 1.875 0 112.652 2.652L10.582 16.07a4.5 4.5 0 01-1.897 1.13L6 18l.8-2.685a4.5 4.5 0 011.13-1.897l8.932-8.931zm0 0L19.5 7.125M18 14v4.75A2.25 2.25 0 0115.75 21H5.25A2.25 2.25 0 013 18.75V8.25A2.25 2.25 0 015.25 6H10" />
    </svg>
  )
}
