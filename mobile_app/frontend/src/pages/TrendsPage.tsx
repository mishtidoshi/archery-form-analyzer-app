import { useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, ReferenceLine, ReferenceArea,
} from 'recharts'
import { getTrends } from '../api'
import type { TrendData, TrendView, MetricMeta, TrendPoint, ViewType } from '../types'

const VIEWS: ViewType[] = ['face', 'back', 'target']
const VIEW_LABELS: Record<ViewType, string> = {
  face: 'Face View', back: 'Back View', target: 'Target View',
}

export default function TrendsPage() {
  const [data, setData] = useState<TrendData>({})
  const [activeView, setActiveView] = useState<ViewType>('face')
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    getTrends().then(d => {
      setData(d)
      setLoading(false)
    }).catch(() => setLoading(false))
  }, [])

  const viewData: TrendView | undefined = data[activeView]

  return (
    <div className="page">
      <div className="bg-blue-600 px-5 pt-5 pb-4 space-y-3">
        <h1 className="text-xl font-bold text-white">Trends</h1>

        <div className="flex gap-2">
          {VIEWS.map(v => (
            <button
              key={v}
              onClick={() => setActiveView(v)}
              className={`flex-1 py-1.5 rounded-xl text-xs font-medium transition-colors ${
                activeView === v
                  ? 'bg-white text-blue-700'
                  : 'bg-blue-500 text-blue-100'
              }`}
            >
              {v.charAt(0).toUpperCase() + v.slice(1)}
            </button>
          ))}
        </div>
      </div>

      <div className="p-4 space-y-5">
        {loading && (
          <div className="text-center py-12 text-gray-400 text-sm">Loading trend data…</div>
        )}

        {!loading && !viewData && (
          <div className="text-center py-12">
            <div className="text-3xl mb-2">📈</div>
            <div className="text-gray-500 text-sm">No {VIEW_LABELS[activeView]} sessions yet</div>
          </div>
        )}

        {viewData &&
          Object.entries(viewData.series).map(([field, points]) => {
            const meta = viewData.meta[field]
            if (!meta || points.length === 0) return null
            return (
              <MetricChart
                key={field}
                points={points}
                meta={meta}
              />
            )
          })}
      </div>
    </div>
  )
}

// ── chart component ────────────────────────────────────────────────────────────

interface ChartProps {
  points: TrendPoint[]
  meta: MetricMeta
}

function MetricChart({ points, meta }: ChartProps) {
  const navigate = useNavigate()
  const gr = meta.good_range

  // Compute y-axis domain with 20% padding around the good range and data
  const values = points.map(p => p.value)
  const dataMin = Math.min(...values)
  const dataMax = Math.max(...values)
  const lo = gr?.[0] ?? dataMin
  const hi = gr?.[1] ?? dataMax
  const span = Math.max(hi - lo, 0.001)
  const yMin = Math.min(dataMin, lo) - span * 0.3
  const yMax = Math.max(dataMax, hi) + span * 0.3

  // Latest value + trend direction
  const latest = points[points.length - 1]?.value
  const prev   = points[points.length - 2]?.value
  const trend  = prev == null ? null : latest - prev
  const trendIcon = trend == null ? '' : trend > 0.001 ? '↑' : trend < -0.001 ? '↓' : '→'

  // Format x-axis labels as short date
  const formatX = (d: string) => {
    try {
      return new Date(d + 'T12:00:00').toLocaleDateString('en-US', { month: 'short', day: 'numeric' })
    } catch { return d }
  }

  const isInRange = (v: number) => {
    if (!gr) return true
    const [l, h] = gr
    return (l == null || v >= l) && (h == null || v <= h)
  }

  return (
    <div className="card p-4">
      <div className="flex items-baseline justify-between mb-3">
        <div>
          <h3 className="text-sm font-semibold text-gray-800">{meta.label}</h3>
          <div className="text-xs text-gray-400 mt-0.5">
            {points.length} session{points.length !== 1 ? 's' : ''}
          </div>
        </div>
        <div className="text-right">
          {latest != null && (
            <div
              className={`text-base font-bold ${
                isInRange(latest) ? 'text-green-600' : 'text-red-600'
              }`}
            >
              {latest.toFixed(Math.abs(latest) < 1 ? 3 : 1)}{meta.unit}
            </div>
          )}
          {trendIcon && (
            <div className="text-xs text-gray-400">{trendIcon} vs last session</div>
          )}
        </div>
      </div>

      <ResponsiveContainer width="100%" height={160}>
        <LineChart
          data={points}
          margin={{ top: 4, right: 4, left: -24, bottom: 0 }}
          onClick={e => {
            const payload = e?.activePayload?.[0]?.payload as TrendPoint | undefined
            if (payload?.session_id) navigate(`/sessions/${payload.session_id}`)
          }}
        >
          <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
          <XAxis
            dataKey="date"
            tickFormatter={formatX}
            tick={{ fontSize: 9, fill: '#9ca3af' }}
            interval="preserveStartEnd"
          />
          <YAxis
            domain={[yMin, yMax]}
            tick={{ fontSize: 9, fill: '#9ca3af' }}
            tickFormatter={v => v.toFixed(Math.abs(v) < 1 ? 2 : 0)}
          />
          <Tooltip
            content={({ payload }) => {
              const p = payload?.[0]?.payload as TrendPoint | undefined
              if (!p) return null
              return (
                <div className="bg-white border border-gray-200 rounded-lg shadow px-3 py-2 text-xs">
                  <div className="font-semibold text-gray-700">{formatX(p.date)}</div>
                  <div className={isInRange(p.value) ? 'text-green-600' : 'text-red-600'}>
                    {p.value.toFixed(3)}{meta.unit}
                  </div>
                  <div className="text-gray-400">{p.arrows} arrows</div>
                </div>
              )
            }}
          />
          {/* Good range band */}
          {gr && gr[0] != null && gr[1] != null && (
            <ReferenceArea
              y1={gr[0]} y2={gr[1]}
              fill="#22c55e" fillOpacity={0.08}
              stroke="#22c55e" strokeOpacity={0.3}
            />
          )}
          {gr && gr[0] == null && gr[1] != null && (
            <ReferenceLine y={gr[1]} stroke="#22c55e" strokeDasharray="4 4" strokeWidth={1.5} />
          )}
          {gr && gr[1] == null && gr[0] != null && (
            <ReferenceLine y={gr[0]} stroke="#22c55e" strokeDasharray="4 4" strokeWidth={1.5} />
          )}
          {meta.baseline != null && (
            <ReferenceLine y={meta.baseline} stroke="#3b82f6" strokeDasharray="4 4" strokeWidth={1} />
          )}
          <Line
            dataKey="value"
            stroke="#2563eb"
            strokeWidth={2}
            dot={(props) => {
              const { cx, cy, payload } = props as { cx: number; cy: number; payload: TrendPoint }
              const inRange = isInRange(payload.value)
              return (
                <circle
                  key={`dot-${cx}-${cy}`}
                  cx={cx} cy={cy} r={4}
                  fill={inRange ? '#22c55e' : '#ef4444'}
                  stroke="white" strokeWidth={1.5}
                />
              )
            }}
            activeDot={{ r: 6, strokeWidth: 2 }}
          />
        </LineChart>
      </ResponsiveContainer>

      {gr && (
        <div className="flex items-center gap-2 mt-2 text-[10px] text-gray-400">
          <span className="flex items-center gap-1">
            <span className="inline-block w-3 h-1.5 bg-green-200 rounded" />
            Good range
            {gr[0] != null && gr[1] != null && ` [${gr[0]}, ${gr[1]}]${meta.unit}`}
            {gr[0] == null && gr[1] != null && ` ≤ ${gr[1]}${meta.unit}`}
            {gr[1] == null && gr[0] != null && ` ≥ ${gr[0]}${meta.unit}`}
          </span>
          <span className="flex items-center gap-1">
            <span className="inline-block w-2 h-2 rounded-full bg-green-500" /> In range
          </span>
          <span className="flex items-center gap-1">
            <span className="inline-block w-2 h-2 rounded-full bg-red-500" /> Flagged
          </span>
        </div>
      )}
    </div>
  )
}
