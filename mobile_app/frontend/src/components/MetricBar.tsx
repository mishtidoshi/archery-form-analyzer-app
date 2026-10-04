import type { EnrichedMetric } from '../types'

interface Props {
  metric: EnrichedMetric
}

const STATUS_COLORS = {
  good:        'bg-green-500',
  low:         'bg-amber-400',
  high:        'bg-red-500',
  observation: 'bg-blue-400',
}

const STATUS_LABEL = {
  good:        'In range',
  low:         'Low',
  high:        'High',
  observation: 'Tracking',
}

const STATUS_BG = {
  good:        'bg-green-50 border-green-200',
  low:         'bg-amber-50 border-amber-200',
  high:        'bg-red-50 border-red-200',
  observation: 'bg-blue-50 border-blue-200',
}

/** Convert a value to a 0–100 fill percentage within or around the good range. */
function toPercent(value: number, meta: EnrichedMetric): number {
  const gr = meta.good_range
  if (!gr) return 50   // observation-only: show centered

  const [lo, hi] = gr
  // Determine display range: 20% padding around the good range
  const rangeLo = lo ?? (hi! - Math.abs(hi!) * 0.5)
  const rangeHi = hi ?? (lo! + Math.abs(lo!) * 0.5)
  const span = (rangeHi - rangeLo) || 1
  const expanded = span * 1.4
  const displayLo = (rangeLo + rangeHi) / 2 - expanded / 2
  const displayHi = (rangeLo + rangeHi) / 2 + expanded / 2

  return Math.max(2, Math.min(98, ((value - displayLo) / (displayHi - displayLo)) * 100))
}

/** Good-range band as % of the display range */
function goodBandPercents(meta: EnrichedMetric): [number, number] {
  const gr = meta.good_range
  if (!gr) return [30, 70]
  const [lo, hi] = gr

  const fakeMetric: EnrichedMetric = { ...meta }
  const loP = lo != null ? toPercent(lo, fakeMetric) : 0
  const hiP = hi != null ? toPercent(hi, fakeMetric) : 100
  return [Math.max(0, loP), Math.min(100, hiP)]
}

export default function MetricBar({ metric }: Props) {
  const fillPct = toPercent(metric.value, metric)
  const [bandLo, bandHi] = goodBandPercents(metric)
  const color = STATUS_COLORS[metric.status]
  const border = STATUS_BG[metric.status]

  const displayValue = metric.value.toFixed(
    Math.abs(metric.value) < 1 ? 3 : Math.abs(metric.value) < 10 ? 2 : 1
  )

  return (
    <div className={`card border px-4 py-3 ${border}`}>
      <div className="flex items-center justify-between mb-2">
        <span className="text-sm font-medium text-gray-700">{metric.label}</span>
        <div className="flex items-center gap-2">
          <span className="text-sm font-bold text-gray-900">
            {displayValue}{metric.unit}
            {metric.std != null && (
              <span className="text-xs font-normal text-gray-400 ml-0.5">
                ±{metric.std.toFixed(2)}
              </span>
            )}
          </span>
          <span
            className={`text-xs px-2 py-0.5 rounded-full font-medium
              ${metric.status === 'good'        ? 'badge-good'  : ''}
              ${metric.status === 'low'         ? 'badge-low'   : ''}
              ${metric.status === 'high'        ? 'badge-high'  : ''}
              ${metric.status === 'observation' ? 'badge-obs'   : ''}
            `}
          >
            {STATUS_LABEL[metric.status]}
          </span>
        </div>
      </div>

      {/* Track */}
      <div className="relative h-2.5 bg-gray-100 rounded-full overflow-hidden">
        {/* Good-range band */}
        <div
          className="absolute top-0 bottom-0 bg-green-200 opacity-60"
          style={{ left: `${bandLo}%`, right: `${100 - bandHi}%` }}
        />
        {/* Value marker */}
        <div
          className={`absolute top-0 bottom-0 w-2 -translate-x-1/2 rounded-full ${color}`}
          style={{ left: `${fillPct}%` }}
        />
      </div>

      {metric.good_range && (
        <div className="flex justify-between mt-1 text-[10px] text-gray-400">
          <span>
            {metric.good_range[0] != null
              ? `${metric.good_range[0]}${metric.unit}`
              : `< ${metric.good_range[1]}${metric.unit}`}
          </span>
          {metric.good_range[0] != null && metric.good_range[1] != null && (
            <span>{`${metric.good_range[1]}${metric.unit}`}</span>
          )}
        </div>
      )}
      {metric.baseline != null && (
        <div className="mt-1 text-[10px] text-blue-400">
          Baseline ≈ {metric.baseline}{metric.unit} (tracking only)
        </div>
      )}
    </div>
  )
}
