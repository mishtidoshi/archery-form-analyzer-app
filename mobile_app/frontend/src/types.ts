export type ViewType = 'face' | 'back' | 'target'
export type MetricStatus = 'good' | 'low' | 'high' | 'observation'

export interface SessionCard {
  id: string
  date: string
  view: ViewType
  location: string
  distance_m: number | null
  arrows: number
  has_metrics: boolean
  metrics_good: number
  metrics_flagged: number
  notes: string
}

export interface EnrichedMetric {
  field: string
  label: string
  unit: string
  value: number
  std: number | null
  good_range: [number | null, number | null] | null
  baseline?: number
  status: MetricStatus
}

export interface Session {
  date: string
  view: ViewType
  location: string
  distance_m: number | null
  session_type: string
  notes: string
  metrics: Record<string, number>
  enriched_metrics: EnrichedMetric[]
  review?: Record<string, unknown>
  detection_notes?: string
}

export interface TrendPoint {
  date: string
  value: number
  session_id: string
  arrows: number
}

export interface MetricMeta {
  field: string
  label: string
  unit: string
  good_range: [number | null, number | null] | null
  baseline?: number
}

export interface TrendView {
  series: Record<string, TrendPoint[]>
  meta: Record<string, MetricMeta>
}

export interface TrendData {
  face?: TrendView
  back?: TrendView
  target?: TrendView
}

export interface Job {
  status: 'queued' | 'analyzing' | 'done' | 'failed'
  session_id?: string
  message?: string
  view: ViewType
  date: string
}

export interface Archer {
  id: string
  name: string
  has_password: boolean
}

export interface AuthResult {
  token: string
  archer_id: string
  name: string
}
