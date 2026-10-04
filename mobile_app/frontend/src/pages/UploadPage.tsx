import { useState, useRef, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import { uploadVideo, getJob } from '../api'
import type { ViewType } from '../types'

const VIEW_OPTIONS: { value: ViewType; label: string; desc: string }[] = [
  { value: 'face', label: 'Face View', desc: 'Camera on your face/anchor side' },
  { value: 'back', label: 'Back View', desc: 'Camera behind your spine (T-shape)' },
  { value: 'target', label: 'Target View', desc: 'Camera directly behind, facing target' },
]

type UploadState = 'idle' | 'uploading' | 'analyzing' | 'done' | 'failed'

export default function UploadPage() {
  const navigate = useNavigate()
  const fileRef = useRef<HTMLInputElement>(null)

  const [file, setFile] = useState<File | null>(null)
  const [view, setView] = useState<ViewType>('face')
  const [date, setDate] = useState(new Date().toISOString().slice(0, 10))
  const [location, setLocation] = useState('backyard')
  const [distanceM, setDistanceM] = useState(18)
  const [notes, setNotes] = useState('')

  const [state, setState] = useState<UploadState>('idle')
  const [jobId, setJobId] = useState<string | null>(null)
  const [message, setMessage] = useState('')
  const [sessionId, setSessionId] = useState<string | null>(null)

  // Poll job status while analyzing
  useEffect(() => {
    if (!jobId || state !== 'analyzing') return
    const iv = setInterval(async () => {
      try {
        const job = await getJob(jobId)
        if (job.status === 'done') {
          setState('done')
          setSessionId(job.session_id ?? null)
          setMessage(job.message ?? 'Analysis complete!')
          clearInterval(iv)
        } else if (job.status === 'failed') {
          setState('failed')
          setMessage(job.message ?? 'Analysis failed.')
          clearInterval(iv)
        }
      } catch {
        // keep polling
      }
    }, 2000)
    return () => clearInterval(iv)
  }, [jobId, state])

  const handleFile = (f: File | null) => {
    if (!f) return
    setFile(f)
    setState('idle')
    setMessage('')
    setSessionId(null)
  }

  const handleSubmit = async () => {
    if (!file) return
    setState('uploading')
    setMessage('')
    try {
      const fd = new FormData()
      fd.append('file', file)
      fd.append('view', view)
      fd.append('date', date)
      fd.append('location', location)
      fd.append('distance_m', String(distanceM))
      fd.append('notes', notes)

      const { job_id } = await uploadVideo(fd)
      setJobId(job_id)
      setState('analyzing')
    } catch (err: unknown) {
      setState('failed')
      setMessage('Upload failed. Is the backend running?')
    }
  }

  const reset = () => {
    setFile(null)
    setState('idle')
    setJobId(null)
    setMessage('')
    setSessionId(null)
    if (fileRef.current) fileRef.current.value = ''
  }

  return (
    <div className="page">
      <div className="bg-blue-600 px-5 pt-5 pb-5">
        <h1 className="text-xl font-bold text-white">Upload Video</h1>
        <p className="text-blue-200 text-xs mt-0.5">Add a practice video for analysis</p>
      </div>

      <div className="p-4 space-y-4">
        {/* Drop zone */}
        <div
          className={`card border-2 border-dashed p-8 text-center cursor-pointer transition-colors ${
            file ? 'border-blue-400 bg-blue-50' : 'border-gray-300 hover:border-blue-300'
          }`}
          onClick={() => fileRef.current?.click()}
        >
          <input
            ref={fileRef}
            type="file"
            accept="video/*,.MOV,.mov,.mp4"
            className="hidden"
            onChange={e => handleFile(e.target.files?.[0] ?? null)}
          />
          {file ? (
            <>
              <div className="text-3xl mb-2">🎥</div>
              <p className="font-medium text-blue-700 text-sm truncate px-2">{file.name}</p>
              <p className="text-xs text-gray-400 mt-1">{(file.size / 1e6).toFixed(1)} MB</p>
            </>
          ) : (
            <>
              <div className="text-3xl mb-2">📹</div>
              <p className="font-medium text-gray-600 text-sm">Tap to choose video</p>
              <p className="text-xs text-gray-400 mt-1">.MOV or .mp4 from camera roll</p>
            </>
          )}
        </div>

        {/* View type */}
        <div className="card p-4 space-y-2">
          <label className="text-xs font-semibold text-gray-500 uppercase tracking-wide">
            Camera View
          </label>
          {VIEW_OPTIONS.map(opt => (
            <button
              key={opt.value}
              onClick={() => setView(opt.value)}
              className={`w-full text-left px-4 py-3 rounded-xl border-2 transition-colors ${
                view === opt.value
                  ? 'border-blue-500 bg-blue-50'
                  : 'border-gray-100 bg-gray-50'
              }`}
            >
              <div className={`font-semibold text-sm ${view === opt.value ? 'text-blue-700' : 'text-gray-700'}`}>
                {opt.label}
              </div>
              <div className="text-xs text-gray-400 mt-0.5">{opt.desc}</div>
            </button>
          ))}
        </div>

        {/* Metadata */}
        <div className="card p-4 space-y-3">
          <label className="text-xs font-semibold text-gray-500 uppercase tracking-wide">
            Session Details
          </label>

          <div>
            <label className="text-xs text-gray-500 mb-1 block">Date</label>
            <input
              type="date"
              value={date}
              onChange={e => setDate(e.target.value)}
              className="w-full border border-gray-200 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-400"
            />
          </div>

          <div>
            <label className="text-xs text-gray-500 mb-1 block">Location</label>
            <select
              value={location}
              onChange={e => setLocation(e.target.value)}
              className="w-full border border-gray-200 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-400 bg-white"
            >
              <option value="backyard">Backyard</option>
              <option value="range">Range</option>
              <option value="indoor">Indoor</option>
              <option value="other">Other</option>
            </select>
          </div>

          <div>
            <label className="text-xs text-gray-500 mb-1 block">Distance (m)</label>
            <input
              type="number"
              value={distanceM}
              onChange={e => setDistanceM(Number(e.target.value))}
              min={5}
              max={90}
              className="w-full border border-gray-200 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-400"
            />
          </div>

          <div>
            <label className="text-xs text-gray-500 mb-1 block">Notes (optional)</label>
            <textarea
              value={notes}
              onChange={e => setNotes(e.target.value)}
              rows={2}
              placeholder=""
              className="w-full border border-gray-200 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-400 resize-none"
            />
          </div>
        </div>

        {/* Status / result */}
        {state === 'analyzing' && (
          <div className="card p-4 flex items-center gap-3 border border-blue-200 bg-blue-50">
            <div className="w-5 h-5 rounded-full border-2 border-blue-500 border-t-transparent animate-spin" />
            <div>
              <div className="text-sm font-medium text-blue-700">Analyzing video…</div>
              <div className="text-xs text-blue-500 mt-0.5">Running YOLO pose detection</div>
            </div>
          </div>
        )}

        {state === 'done' && (
          <div className="card p-4 border border-green-200 bg-green-50 space-y-3">
            <div className="flex items-start gap-2">
              <span className="text-green-600 text-lg">✓</span>
              <div>
                <div className="text-sm font-semibold text-green-700">Analysis complete</div>
                <div className="text-xs text-green-600 mt-0.5">{message}</div>
              </div>
            </div>
            <div className="flex gap-2">
              {sessionId && (
                <button
                  className="btn-primary text-sm py-2 flex-1"
                  onClick={() => navigate(`/sessions/${sessionId}`)}
                >
                  View Session
                </button>
              )}
              <button className="btn-secondary text-sm py-2 flex-1" onClick={reset}>
                Upload Another
              </button>
            </div>
          </div>
        )}

        {state === 'failed' && (
          <div className="card p-4 border border-red-200 bg-red-50 space-y-2">
            <div className="text-sm font-semibold text-red-700">Something went wrong</div>
            <div className="text-xs text-red-600">{message}</div>
            <button className="btn-secondary text-sm py-2 w-full" onClick={reset}>
              Try Again
            </button>
          </div>
        )}

        {/* Submit */}
        {(state === 'idle' || state === 'uploading') && (
          <button
            className="btn-primary w-full"
            onClick={handleSubmit}
            disabled={!file || state === 'uploading'}
          >
            {state === 'uploading' ? 'Uploading…' : 'Analyze Video'}
          </button>
        )}
      </div>
    </div>
  )
}
