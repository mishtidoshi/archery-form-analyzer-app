import { useState, useEffect } from 'react'
import { getCoachNotes, addCoachNoteEntry } from '../api'

const CATEGORIES = [
  'anchor', 'back tension', 'hold', 'follow-through',
  'shoulder', 'head position', 'grip', 'mental', 'other',
]

// Fixed, generic options every archer gets regardless of who their real coaches are —
// never a real name, so these need no anonymizing.
const GENERIC_COACH_OPTIONS = ['Self observation', 'Other']

export default function CoachNotesPage() {
  const [loading, setLoading] = useState(true)

  // realCoaches holds this archer's own coach names (from their own profile.json,
  // already excludes anyone marked hidden_from_app) — scoped to the signed-in archer
  // by the backend (GET /api/coach-notes reads only the caller's own profile), so
  // another archer on the same deployment never sees these. Shown as-is in the picker
  // and in generated feedback — see docs/WORKFLOW.md.
  const [realCoaches, setRealCoaches] = useState<string[]>([])
  const [coach, setCoach] = useState('Self observation')
  const [customCoach, setCustomCoach] = useState('')
  const [entryDate, setEntryDate] = useState(new Date().toISOString().slice(0, 10))
  const [cue, setCue] = useState('')
  const [context, setContext] = useState('')
  const [category, setCategory] = useState('anchor')
  const [addLoading, setAddLoading] = useState(false)
  const [addMsg, setAddMsg] = useState('')

  useEffect(() => {
    getCoachNotes().then(data => {
      setRealCoaches(data.coaches)
      if (data.coaches.length > 0) setCoach(data.coaches[0])
      setLoading(false)
    }).catch(() => setLoading(false))
  }, [])

  const pickerOptions = [...realCoaches, ...GENERIC_COACH_OPTIONS]

  const handleAddEntry = async () => {
    if (!cue.trim()) return
    setAddLoading(true)
    setAddMsg('')
    try {
      const coachName = coach === 'Other' ? customCoach.trim() || 'Unknown' : coach
      await addCoachNoteEntry({ coach: coachName, date: entryDate, cue: cue.trim(), context, category })
      // A name typed via "Other" is now a saved coach — refresh so it appears as its
      // own button immediately.
      const data = await getCoachNotes()
      setRealCoaches(data.coaches)
      setCustomCoach('')
      setCue('')
      setContext('')
      setAddMsg('Cue saved')
      setTimeout(() => setAddMsg(''), 2000)
    } catch {
      setAddMsg('Failed to add entry')
    } finally {
      setAddLoading(false)
    }
  }

  return (
    <div className="flex flex-col h-full">
      <div className="bg-blue-600 px-5 pt-5 pb-4 flex-shrink-0">
        <h1 className="text-xl font-bold text-white">Coach Notes</h1>
        <p className="text-blue-200 text-xs mt-0.5">Log a cue from a coaching session</p>
      </div>

      <div className="flex-1 overflow-y-auto flex flex-col">
        <div className="p-4 flex-1 flex flex-col space-y-4">
          {loading && <div className="text-center py-12 text-gray-400 text-sm">Loading…</div>}

          {!loading && (
            <>
              <div className="card p-4 space-y-4">
                {/* Coach */}
                <div>
                  <label className="text-xs font-semibold text-gray-500 block mb-1.5">Coach</label>
                  <div className="flex flex-wrap gap-2">
                    {pickerOptions.map(c => (
                      <button
                        key={c}
                        onClick={() => setCoach(c)}
                        className={`px-3 py-1.5 rounded-xl text-xs font-medium border transition-colors ${
                          coach === c
                            ? 'border-blue-500 bg-blue-50 text-blue-700'
                            : 'border-gray-200 bg-gray-50 text-gray-600'
                        }`}
                      >
                        {c}
                      </button>
                    ))}
                  </div>
                  {coach === 'Other' && (
                    <input
                      type="text"
                      value={customCoach}
                      onChange={e => setCustomCoach(e.target.value)}
                      placeholder="Coach name…"
                      className="mt-2 w-full border border-gray-200 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-400"
                    />
                  )}
                </div>

                {/* Date */}
                <div>
                  <label className="text-xs font-semibold text-gray-500 block mb-1.5">Date</label>
                  <input
                    type="date"
                    value={entryDate}
                    onChange={e => setEntryDate(e.target.value)}
                    className="w-full border border-gray-200 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-400"
                  />
                </div>

                {/* Category */}
                <div>
                  <label className="text-xs font-semibold text-gray-500 block mb-1.5">Category</label>
                  <div className="flex flex-wrap gap-2">
                    {CATEGORIES.map(c => (
                      <button
                        key={c}
                        onClick={() => setCategory(c)}
                        className={`px-2.5 py-1 rounded-full text-xs font-medium transition-colors ${
                          category === c
                            ? 'bg-blue-600 text-white'
                            : 'bg-gray-100 text-gray-500'
                        }`}
                      >
                        {c}
                      </button>
                    ))}
                  </div>
                </div>

                {/* Cue */}
                <div>
                  <label className="text-xs font-semibold text-gray-500 block mb-1.5">
                    Cue <span className="text-red-400">*</span>
                  </label>
                  <textarea
                    value={cue}
                    onChange={e => setCue(e.target.value)}
                    rows={3}
                    placeholder="e.g. Pull through the clicker, not to the clicker"
                    className="w-full border border-gray-200 rounded-lg px-3 py-2 text-sm resize-none focus:outline-none focus:ring-2 focus:ring-blue-400"
                  />
                </div>

                {/* Context */}
                <div>
                  <label className="text-xs font-semibold text-gray-500 block mb-1.5">
                    Context / explanation (optional)
                  </label>
                  <textarea
                    value={context}
                    onChange={e => setContext(e.target.value)}
                    rows={2}
                    placeholder="e.g. Said after a few off shots in a row — arrows drifting right"
                    className="w-full border border-gray-200 rounded-lg px-3 py-2 text-sm resize-none focus:outline-none focus:ring-2 focus:ring-blue-400"
                  />
                </div>
              </div>

              {addMsg && (
                <div className={`text-sm text-center font-medium py-2 rounded-xl ${
                  addMsg.includes('Failed') ? 'text-red-600 bg-red-50' : 'text-green-600 bg-green-50'
                }`}>
                  {addMsg}
                </div>
              )}

              <button
                className="btn-primary w-full"
                onClick={handleAddEntry}
                disabled={!cue.trim() || addLoading}
              >
                {addLoading ? 'Saving…' : 'Save Cue'}
              </button>
            </>
          )}
        </div>
      </div>
    </div>
  )
}
