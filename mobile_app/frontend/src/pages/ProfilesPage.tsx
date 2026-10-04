import { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { getArchers, login, setArcherPassword, createArcher } from '../api'
import { useAuth } from '../auth'
import type { Archer } from '../types'

const AVATAR_COLORS = [
  'bg-blue-100 text-blue-700', 'bg-indigo-100 text-indigo-700',
  'bg-green-100 text-green-700', 'bg-amber-100 text-amber-700',
  'bg-rose-100 text-rose-700', 'bg-purple-100 text-purple-700',
]

function initials(name: string): string {
  const parts = name.trim().split(/\s+/)
  return ((parts[0]?.[0] ?? '') + (parts[1]?.[0] ?? '')).toUpperCase() || '?'
}

function colorFor(id: string): string {
  let hash = 0
  for (const ch of id) hash = (hash * 31 + ch.charCodeAt(0)) % AVATAR_COLORS.length
  return AVATAR_COLORS[hash]
}

function slugify(name: string): string {
  return name.trim().toLowerCase().replace(/[^a-z0-9]+/g, '_').replace(/^_+|_+$/g, '').slice(0, 32)
}

type Mode = { kind: 'picker' } | { kind: 'unlock'; archer: Archer } | { kind: 'create' }

export default function ProfilesPage() {
  const navigate = useNavigate()
  const { login: setAuth } = useAuth()

  const [archers, setArchers] = useState<Archer[]>([])
  const [loading, setLoading] = useState(true)
  const [mode, setMode] = useState<Mode>({ kind: 'picker' })
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)

  useEffect(() => {
    getArchers().then(setArchers).finally(() => setLoading(false))
  }, [])

  const onUnlocked = (token: string, archerId: string, name: string) => {
    setAuth(token, archerId, name)
    navigate('/')
  }

  return (
    <div className="page bg-blue-600">
      <div className="px-6 pt-12 pb-8 text-center">
        <div className="text-4xl mb-2">🏹</div>
        <h1 className="text-xl font-bold text-white">Who's shooting?</h1>
        <p className="text-blue-200 text-xs mt-1">Pick a profile to continue</p>
      </div>

      <div className="px-6 pb-8">
        {loading && <div className="text-center text-blue-200 text-sm py-6">Loading profiles…</div>}

        {!loading && mode.kind === 'picker' && (
          <div className="grid grid-cols-3 gap-4">
            {archers.map(a => (
              <button
                key={a.id}
                onClick={() => { setError(''); setMode({ kind: 'unlock', archer: a }) }}
                className="flex flex-col items-center gap-2 active:scale-95 transition-transform"
              >
                <div className={`w-16 h-16 rounded-2xl flex items-center justify-center text-lg font-bold ${colorFor(a.id)}`}>
                  {initials(a.name)}
                </div>
                <span className="text-white text-xs font-medium truncate max-w-full">{a.name}</span>
              </button>
            ))}

            <button
              onClick={() => { setError(''); setMode({ kind: 'create' }) }}
              className="flex flex-col items-center gap-2 active:scale-95 transition-transform"
            >
              <div className="w-16 h-16 rounded-2xl flex items-center justify-center text-2xl font-bold bg-white/10 text-white border-2 border-dashed border-white/40">
                +
              </div>
              <span className="text-blue-200 text-xs font-medium">Add Archer</span>
            </button>
          </div>
        )}

        {mode.kind === 'unlock' && (
          <UnlockPanel
            archer={mode.archer}
            error={error}
            busy={busy}
            onCancel={() => setMode({ kind: 'picker' })}
            onSubmit={async (password) => {
              setBusy(true); setError('')
              try {
                const res = mode.archer.has_password
                  ? await login(mode.archer.id, password)
                  : await setArcherPassword(mode.archer.id, password)
                onUnlocked(res.token, res.archer_id, res.name)
              } catch (err: any) {
                setError(err?.response?.data?.detail ?? 'Something went wrong')
              } finally {
                setBusy(false)
              }
            }}
          />
        )}

        {mode.kind === 'create' && (
          <CreatePanel
            error={error}
            busy={busy}
            onCancel={() => setMode({ kind: 'picker' })}
            onSubmit={async (id, name, password) => {
              setBusy(true); setError('')
              try {
                const res = await createArcher(id, name, password)
                onUnlocked(res.token, res.archer_id, res.name)
              } catch (err: any) {
                setError(err?.response?.data?.detail ?? 'Something went wrong')
              } finally {
                setBusy(false)
              }
            }}
          />
        )}
      </div>
    </div>
  )
}

function UnlockPanel({
  archer, error, busy, onCancel, onSubmit,
}: {
  archer: Archer; error: string; busy: boolean
  onCancel: () => void; onSubmit: (password: string) => void
}) {
  const [password, setPassword] = useState('')
  const isNew = !archer.has_password

  return (
    <div className="card bg-white p-5 space-y-3">
      <div className="flex items-center gap-3">
        <div className={`w-10 h-10 rounded-xl flex items-center justify-center text-sm font-bold ${colorFor(archer.id)}`}>
          {initials(archer.name)}
        </div>
        <div>
          <div className="font-semibold text-gray-900 text-sm">{archer.name}</div>
          <div className="text-xs text-gray-400">{isNew ? 'Set up a password' : 'Enter your password'}</div>
        </div>
      </div>

      <input
        type="password"
        autoFocus
        value={password}
        onChange={e => setPassword(e.target.value)}
        onKeyDown={e => { if (e.key === 'Enter' && password) onSubmit(password) }}
        placeholder={isNew ? 'Choose a password' : 'Password'}
        className="w-full border border-gray-200 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-400"
      />

      {error && <div className="text-xs text-red-600">{error}</div>}

      <div className="flex gap-2">
        <button className="btn-secondary text-sm py-2 flex-1" onClick={onCancel}>Back</button>
        <button
          className="btn-primary text-sm py-2 flex-1"
          disabled={!password || busy}
          onClick={() => onSubmit(password)}
        >
          {busy ? 'Please wait…' : isNew ? 'Set Password' : 'Log In'}
        </button>
      </div>
    </div>
  )
}

function CreatePanel({
  error, busy, onCancel, onSubmit,
}: {
  error: string; busy: boolean
  onCancel: () => void; onSubmit: (id: string, name: string, password: string) => void
}) {
  const [name, setName] = useState('')
  const [id, setId] = useState('')
  const [idTouched, setIdTouched] = useState(false)
  const [password, setPassword] = useState('')
  const [confirm, setConfirm] = useState('')

  const effectiveId = idTouched ? id : slugify(name)
  const canSubmit = name.trim().length > 0 && effectiveId.length >= 2
    && password.length >= 4 && password === confirm

  return (
    <div className="card bg-white p-5 space-y-3">
      <div className="font-semibold text-gray-900 text-sm">New archer profile</div>

      <div>
        <label className="text-xs text-gray-500 mb-1 block">Name</label>
        <input
          value={name}
          onChange={e => setName(e.target.value)}
          placeholder="e.g. Jordan Lee"
          className="w-full border border-gray-200 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-400"
        />
      </div>

      <div>
        <label className="text-xs text-gray-500 mb-1 block">Profile ID</label>
        <input
          value={effectiveId}
          onChange={e => { setIdTouched(true); setId(e.target.value.toLowerCase()) }}
          placeholder="lowercase, no spaces"
          className="w-full border border-gray-200 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-400"
        />
      </div>

      <div>
        <label className="text-xs text-gray-500 mb-1 block">Password</label>
        <input
          type="password"
          value={password}
          onChange={e => setPassword(e.target.value)}
          placeholder="At least 4 characters"
          className="w-full border border-gray-200 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-400"
        />
      </div>

      <div>
        <label className="text-xs text-gray-500 mb-1 block">Confirm Password</label>
        <input
          type="password"
          value={confirm}
          onChange={e => setConfirm(e.target.value)}
          onKeyDown={e => { if (e.key === 'Enter' && canSubmit) onSubmit(effectiveId, name, password) }}
          className="w-full border border-gray-200 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-400"
        />
      </div>

      {error && <div className="text-xs text-red-600">{error}</div>}

      <div className="flex gap-2">
        <button className="btn-secondary text-sm py-2 flex-1" onClick={onCancel}>Back</button>
        <button
          className="btn-primary text-sm py-2 flex-1"
          disabled={!canSubmit || busy}
          onClick={() => onSubmit(effectiveId, name, password)}
        >
          {busy ? 'Creating…' : 'Create Profile'}
        </button>
      </div>
    </div>
  )
}
