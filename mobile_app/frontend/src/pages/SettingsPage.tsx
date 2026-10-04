import { useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import { getSessions, updateArcherName, changeArcherPassword, deleteArcherAccount } from '../api'
import { useAuth } from '../auth'

function errorDetail(err: unknown, fallback: string): string {
  const detail = (err as { response?: { data?: { detail?: unknown } } })?.response?.data?.detail
  return typeof detail === 'string' ? detail : fallback
}

export default function SettingsPage() {
  const navigate = useNavigate()
  const { archerId, archerName, setName, login } = useAuth()

  const [sessionCount, setSessionCount] = useState<number | null>(null)

  useEffect(() => {
    getSessions().then(s => setSessionCount(s.length)).catch(() => setSessionCount(null))
  }, [])

  return (
    <div className="page">
      <div className="bg-blue-600 px-4 pt-4 pb-5">
        <div className="flex items-center gap-2">
          <button
            onClick={() => navigate('/')}
            className="text-blue-200 -ml-1 p-1 hover:text-white transition-colors"
          >
            <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
            </svg>
          </button>
          <h1 className="text-base font-bold text-white">Settings</h1>
        </div>
      </div>

      <div className="p-4 space-y-4">
        <NameSection currentName={archerName ?? ''} onSaved={setName} />
        <PasswordSection
          onChanged={(token, name) => { if (archerId) login(token, archerId, name) }}
        />
        <DangerZone sessionCount={sessionCount} archerName={archerName ?? 'this profile'} />
      </div>
    </div>
  )
}

function NameSection({ currentName, onSaved }: { currentName: string; onSaved: (name: string) => void }) {
  const [name, setNameField] = useState(currentName)
  const [saving, setSaving] = useState(false)
  const [msg, setMsg] = useState('')
  const [error, setError] = useState('')

  const handleSave = async () => {
    setSaving(true); setMsg(''); setError('')
    try {
      const res = await updateArcherName(name)
      onSaved(res.name)
      setMsg('Saved')
    } catch (err) {
      setError(errorDetail(err, 'Could not save name'))
    } finally {
      setSaving(false)
    }
  }

  return (
    <div className="card p-4 space-y-3">
      <label className="text-xs font-semibold text-gray-500 uppercase tracking-wide">Profile</label>
      <div>
        <label className="text-xs text-gray-500 mb-1 block">Name</label>
        <input
          value={name}
          onChange={e => setNameField(e.target.value)}
          className="w-full border border-gray-200 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-400"
        />
      </div>
      {error && <div className="text-xs text-red-600">{error}</div>}
      {msg && <div className="text-xs text-green-600">{msg}</div>}
      <button
        className="btn-primary text-sm py-2 w-full"
        disabled={!name.trim() || name.trim() === currentName || saving}
        onClick={handleSave}
      >
        {saving ? 'Saving…' : 'Save Name'}
      </button>
    </div>
  )
}

function PasswordSection({ onChanged }: { onChanged: (token: string, name: string) => void }) {
  const [current, setCurrent] = useState('')
  const [next, setNext] = useState('')
  const [confirm, setConfirm] = useState('')
  const [saving, setSaving] = useState(false)
  const [msg, setMsg] = useState('')
  const [error, setError] = useState('')

  const canSubmit = current.length > 0 && next.length >= 4 && next === confirm

  const handleSave = async () => {
    setSaving(true); setMsg(''); setError('')
    try {
      const res = await changeArcherPassword(current, next)
      onChanged(res.token, res.name)
      setMsg('Password updated')
      setCurrent(''); setNext(''); setConfirm('')
    } catch (err) {
      setError(errorDetail(err, 'Could not change password'))
    } finally {
      setSaving(false)
    }
  }

  return (
    <div className="card p-4 space-y-3">
      <label className="text-xs font-semibold text-gray-500 uppercase tracking-wide">Change Password</label>

      <div>
        <label className="text-xs text-gray-500 mb-1 block">Current password</label>
        <input
          type="password"
          value={current}
          onChange={e => setCurrent(e.target.value)}
          className="w-full border border-gray-200 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-400"
        />
      </div>
      <div>
        <label className="text-xs text-gray-500 mb-1 block">New password</label>
        <input
          type="password"
          value={next}
          onChange={e => setNext(e.target.value)}
          placeholder="At least 4 characters"
          className="w-full border border-gray-200 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-400"
        />
      </div>
      <div>
        <label className="text-xs text-gray-500 mb-1 block">Confirm new password</label>
        <input
          type="password"
          value={confirm}
          onChange={e => setConfirm(e.target.value)}
          className="w-full border border-gray-200 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-400"
        />
      </div>

      {error && <div className="text-xs text-red-600">{error}</div>}
      {msg && <div className="text-xs text-green-600">{msg}</div>}

      <button
        className="btn-primary text-sm py-2 w-full"
        disabled={!canSubmit || saving}
        onClick={handleSave}
      >
        {saving ? 'Saving…' : 'Change Password'}
      </button>
    </div>
  )
}

function DangerZone({ sessionCount, archerName }: { sessionCount: number | null; archerName: string }) {
  const navigate = useNavigate()
  const { logout } = useAuth()

  const [open, setOpen] = useState(false)
  const [password, setPassword] = useState('')
  const [confirmText, setConfirmText] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')

  const canDelete = password.length > 0 && confirmText === 'DELETE'

  const handleDelete = async () => {
    setBusy(true); setError('')
    try {
      await deleteArcherAccount(password)
      logout()
      navigate('/welcome')
    } catch (err) {
      setError(errorDetail(err, 'Could not delete account'))
      setBusy(false)
    }
  }

  return (
    <div className="card p-4 space-y-3 border-2 border-red-200 bg-red-50">
      <label className="text-xs font-semibold text-red-600 uppercase tracking-wide">Danger Zone</label>

      {!open && (
        <>
          <p className="text-xs text-red-700">
            Permanently delete {archerName}'s profile{sessionCount != null && sessionCount > 0
              ? ` and all ${sessionCount} practice session${sessionCount === 1 ? '' : 's'}`
              : ''}. This cannot be undone.
          </p>
          <button
            className="text-sm py-2 w-full rounded-xl font-semibold bg-red-600 text-white active:bg-red-700 transition-colors"
            onClick={() => setOpen(true)}
          >
            Delete Account
          </button>
        </>
      )}

      {open && (
        <div className="space-y-3">
          <p className="text-xs text-red-700 font-medium">
            This will permanently delete {archerName}'s profile{sessionCount != null && sessionCount > 0
              ? ` and all ${sessionCount} practice session${sessionCount === 1 ? '' : 's'}`
              : ''}. There is no way to undo this.
          </p>

          <div>
            <label className="text-xs text-red-700 mb-1 block">Enter your password</label>
            <input
              type="password"
              value={password}
              onChange={e => setPassword(e.target.value)}
              className="w-full border border-red-200 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-red-400"
            />
          </div>
          <div>
            <label className="text-xs text-red-700 mb-1 block">
              Type <span className="font-mono font-bold">DELETE</span> to confirm
            </label>
            <input
              value={confirmText}
              onChange={e => setConfirmText(e.target.value)}
              className="w-full border border-red-200 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-red-400"
            />
          </div>

          {error && <div className="text-xs text-red-600">{error}</div>}

          <div className="flex gap-2">
            <button
              className="btn-secondary text-sm py-2 flex-1"
              onClick={() => { setOpen(false); setPassword(''); setConfirmText(''); setError('') }}
              disabled={busy}
            >
              Cancel
            </button>
            <button
              className="text-sm py-2 flex-1 rounded-xl font-semibold bg-red-600 text-white
                         active:bg-red-700 disabled:opacity-50 transition-colors"
              disabled={!canDelete || busy}
              onClick={handleDelete}
            >
              {busy ? 'Deleting…' : 'Permanently Delete'}
            </button>
          </div>
        </div>
      )}
    </div>
  )
}
