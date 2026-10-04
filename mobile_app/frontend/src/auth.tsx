import { createContext, useContext, useState } from 'react'
import type { ReactNode } from 'react'
import { logout as apiLogout } from './api'

interface AuthState {
  token: string | null
  archerId: string | null
  archerName: string | null
}

interface AuthContextValue extends AuthState {
  login: (token: string, archerId: string, archerName: string) => void
  logout: () => void
  setName: (name: string) => void
}

const STORAGE_KEY = 'archer_auth'   // { token, archerId, archerName }

function loadStored(): AuthState {
  try {
    const raw = localStorage.getItem(STORAGE_KEY)
    if (!raw) return { token: null, archerId: null, archerName: null }
    const parsed = JSON.parse(raw)
    return {
      token: parsed.token ?? null,
      archerId: parsed.archerId ?? null,
      archerName: parsed.archerName ?? null,
    }
  } catch {
    return { token: null, archerId: null, archerName: null }
  }
}

const AuthContext = createContext<AuthContextValue | null>(null)

export function AuthProvider({ children }: { children: ReactNode }) {
  const [state, setState] = useState<AuthState>(loadStored)

  const login = (token: string, archerId: string, archerName: string) => {
    const next = { token, archerId, archerName }
    localStorage.setItem(STORAGE_KEY, JSON.stringify(next))
    // api.ts reads the bearer token straight out of localStorage under this key.
    localStorage.setItem('archer_token', token)
    setState(next)
  }

  const logout = () => {
    apiLogout()
    localStorage.removeItem(STORAGE_KEY)
    localStorage.removeItem('archer_token')
    setState({ token: null, archerId: null, archerName: null })
  }

  // Patches the display name only — used after a name edit, which doesn't touch the
  // token (unlike login/change-password, which issue a fresh one).
  const setName = (name: string) => {
    setState(prev => {
      const next = { ...prev, archerName: name }
      localStorage.setItem(STORAGE_KEY, JSON.stringify(next))
      return next
    })
  }

  return (
    <AuthContext.Provider value={{ ...state, login, logout, setName }}>
      {children}
    </AuthContext.Provider>
  )
}

export function useAuth(): AuthContextValue {
  const ctx = useContext(AuthContext)
  if (!ctx) throw new Error('useAuth must be used within an AuthProvider')
  return ctx
}
