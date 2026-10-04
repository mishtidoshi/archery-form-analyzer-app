import type { ReactElement } from 'react'
import { BrowserRouter, Navigate, Route, Routes, useLocation } from 'react-router-dom'
import { AuthProvider, useAuth } from './auth'
import BottomNav from './components/BottomNav'
import WelcomePage from './pages/WelcomePage'
import ProfilesPage from './pages/ProfilesPage'
import HomePage from './pages/HomePage'
import SettingsPage from './pages/SettingsPage'
import UploadPage from './pages/UploadPage'
import SessionsPage from './pages/SessionsPage'
import SessionDetailPage from './pages/SessionDetailPage'
import TrendsPage from './pages/TrendsPage'
import CoachingPage from './pages/CoachingPage'
import CoachNotesPage from './pages/CoachNotesPage'

function RequireAuth({ children }: { children: ReactElement }) {
  const { token } = useAuth()
  if (!token) return <Navigate to="/welcome" replace />
  return children
}

const PUBLIC_PATHS = ['/welcome', '/profiles']

function Shell() {
  const { token } = useAuth()
  const location = useLocation()
  const showNav = Boolean(token) && !PUBLIC_PATHS.includes(location.pathname)

  return (
    <div className="relative max-w-md mx-auto h-screen bg-slate-50 overflow-hidden flex flex-col">
      <div className="flex-1 min-h-0 flex flex-col">
        <Routes>
          <Route path="/welcome" element={<WelcomePage />} />
          <Route path="/profiles" element={<ProfilesPage />} />
          <Route path="/" element={<RequireAuth><HomePage /></RequireAuth>} />
          <Route path="/settings" element={<RequireAuth><SettingsPage /></RequireAuth>} />
          <Route path="/upload" element={<RequireAuth><UploadPage /></RequireAuth>} />
          <Route path="/sessions" element={<RequireAuth><SessionsPage /></RequireAuth>} />
          <Route path="/sessions/:id" element={<RequireAuth><SessionDetailPage /></RequireAuth>} />
          <Route path="/trends" element={<RequireAuth><TrendsPage /></RequireAuth>} />
          <Route path="/coaching" element={<RequireAuth><CoachingPage /></RequireAuth>} />
          <Route path="/coach-notes" element={<RequireAuth><CoachNotesPage /></RequireAuth>} />
        </Routes>
      </div>
      {showNav && <BottomNav />}
    </div>
  )
}

export default function App() {
  return (
    <BrowserRouter>
      <AuthProvider>
        <Shell />
      </AuthProvider>
    </BrowserRouter>
  )
}
