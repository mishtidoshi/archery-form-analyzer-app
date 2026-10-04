import { useNavigate } from 'react-router-dom'

export default function WelcomePage() {
  const navigate = useNavigate()

  return (
    <div className="page bg-blue-600 flex flex-col items-center justify-center text-center px-8">
      <div className="text-6xl mb-4">🏹</div>
      <h1 className="text-3xl font-bold text-white leading-tight">Archery Form Analyzer</h1>
      <p className="text-blue-200 text-sm mt-3 max-w-xs">
        Quantitative biomechanical feedback on every arrow — from your phone.
      </p>

      <button
        onClick={() => navigate('/profiles')}
        className="mt-10 bg-white text-blue-700 font-semibold px-8 py-3 rounded-xl shadow-md
                   active:scale-95 transition-transform"
      >
        Get Started
      </button>
    </div>
  )
}
