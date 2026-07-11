import React, { useState } from 'react'
import client from '../api/client'

function Home() {
  const [healthStatus, setHealthStatus] = useState(null)
  const [loading, setLoading] = useState(false)

  const checkBackendHealth = async () => {
    setLoading(true)
    try {
      const response = await client.get('/health')
      setHealthStatus(response.data)
    } catch (err) {
      setHealthStatus({ status: 'unreachable', error: err.message })
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="flex flex-col items-center justify-center min-h-screen px-4 bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950">
      <div className="max-w-2xl w-full text-center space-y-8 p-8 rounded-2xl bg-slate-900/50 border border-slate-800 backdrop-blur-xl shadow-2xl">
        <div className="inline-flex items-center justify-center w-16 h-16 rounded-full bg-indigo-500/10 border border-indigo-500/30 text-indigo-400 text-3xl">
          ⚖️
        </div>
        
        <div className="space-y-3">
          <h1 className="text-4xl sm:text-5xl font-extrabold tracking-tight bg-gradient-to-r from-indigo-400 via-purple-400 to-pink-400 bg-clip-text text-transparent">
            LexEase
          </h1>
          <p className="text-slate-400 text-lg">
            Legal Document Simplification & Clause Risk Analysis
          </p>
        </div>

        <div className="h-px bg-gradient-to-r from-transparent via-slate-800 to-transparent"></div>

        <div className="space-y-4">
          <p className="text-slate-500 text-sm">
            Workspace Scaffolding Successfully Initialized.
          </p>

          <div className="flex flex-col items-center gap-4">
            <button
              onClick={checkBackendHealth}
              disabled={loading}
              className="px-6 py-2.5 rounded-lg bg-indigo-600 hover:bg-indigo-500 text-white font-medium shadow-lg hover:shadow-indigo-500/20 transition-all disabled:opacity-50"
            >
              {loading ? 'Checking...' : 'Verify Backend Connection'}
            </button>

            {healthStatus && (
              <pre className="p-4 rounded-lg bg-slate-950/80 border border-slate-800 text-left text-xs font-mono max-w-full overflow-x-auto w-full text-emerald-400">
                {JSON.stringify(healthStatus, null, 2)}
              </pre>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}

export default Home
