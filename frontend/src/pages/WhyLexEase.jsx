import React, { useState, useEffect, useCallback } from 'react'
import { useNavigate } from 'react-router-dom'
import Navbar from '../components/Navbar'
import { checkHealth, getErrorMessage } from '../api/client'

function WhyLexEase() {
  const navigate = useNavigate()
  const [healthStatus, setHealthStatus] = useState(null)
  const [checkingHealth, setCheckingHealth] = useState(false)

  const verifyHealth = useCallback(async () => {
    setCheckingHealth(true)
    try {
      const data = await checkHealth()
      setHealthStatus(data)
    } catch (err) {
      setHealthStatus({ status: 'unreachable', error: getErrorMessage(err) })
    } finally {
      setCheckingHealth(false)
    }
  }, [])

  useEffect(() => {
    verifyHealth()
  }, [verifyHealth])

  return (
    <div className="min-h-screen bg-[#F8FAF8] text-[#16221C] flex flex-col font-sans selection:bg-[#176B4D]/15">
      <Navbar
        healthStatus={healthStatus}
        checkingHealth={checkingHealth}
        onCheckHealth={verifyHealth}
        hasDocument={false}
      />

      <main className="flex-1 max-w-4xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-10 sm:py-16 space-y-14 sm:space-y-18">
        {/* Header / Proposition */}
        <section className="space-y-4 text-left sm:text-center max-w-3xl mx-auto">
          <div className="inline-flex items-center gap-1.5 text-[11px] font-mono font-bold tracking-widest text-[#176B4D] uppercase">
            <span>&bull;</span>
            <span>Why LexEase</span>
            <span>&bull;</span>
          </div>

          <h1 className="text-3xl sm:text-4xl lg:text-[44px] font-extrabold text-[#16221C] tracking-tight leading-[1.18]">
            Legal agreements shouldn't require a{' '}
            <span className="font-serif italic font-normal text-[#176B4D] block sm:inline">
              law degree to understand.
            </span>
          </h1>

          <div className="pt-2 text-left max-w-xl mx-auto bg-white p-5 sm:p-6 rounded-2xl border border-[#DCE6E0] shadow-2xs">
            <p className="text-xs font-mono font-bold text-[#176B4D] uppercase tracking-wider mb-2.5">
              LexEase helps users:
            </p>
            <ul className="space-y-2 text-sm text-[#66736C]">
              <li className="flex items-start gap-2.5">
                <span className="text-[#176B4D] font-bold">&bull;</span>
                <span>Understand complex contract language without legal jargon</span>
              </li>
              <li className="flex items-start gap-2.5">
                <span className="text-[#176B4D] font-bold">&bull;</span>
                <span>Identify potentially risky clauses and hidden liabilities</span>
              </li>
              <li className="flex items-start gap-2.5">
                <span className="text-[#176B4D] font-bold">&bull;</span>
                <span>Review important obligations and key deadlines</span>
              </li>
              <li className="flex items-start gap-2.5">
                <span className="text-[#176B4D] font-bold">&bull;</span>
                <span>Ask questions about the actual agreement with grounded answers</span>
              </li>
            </ul>
          </div>
        </section>

        {/* How It Works Section */}
        <section className="p-6 sm:p-8 rounded-2xl bg-white border border-[#DCE6E0] shadow-2xs space-y-6">
          <div className="border-b border-[#F0F4F2] pb-4">
            <span className="text-[11px] font-mono font-bold text-[#176B4D] tracking-widest uppercase">
              Step-by-Step Workflow
            </span>
            <h2 className="text-xl sm:text-2xl font-extrabold text-[#16221C] mt-1 tracking-tight">
              HOW IT WORKS
            </h2>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-5">
            <div className="p-5 rounded-xl bg-[#F8FAF8] border border-[#DCE6E0] space-y-2">
              <div className="flex items-center justify-between">
                <h3 className="text-sm font-bold text-[#16221C]">01 — Upload</h3>
                <span className="font-mono text-[10px] font-bold text-[#176B4D] bg-[#E8F1EC] px-2 py-0.5 rounded">PDF</span>
              </div>
              <p className="text-xs text-[#66736C] leading-relaxed">
                Upload your PDF agreement.
              </p>
            </div>

            <div className="p-5 rounded-xl bg-[#F8FAF8] border border-[#DCE6E0] space-y-2">
              <div className="flex items-center justify-between">
                <h3 className="text-sm font-bold text-[#16221C]">02 — Analyze</h3>
                <span className="font-mono text-[10px] font-bold text-[#176B4D] bg-[#E8F1EC] px-2 py-0.5 rounded">ML Engine</span>
              </div>
              <p className="text-xs text-[#66736C] leading-relaxed">
                LexEase processes the agreement and evaluates its clauses.
              </p>
            </div>

            <div className="p-5 rounded-xl bg-[#F8FAF8] border border-[#DCE6E0] space-y-2">
              <div className="flex items-center justify-between">
                <h3 className="text-sm font-bold text-[#16221C]">03 — Understand</h3>
                <span className="font-mono text-[10px] font-bold text-[#176B4D] bg-[#E8F1EC] px-2 py-0.5 rounded">Synthesis</span>
              </div>
              <p className="text-xs text-[#66736C] leading-relaxed">
                Review plain-English explanations, summaries, and potential risks.
              </p>
            </div>

            <div className="p-5 rounded-xl bg-[#F8FAF8] border border-[#DCE6E0] space-y-2">
              <div className="flex items-center justify-between">
                <h3 className="text-sm font-bold text-[#16221C]">04 — Ask</h3>
                <span className="font-mono text-[10px] font-bold text-[#176B4D] bg-[#E8F1EC] px-2 py-0.5 rounded">Grounded Q&amp;A</span>
              </div>
              <p className="text-xs text-[#66736C] leading-relaxed">
                Ask questions grounded in the actual agreement.
              </p>
            </div>
          </div>

          {/* Action Trigger */}
          <div className="pt-4 text-center border-t border-[#F0F4F2]">
            <button
              type="button"
              onClick={() => navigate('/', { state: { triggerUpload: true } })}
              className="inline-flex items-center justify-center gap-2.5 px-7 py-3 rounded-xl text-sm font-bold text-white bg-[#176B4D] hover:bg-[#0F5139] active:scale-[0.98] transition-all duration-150 shadow-sm cursor-pointer"
            >
              <svg className="w-4 h-4 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="2.2">
                <path strokeLinecap="round" strokeLinejoin="round" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12" />
              </svg>
              <span>Analyze a Contract</span>
            </button>
          </div>
        </section>
      </main>

      {/* Footer */}
      <footer className="border-t border-[#DCE6E0] py-6 text-center text-xs text-[#8C9A92] bg-white/50">
        <p className="max-w-xl mx-auto leading-relaxed px-4">
          LexEase provides automated analysis for informational purposes only and is not a substitute for professional legal advice.
        </p>
      </footer>
    </div>
  )
}

export default WhyLexEase
