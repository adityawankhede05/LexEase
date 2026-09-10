import React, { useState, useEffect, useCallback } from 'react'
import { useNavigate } from 'react-router-dom'
import Navbar from '../components/Navbar'
import { checkHealth, getErrorMessage } from '../api/client'

function About() {
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

      <main className="flex-1 max-w-3xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-10 sm:py-16 space-y-12">
        {/* Page Heading */}
        <div className="space-y-3 text-left">
          <div className="inline-flex items-center gap-1.5 text-[11px] font-mono font-bold tracking-widest text-[#176B4D] uppercase">
            <span>&bull;</span>
            <span>About LexEase</span>
            <span>&bull;</span>
          </div>
          <h1 className="text-3xl sm:text-4xl font-extrabold text-[#16221C] tracking-tight">
            About LexEase
          </h1>
          <p className="text-base sm:text-lg text-[#66736C] leading-relaxed">
            LexEase is a legal-document analysis application designed to make complex agreements easier to understand.
          </p>
        </div>

        {/* Structured Sections */}
        <div className="space-y-6">
          {/* Section 1: PURPOSE */}
          <div className="p-6 sm:p-7 rounded-2xl bg-white border border-[#DCE6E0] shadow-2xs space-y-2.5">
            <h2 className="text-xs font-mono font-bold text-[#176B4D] uppercase tracking-wider flex items-center gap-2">
              <span className="w-2 h-2 rounded-full bg-[#176B4D]"></span>
              <span>PURPOSE</span>
            </h2>
            <p className="text-sm text-[#16221C] leading-relaxed">
              LexEase helps users understand complex legal agreements by turning difficult legal language into clearer explanations and surfacing potentially important clauses.
            </p>
          </div>

          {/* Section 2: WHAT IT DOES */}
          <div className="p-6 sm:p-7 rounded-2xl bg-white border border-[#DCE6E0] shadow-2xs space-y-3.5">
            <h2 className="text-xs font-mono font-bold text-[#176B4D] uppercase tracking-wider flex items-center gap-2">
              <span className="w-2 h-2 rounded-full bg-[#176B4D]"></span>
              <span>WHAT IT DOES</span>
            </h2>
            <ul className="space-y-2 text-sm text-[#66736C]">
              <li className="flex items-start gap-2.5">
                <span className="text-[#176B4D] font-bold">&bull;</span>
                <span>Contract summarization</span>
              </li>
              <li className="flex items-start gap-2.5">
                <span className="text-[#176B4D] font-bold">&bull;</span>
                <span>Clause-level risk analysis</span>
              </li>
              <li className="flex items-start gap-2.5">
                <span className="text-[#176B4D] font-bold">&bull;</span>
                <span>Plain-English explanations</span>
              </li>
              <li className="flex items-start gap-2.5">
                <span className="text-[#176B4D] font-bold">&bull;</span>
                <span>Grounded document Q&amp;A</span>
              </li>
              <li className="flex items-start gap-2.5">
                <span className="text-[#176B4D] font-bold">&bull;</span>
                <span>Indian PII redaction</span>
              </li>
            </ul>
          </div>

          {/* Section 3: TECHNOLOGY */}
          <div className="p-6 sm:p-7 rounded-2xl bg-white border border-[#DCE6E0] shadow-2xs space-y-2.5">
            <h2 className="text-xs font-mono font-bold text-[#176B4D] uppercase tracking-wider flex items-center gap-2">
              <span className="w-2 h-2 rounded-full bg-[#176B4D]"></span>
              <span>TECHNOLOGY</span>
            </h2>
            <p className="text-sm text-[#66736C] leading-relaxed">
              LexEase combines document processing, machine-learning-based clause analysis, and contextual AI to analyze agreements.
            </p>
          </div>

          {/* Section 4: DISCLAIMER */}
          <div className="p-6 sm:p-7 rounded-2xl bg-[#F8FAF8] border border-[#DCE6E0] space-y-2">
            <h2 className="text-xs font-mono font-bold text-[#66736C] uppercase tracking-wider">
              DISCLAIMER
            </h2>
            <p className="text-xs text-[#8C9A92] leading-relaxed">
              LexEase provides automated analysis for informational purposes only and is not a substitute for professional legal advice.
            </p>
          </div>
        </div>

        {/* CTA to Analyze a Contract */}
        <div className="pt-4 text-center">
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
      </main>

      {/* Footer */}
      <footer className="border-t border-[#DCE6E0] py-6 text-center text-xs text-[#8C9A92] bg-white/50">
        <p className="max-w-xl mx-auto leading-relaxed px-4">
          LexEase &bull; AI-Powered Legal Document Simplification &amp; Risk Intelligence
        </p>
      </footer>
    </div>
  )
}

export default About
