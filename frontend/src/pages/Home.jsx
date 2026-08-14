import React from 'react'
import { useNavigate } from 'react-router-dom'
import Card from '../components/Card'

function Home() {
  const navigate = useNavigate()

  return (
    <div className="space-y-20 py-4 sm:py-10">
      {/* Hero Section */}
      <div className="flex flex-col items-center text-center space-y-6 max-w-4xl mx-auto py-8">
        <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full text-xs font-semibold bg-[#2563EB]/10 border border-[#2563EB]/25 text-[#D4AF37]">
          <span>✨</span> Legal-Tech SaaS Platform
        </div>
        
        <h1 className="text-4xl sm:text-6xl font-extrabold tracking-tight text-[#F8FAFC]">
          Lex<span className="text-[#D4AF37]">Ease</span>
        </h1>
        
        <h2 className="text-xl sm:text-3xl font-bold text-[#94A3B8] tracking-tight">
          AI-Powered Legal Document Analysis
        </h2>
        
        <p className="text-base sm:text-lg text-[#94A3B8] max-w-2xl leading-relaxed">
          Translate dense legal agreements into plain English, highlight liabilities, and review contracts in seconds with state-of-the-art AI analysis.
        </p>

        <div className="pt-4">
          <button
            onClick={() => navigate('/upload')}
            className="px-8 py-3.5 rounded-lg bg-[#2563EB] hover:bg-[#2563EB]/90 text-white font-bold tracking-wide shadow-lg shadow-[#2563EB]/20 hover:shadow-[#2563EB]/30 transition-all hover:scale-[1.02] active:scale-95 flex items-center gap-3 border border-[#2563EB]"
          >
            Get Started - Analyze Document
            <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M14 5l7 7m0 0l-7 7m7-7H3" />
            </svg>
          </button>
        </div>
      </div>

      {/* Features Section */}
      <div className="space-y-8">
        <div className="text-center space-y-2">
          <h3 className="text-2xl sm:text-3xl font-bold text-[#F8FAFC]">Platform Capabilities</h3>
          <p className="text-sm text-[#94A3B8] max-w-md mx-auto">
            Advanced features designed to streamline legal document management and risk shielding.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <Card hoverable className="h-full">
            <div className="space-y-4">
              <div className="w-12 h-12 rounded-lg bg-[#2563EB]/10 border border-[#2563EB]/20 flex items-center justify-center text-2xl text-[#2563EB]">
                ⚖️
              </div>
              <h4 className="text-lg font-bold text-[#F8FAFC]">Document Simplification</h4>
              <p className="text-sm text-[#94A3B8] leading-relaxed">
                Breaks down impenetrable legal jargon into plain, actionable clauses that anyone can read and understand without legal training.
              </p>
            </div>
          </Card>

          <Card hoverable className="h-full">
            <div className="space-y-4">
              <div className="w-12 h-12 rounded-lg bg-red-500/10 border border-red-500/20 flex items-center justify-center text-2xl text-red-400">
                ⚠️
              </div>
              <h4 className="text-lg font-bold text-[#F8FAFC]">Risk Detection</h4>
              <p className="text-sm text-[#94A3B8] leading-relaxed">
                Flags unilateral indemnities, hidden costs, jurisdiction traps, and unfavourable clauses to guide liability decisions before sign-off.
              </p>
            </div>
          </Card>

          <Card hoverable className="h-full">
            <div className="space-y-4">
              <div className="w-12 h-12 rounded-lg bg-[#D4AF37]/10 border border-[#D4AF37]/20 flex items-center justify-center text-2xl text-[#D4AF37]">
                💡
              </div>
              <h4 className="text-lg font-bold text-[#F8FAFC]">AI Insights</h4>
              <p className="text-sm text-[#94A3B8] leading-relaxed">
                Delivers remedial suggestions and recommended contract revisions to negotiate safer agreements based on industry-standard clauses.
              </p>
            </div>
          </Card>
        </div>
      </div>

      {/* How It Works Section */}
      <div className="space-y-10 py-4">
        <div className="text-center space-y-2">
          <h3 className="text-2xl sm:text-3xl font-bold text-[#F8FAFC]">How It Works</h3>
          <p className="text-sm text-[#94A3B8]">Review contracts inside our secure sandboxed interface in three steps.</p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-8 max-w-5xl mx-auto relative">
          {/* Connector Line (visible on desktop) */}
          <div className="hidden md:block absolute top-12 left-[15%] right-[15%] h-0.5 bg-gradient-to-r from-slate-800 via-[#2563EB]/40 to-slate-800 -z-10"></div>

          {/* Step 1 */}
          <div className="flex flex-col items-center text-center space-y-4">
            <div className="w-12 h-12 rounded-full bg-[#0F172A] border-2 border-slate-700 flex items-center justify-center text-[#D4AF37] font-bold text-lg shadow-md">
              01
            </div>
            <h5 className="font-bold text-[#F8FAFC]">Upload Document</h5>
            <p className="text-xs text-[#94A3B8] max-w-[220px] leading-relaxed">
              Drag and drop any contract, lease, or NDA document in PDF format to get started.
            </p>
          </div>

          {/* Step 2 */}
          <div className="flex flex-col items-center text-center space-y-4">
            <div className="w-12 h-12 rounded-full bg-[#0F172A] border-2 border-[#2563EB]/70 flex items-center justify-center text-[#2563EB] font-bold text-lg shadow-md shadow-[#2563EB]/5">
              02
            </div>
            <h5 className="font-bold text-[#F8FAFC]">AI Processing</h5>
            <p className="text-xs text-[#94A3B8] max-w-[220px] leading-relaxed">
              Our models flag risks, score exposure, and translate terminology.
            </p>
          </div>

          {/* Step 3 */}
          <div className="flex flex-col items-center text-center space-y-4">
            <div className="w-12 h-12 rounded-full bg-[#0F172A] border-2 border-slate-700 flex items-center justify-center text-[#D4AF37] font-bold text-lg shadow-md">
              03
            </div>
            <h5 className="font-bold text-[#F8FAFC]">Review Dashboard</h5>
            <p className="text-xs text-[#94A3B8] max-w-[220px] leading-relaxed">
              Inspect side-by-side clause comparisons and overall liability gauges.
            </p>
          </div>
        </div>
      </div>

      {/* Final CTA Section */}
      <div className="bg-gradient-to-br from-[#1E293B] to-[#1E293B]/70 border border-slate-800 rounded-2xl p-8 sm:p-12 text-center max-w-4xl mx-auto shadow-xl relative overflow-hidden space-y-6">
        <div className="absolute top-0 right-0 w-24 h-24 bg-[#D4AF37]/5 rounded-bl-full border-b border-l border-slate-800"></div>
        <div className="absolute bottom-0 left-0 w-24 h-24 bg-[#2563EB]/5 rounded-tr-full border-t border-r border-slate-800"></div>

        <h3 className="text-2xl sm:text-3xl font-extrabold text-[#F8FAFC]">
          Ready to simplify your contracts?
        </h3>
        <p className="text-sm sm:text-base text-[#94A3B8] max-w-xl mx-auto leading-relaxed">
          Ensure peace of mind before signing. Run a comprehensive risk identification report instantly.
        </p>
        <div className="pt-2 flex justify-center">
          <button
            onClick={() => navigate('/upload')}
            className="px-6 py-3 rounded-lg bg-[#2563EB] hover:bg-[#2563EB]/90 text-white font-bold tracking-wide transition-all shadow hover:scale-[1.02] border border-[#2563EB]"
          >
            Upload a Contract Now
          </button>
        </div>
      </div>
    </div>
  )
}

export default Home
