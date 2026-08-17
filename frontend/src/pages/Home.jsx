import React, { useState, useEffect, useCallback, useRef } from 'react'
import {
  checkHealth,
  uploadDocument,
  summarizeDocument,
  analyzeClauses,
  getErrorMessage,
} from '../api/client'
import Navbar from '../components/Navbar'
import FileUpload from '../components/FileUpload'
import DocumentSummary from '../components/DocumentSummary'
import ClauseRiskAnalysis from '../components/ClauseRiskAnalysis'
import DocumentQA from '../components/DocumentQA'
import DocumentProcessor from '../components/DocumentProcessor'

function Home() {
  // Backend Health State
  const [healthStatus, setHealthStatus] = useState(null)
  const [checkingHealth, setCheckingHealth] = useState(false)

  // Document & Workflow State
  const [documentData, setDocumentData] = useState(null)
  const [isUploading, setIsUploading] = useState(false)
  const [uploadError, setUploadError] = useState(null)
  const [selectedFile, setSelectedFile] = useState(null)
  const [processingStage, setProcessingStage] = useState('idle') // 'idle' | 'uploading' | 'reading' | 'identifying' | 'preparing' | 'ready' | 'error'

  // Summary State
  const [summaryData, setSummaryData] = useState(null)
  const [isSummarizing, setIsSummarizing] = useState(false)
  const [summaryError, setSummaryError] = useState(null)

  // Clause Analysis State
  const [clauseAnalysisData, setClauseAnalysisData] = useState(null)
  const [isAnalyzingClauses, setIsAnalyzingClauses] = useState(false)
  const [clauseAnalysisError, setClauseAnalysisError] = useState(null)

  // Active Navigation Tab
  const [activeTab, setActiveTab] = useState('summary') // 'summary' | 'risks' | 'qa'

  const headingRef = useRef(null)

  // Lightweight Mouse Ambient Light Handler (Zero React re-renders, max 5-10px shift)
  useEffect(() => {
    if (typeof window === 'undefined') return
    const prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches
    const isTouchDevice = window.matchMedia('(pointer: coarse)').matches

    if (prefersReducedMotion || isTouchDevice) return

    let rafId = null

    const handleMouseMove = (e) => {
      if (rafId) return
      rafId = requestAnimationFrame(() => {
        const x = e.clientX
        const y = e.clientY
        const w = window.innerWidth
        const h = window.innerHeight

        // Subtle 5–10px maximum shift (max 8px offset)
        const offsetX = ((x / w) - 0.5) * 16
        const offsetY = ((y / h) - 0.5) * 16

        document.documentElement.style.setProperty('--mouse-offset-x', `${offsetX.toFixed(1)}px`)
        document.documentElement.style.setProperty('--mouse-offset-y', `${offsetY.toFixed(1)}px`)

        rafId = null
      })
    }

    window.addEventListener('mousemove', handleMouseMove, { passive: true })
    return () => {
      window.removeEventListener('mousemove', handleMouseMove)
      if (rafId) cancelAnimationFrame(rafId)
    }
  }, [])

  // Subtle cursor-reactive effect on the hero heading
  useEffect(() => {
    const heading = headingRef.current
    if (!heading) return

    // Disable on touch/mobile and prefers-reduced-motion
    const prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches
    const isTouchDevice = window.matchMedia('(pointer: coarse)').matches
    if (prefersReducedMotion || isTouchDevice) return

    let rafId = null

    const handleMouseMove = (e) => {
      if (rafId) return
      rafId = requestAnimationFrame(() => {
        const rect = heading.getBoundingClientRect()
        const x = e.clientX - rect.left
        const pct = Math.max(0, Math.min(100, (x / rect.width) * 100))

        heading.style.setProperty('--accent-position', `${pct}%`)
        rafId = null
      })
    }

    const handleMouseEnter = () => {
      heading.classList.add('active')
    }

    const handleMouseLeave = () => {
      heading.classList.remove('active')
      heading.style.setProperty('--accent-position', '80%')
    }

    heading.addEventListener('mousemove', handleMouseMove, { passive: true })
    heading.addEventListener('mouseenter', handleMouseEnter)
    heading.addEventListener('mouseleave', handleMouseLeave)

    return () => {
      heading.removeEventListener('mousemove', handleMouseMove)
      heading.removeEventListener('mouseenter', handleMouseEnter)
      heading.removeEventListener('mouseleave', handleMouseLeave)
      if (rafId) cancelAnimationFrame(rafId)
    }
  }, [])

  // Health Check Handler
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

  // Summarize Handler
  const handleSummarize = useCallback(async (clauses) => {
    if (!clauses || clauses.length === 0) return
    setIsSummarizing(true)
    setSummaryError(null)

    try {
      const data = await summarizeDocument(clauses)
      setSummaryData(data)
    } catch (err) {
      setSummaryError(getErrorMessage(err))
    } finally {
      setIsSummarizing(false)
    }
  }, [])

  // Analyze Clauses Handler
  const handleAnalyzeClauses = useCallback(async (clauses) => {
    if (!clauses || clauses.length === 0) return
    setIsAnalyzingClauses(true)
    setClauseAnalysisError(null)

    try {
      const data = await analyzeClauses(clauses)
      setClauseAnalysisData(data)
    } catch (err) {
      setClauseAnalysisError(getErrorMessage(err))
    } finally {
      setIsAnalyzingClauses(false)
    }
  }, [])

  // Full Document Upload & Pipeline Trigger
  const handleFileUpload = async (file) => {
    setSelectedFile(file)
    setProcessingStage('uploading')
    setIsUploading(true)
    setUploadError(null)
    setSummaryData(null)
    setClauseAnalysisData(null)
    setSummaryError(null)
    setClauseAnalysisError(null)

    const readingTimer = setTimeout(() => {
      setProcessingStage('reading')
    }, 1800)

    try {
      // Step 1: Upload and pre-process PDF (Validation, PyMuPDF extraction, cleaning, segmentation, PII masking)
      const uploadResp = await uploadDocument(file)
      clearTimeout(readingTimer)

      // Step 2: Automatically trigger Document Summary and Clause Risk Analysis concurrently
      setProcessingStage('identifying')

      if (uploadResp.clauses && uploadResp.clauses.length > 0) {
        setIsSummarizing(true)
        setIsAnalyzingClauses(true)

        const preparingTimer = setTimeout(() => {
          setProcessingStage('preparing')
        }, 1500)

        const sumPromise = summarizeDocument(uploadResp.clauses)
          .then((res) => {
            setSummaryData(res)
            setIsSummarizing(false)
            return res
          })
          .catch((err) => {
            const errMsg = getErrorMessage(err)
            setSummaryError(errMsg)
            setIsSummarizing(false)
            throw new Error(errMsg)
          })

        const analyzePromise = analyzeClauses(uploadResp.clauses)
          .then((res) => {
            setClauseAnalysisData(res)
            setIsAnalyzingClauses(false)
            return res
          })
          .catch((err) => {
            const errMsg = getErrorMessage(err)
            setClauseAnalysisError(errMsg)
            setIsAnalyzingClauses(false)
            throw new Error(errMsg)
          })

        await Promise.all([sumPromise, analyzePromise])
        clearTimeout(preparingTimer)
      } else {
        setSummaryData({ summary: 'No clauses detected in this document.', key_points: [] })
        setClauseAnalysisData({ results: [] })
      }

      // Step 3: Transition to ready
      setProcessingStage('ready')

      // Short subtle confirmation transition (1.5 seconds)
      setTimeout(() => {
        setDocumentData(uploadResp)
        setActiveTab('summary')
        setIsUploading(false)
      }, 1500)

    } catch (err) {
      clearTimeout(readingTimer)
      setProcessingStage('error')
      setUploadError(getErrorMessage(err))
      setIsUploading(false)
    }
  }

  // Reset Document State to Upload New
  const handleResetDocument = () => {
    setDocumentData(null)
    setSummaryData(null)
    setClauseAnalysisData(null)
    setUploadError(null)
    setSummaryError(null)
    setClauseAnalysisError(null)
    setSelectedFile(null)
    setProcessingStage('idle')
    setActiveTab('summary')
  }

  // Jump from Q&A Source Clause Chip directly to Clause Risk Analysis Card
  const handleSelectClause = (clauseId) => {
    setActiveTab('risks')
    setTimeout(() => {
      const el = document.getElementById(`clause-${clauseId}`)
      if (el) {
        el.scrollIntoView({ behavior: 'smooth', block: 'center' })
        el.classList.add('ring-1', 'ring-legal-info')
        setTimeout(() => el.classList.remove('ring-1', 'ring-legal-info'), 2500)
      }
    }, 100)
  }

  return (
    <div className="min-h-screen legal-workspace-bg text-legal-text flex flex-col font-sans selection:bg-legal-accent/20">
      {/* Top Navigation */}
      <Navbar
        healthStatus={healthStatus}
        checkingHealth={checkingHealth}
        onCheckHealth={verifyHealth}
        hasDocument={Boolean(documentData)}
        onResetDocument={handleResetDocument}
      />

      {/* Main Content */}
      <main className="flex-1 max-w-6xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-12 sm:py-16">
        {!documentData ? (
          /* ================= Legal Workspace Landing State ================= */
          <div className="relative flex flex-col items-center justify-center space-y-10 sm:space-y-12 text-center">

            {/* Hero Section */}
            <div className="space-y-3.5 max-w-2xl mx-auto">
              <h1 ref={headingRef} className="hero-glow-text text-3xl sm:text-4xl md:text-5xl font-semibold tracking-tight animate-entrance-hero-title">
                Understand before you sign.
              </h1>
              <p className="text-sm sm:text-base text-[#A4AEB9] max-w-xl mx-auto leading-relaxed animate-entrance-hero-sub">
                Review complex agreements in plain language, identify potential risks, and ask questions grounded in your document.
              </p>
            </div>

            {/* Document Intake & Connected Workflow */}
            {processingStage === 'idle' ? (
              <FileUpload
                onFileUpload={handleFileUpload}
                isUploading={isUploading}
                uploadError={uploadError}
              />
            ) : (
              <DocumentProcessor
                file={selectedFile}
                stage={processingStage}
                error={uploadError}
                onRetry={() => handleFileUpload(selectedFile)}
                onCancel={handleResetDocument}
              />
            )}
          </div>
        ) : (
          /* ================= Document Analysis Dashboard ================= */
          <div className="space-y-6">
            {/* Document Context Header Banner */}
            <div className="p-5 rounded-xl bg-legal-surface border border-legal-border flex flex-wrap items-center justify-between gap-4">
              <div className="flex items-center gap-3 min-w-0">
                <div className="w-10 h-10 rounded-lg bg-legal-secondary border border-legal-border text-legal-info flex items-center justify-center shrink-0">
                  <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="1.75">
                    <path strokeLinecap="round" strokeLinejoin="round" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                  </svg>
                </div>
                <div className="min-w-0">
                  <h2 className="text-sm sm:text-base font-semibold text-legal-text truncate">
                    {documentData.filename}
                  </h2>
                  <div className="flex flex-wrap items-center gap-2 text-xs text-legal-textSec font-mono mt-0.5">
                    <span>{documentData.page_count} {documentData.page_count === 1 ? 'page' : 'pages'}</span>
                    <span>•</span>
                    <span>{documentData.clauses?.length || 0} clauses</span>
                    <span>•</span>
                    <span>{documentData.character_count?.toLocaleString()} chars</span>
                    {documentData.document_id && (
                      <>
                        <span>•</span>
                        <span className="text-legal-textMuted">
                          ID: {documentData.document_id.substring(0, 8)}...
                        </span>
                      </>
                    )}
                  </div>
                </div>
              </div>

              {/* Status Chips */}
              <div className="flex items-center gap-2">
                {isSummarizing || isAnalyzingClauses ? (
                  <div className="flex items-center gap-2 px-3 py-1 rounded-lg bg-legal-secondary border border-legal-border text-legal-warning text-xs font-medium">
                    <span className="w-1.5 h-1.5 rounded-full bg-legal-warning animate-pulse"></span>
                    <span>Analyzing Document...</span>
                  </div>
                ) : (
                  <div className="flex items-center gap-2 px-3 py-1 rounded-lg bg-legal-secondary border border-legal-border text-legal-success text-xs font-medium">
                    <span className="w-1.5 h-1.5 rounded-full bg-legal-success"></span>
                    <span>Analysis Ready</span>
                  </div>
                )}
              </div>
            </div>

            {/* Dashboard Tabs */}
            <div className="border-b border-legal-border">
              <nav className="flex space-x-2">
                <button
                  type="button"
                  onClick={() => setActiveTab('summary')}
                  className={`py-2 px-3.5 text-xs sm:text-sm font-medium rounded-t-xl transition-all border-b-2 flex items-center gap-2 ${
                    activeTab === 'summary'
                      ? 'border-legal-info text-legal-text bg-legal-surface'
                      : 'border-transparent text-legal-textSec hover:text-legal-text hover:bg-legal-surface/50'
                  }`}
                >
                  <span>Document Summary</span>
                  {isSummarizing && (
                    <span className="w-1.5 h-1.5 rounded-full bg-legal-info animate-pulse"></span>
                  )}
                </button>

                <button
                  type="button"
                  onClick={() => setActiveTab('risks')}
                  className={`py-2 px-3.5 text-xs sm:text-sm font-medium rounded-t-xl transition-all border-b-2 flex items-center gap-2 ${
                    activeTab === 'risks'
                      ? 'border-legal-info text-legal-text bg-legal-surface'
                      : 'border-transparent text-legal-textSec hover:text-legal-text hover:bg-legal-surface/50'
                  }`}
                >
                  <span>Clause Risk Analysis</span>
                  {clauseAnalysisData?.results?.length > 0 && (
                    <span className="px-1.5 py-0.2 rounded-md bg-legal-secondary text-[10px] text-legal-textSec font-mono border border-legal-border">
                      {clauseAnalysisData.results.length}
                    </span>
                  )}
                  {isAnalyzingClauses && (
                    <span className="w-1.5 h-1.5 rounded-full bg-legal-warning animate-pulse"></span>
                  )}
                </button>

                <button
                  type="button"
                  onClick={() => setActiveTab('qa')}
                  className={`py-2 px-3.5 text-xs sm:text-sm font-medium rounded-t-xl transition-all border-b-2 flex items-center gap-2 ${
                    activeTab === 'qa'
                      ? 'border-legal-info text-legal-text bg-legal-surface'
                      : 'border-transparent text-legal-textSec hover:text-legal-text hover:bg-legal-surface/50'
                  }`}
                >
                  <span>Ask Document (Q&A)</span>
                </button>
              </nav>
            </div>

            {/* Tab View Content */}
            <div className="pt-2">
              {activeTab === 'summary' && (
                <DocumentSummary
                  summaryData={summaryData}
                  isLoading={isSummarizing}
                  error={summaryError}
                  onRetry={() => handleSummarize(documentData.clauses)}
                />
              )}

              {activeTab === 'risks' && (
                <ClauseRiskAnalysis
                  analysisData={clauseAnalysisData}
                  originalClauses={documentData.clauses}
                  isLoading={isAnalyzingClauses}
                  error={clauseAnalysisError}
                  onRetry={() => handleAnalyzeClauses(documentData.clauses)}
                />
              )}

              {activeTab === 'qa' && (
                <DocumentQA
                  documentId={documentData.document_id}
                  filename={documentData.filename}
                  onSelectClause={handleSelectClause}
                />
              )}
            </div>
          </div>
        )}
      </main>

      {/* Footer */}
      <footer className="border-t border-legal-border py-6 text-center text-xs text-legal-textMuted">
        <p>LexEase &bull; Legal Document Simplification &bull; Sprint 8 AI Architecture</p>
      </footer>
    </div>
  )
}

export default Home
