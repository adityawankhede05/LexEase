import React, { useState, useEffect, useCallback, useRef } from 'react'
import { useLocation } from 'react-router-dom'
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
  const location = useLocation()

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

  // Automatically trigger native file picker if navigated with triggerUpload state
  useEffect(() => {
    if (location.state?.triggerUpload) {
      window.history.replaceState({}, document.title)
      setTimeout(() => {
        const fileInput = document.querySelector('input[type="file"]')
        if (fileInput) fileInput.click()
      }, 150)
    }
  }, [location.state])

  // Summarize Handler
  const handleSummarize = useCallback(async (clauses, docId = null) => {
    if (!clauses || clauses.length === 0) return
    setIsSummarizing(true)
    setSummaryError(null)

    try {
      const targetDocId = docId || documentData?.document_id
      const data = await summarizeDocument(clauses, targetDocId)
      setSummaryData(data)
    } catch (err) {
      setSummaryError(getErrorMessage(err))
    } finally {
      setIsSummarizing(false)
    }
  }, [documentData?.document_id])

  // Analyze Clauses Handler
  const handleAnalyzeClauses = useCallback(async (clauses, docId = null) => {
    if (!clauses || clauses.length === 0) return
    setIsAnalyzingClauses(true)
    setClauseAnalysisError(null)

    try {
      const targetDocId = docId || documentData?.document_id
      const data = await analyzeClauses(clauses, targetDocId)
      setClauseAnalysisData(data)
    } catch (err) {
      setClauseAnalysisError(getErrorMessage(err))
    } finally {
      setIsAnalyzingClauses(false)
    }
  }, [documentData?.document_id])

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

        const sumPromise = summarizeDocument(uploadResp.clauses, uploadResp.document_id)
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

        const analyzePromise = analyzeClauses(uploadResp.clauses, uploadResp.document_id)
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
    <div className="min-h-screen bg-[#F8FAF8] text-[#16221C] flex flex-col font-sans selection:bg-[#176B4D]/15">
      {/* Top Navigation */}
      <Navbar
        healthStatus={healthStatus}
        checkingHealth={checkingHealth}
        onCheckHealth={verifyHealth}
        hasDocument={Boolean(documentData)}
        onResetDocument={handleResetDocument}
      />

      {/* Main Content */}
      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 pt-0 pb-12 sm:pb-16">
        {!documentData ? (
          /* ================= Legal Workspace Landing State ================= */
          <div className="w-full">
            {processingStage === 'idle' ? (
              <FileUpload
                onFileUpload={handleFileUpload}
                isUploading={isUploading}
                uploadError={uploadError}
              />
            ) : (
              <div className="w-full py-4 sm:py-8 lg:py-10">
                <DocumentProcessor
                  file={selectedFile}
                  stage={processingStage}
                  error={uploadError}
                  onRetry={() => handleFileUpload(selectedFile)}
                  onCancel={handleResetDocument}
                />
              </div>
            )}
          </div>
        ) : (
          /* ================= Document Analysis Dashboard ================= */
          <div className="space-y-6">
            {/* Document Context Header Banner */}
            <div className="p-5 sm:p-6 rounded-2xl bg-white border border-[#DCE6E0] shadow-sm flex flex-col md:flex-row md:items-center md:justify-between gap-5">
              <div className="flex items-start sm:items-center gap-4 min-w-0">
                <div className="w-12 h-12 rounded-2xl bg-[#E8F1EC] border border-[#DCE6E0] text-[#176B4D] flex items-center justify-center text-xl shrink-0 shadow-2xs">
                  📄
                </div>
                <div className="min-w-0 space-y-1">
                  <div className="flex items-center gap-2.5 flex-wrap">
                    <h2 className="text-base sm:text-lg font-bold text-[#16221C] truncate tracking-tight">
                      {documentData.filename}
                    </h2>
                    <span className="px-2.5 py-0.5 rounded-full text-[10px] font-mono font-bold uppercase bg-[#E8F1EC] text-[#176B4D] border border-[#DCE6E0]">
                      Verified Context
                    </span>
                  </div>
                  <div className="flex flex-wrap items-center gap-x-3 gap-y-1 text-xs text-[#66736C]">
                    <span className="font-medium text-[#16221C]">
                      {documentData.page_count} {documentData.page_count === 1 ? 'page' : 'pages'}
                    </span>
                    <span className="text-[#DCE6E0]">&bull;</span>
                    <span className="font-medium text-[#16221C]">
                      {documentData.clauses?.length || 0} clauses detected
                    </span>
                    <span className="text-[#DCE6E0]">&bull;</span>
                    <span>
                      {documentData.character_count?.toLocaleString()} characters
                    </span>
                    {documentData.document_id && (
                      <>
                        <span className="text-[#DCE6E0]">&bull;</span>
                        <span className="text-[#8C9A92] font-mono text-[11px]">
                          ID: {documentData.document_id.substring(0, 8)}...
                        </span>
                      </>
                    )}
                  </div>
                </div>
              </div>

              {/* Status & Actions */}
              <div className="flex items-center gap-3 self-end md:self-center">
                {isSummarizing || isAnalyzingClauses ? (
                  <div className="flex items-center gap-2 px-3.5 py-1.5 rounded-xl bg-[#FEF3E2] border border-[#F8DEC0] text-[#D18A24] text-xs font-bold">
                    <span className="w-2 h-2 rounded-full bg-[#D18A24] animate-pulse"></span>
                    <span>Analyzing Document...</span>
                  </div>
                ) : (
                  <div className="flex items-center gap-2 px-3.5 py-1.5 rounded-xl bg-[#E8F1EC] border border-[#DCE6E0] text-[#176B4D] text-xs font-bold">
                    <span className="w-2 h-2 rounded-full bg-[#3E8E63]"></span>
                    <span>Analysis Complete</span>
                  </div>
                )}

                <button
                  type="button"
                  onClick={handleResetDocument}
                  className="px-3.5 py-1.5 text-xs font-semibold rounded-xl text-[#66736C] hover:text-[#16221C] bg-[#F3F7F5] border border-[#DCE6E0] hover:border-[#176B4D]/40 transition-all focus:outline-none"
                  title="Upload a different document"
                >
                  Change Document
                </button>
              </div>
            </div>

            {/* Dashboard Segmented Tabs */}
            <div className="border-b border-[#DCE6E0]">
              <nav className="flex space-x-2 sm:space-x-3 overflow-x-auto pb-px">
                <button
                  type="button"
                  onClick={() => setActiveTab('summary')}
                  className={`py-2.5 px-4 text-xs sm:text-sm font-bold rounded-t-xl transition-all border-b-2 flex items-center gap-2.5 cursor-pointer whitespace-nowrap ${
                    activeTab === 'summary'
                      ? 'border-[#176B4D] text-[#12372A] bg-white shadow-2xs'
                      : 'border-transparent text-[#66736C] hover:text-[#16221C] hover:bg-[#F3F7F5]'
                  }`}
                >
                  <span>📋</span>
                  <span>Document Summary</span>
                  {isSummarizing && (
                    <span className="w-1.5 h-1.5 rounded-full bg-[#176B4D] animate-pulse"></span>
                  )}
                </button>

                <button
                  type="button"
                  onClick={() => setActiveTab('risks')}
                  className={`py-2.5 px-4 text-xs sm:text-sm font-bold rounded-t-xl transition-all border-b-2 flex items-center gap-2.5 cursor-pointer whitespace-nowrap ${
                    activeTab === 'risks'
                      ? 'border-[#176B4D] text-[#12372A] bg-white shadow-2xs'
                      : 'border-transparent text-[#66736C] hover:text-[#16221C] hover:bg-[#F3F7F5]'
                  }`}
                >
                  <span>⚖️</span>
                  <span>Clause Risk Analysis</span>
                  {clauseAnalysisData?.results?.length > 0 && (
                    <span className="px-2 py-0.5 rounded-md bg-[#E8F1EC] text-[11px] text-[#176B4D] font-mono font-bold border border-[#DCE6E0]">
                      {clauseAnalysisData.results.length}
                    </span>
                  )}
                  {isAnalyzingClauses && (
                    <span className="w-1.5 h-1.5 rounded-full bg-[#D18A24] animate-pulse"></span>
                  )}
                </button>

                <button
                  type="button"
                  onClick={() => setActiveTab('qa')}
                  className={`py-2.5 px-4 text-xs sm:text-sm font-bold rounded-t-xl transition-all border-b-2 flex items-center gap-2.5 cursor-pointer whitespace-nowrap ${
                    activeTab === 'qa'
                      ? 'border-[#176B4D] text-[#12372A] bg-white shadow-2xs'
                      : 'border-transparent text-[#66736C] hover:text-[#16221C] hover:bg-[#F3F7F5]'
                  }`}
                >
                  <span>💬</span>
                  <span>Ask Document (Q&amp;A)</span>
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
      <footer className="border-t border-[#DCE6E0] py-6 text-center text-xs text-[#8C9A92] bg-white/50">
        <p>LexEase &bull; AI-Powered Legal Document Simplification &amp; Risk Intelligence</p>
      </footer>
    </div>
  )
}

export default Home
