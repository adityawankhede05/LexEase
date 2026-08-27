import React, { useState, useRef } from 'react'

function FileUpload({ onFileUpload, isUploading, uploadError }) {
  const [isDragOver, setIsDragOver] = useState(false)
  const [localError, setLocalError] = useState(null)
  const fileInputRef = useRef(null)

  const handleDragOver = (e) => {
    e.preventDefault()
    e.stopPropagation()
    setIsDragOver(true)
  }

  const handleDragLeave = (e) => {
    e.preventDefault()
    e.stopPropagation()
    setIsDragOver(false)
  }

  const validateAndProcessFile = (file) => {
    setLocalError(null)
    if (!file) return

    if (!file.name.toLowerCase().endsWith('.pdf') && file.type !== 'application/pdf') {
      setLocalError('Please select a valid PDF document (.pdf).')
      return
    }

    if (file.size === 0) {
      setLocalError('The selected file is empty. Please select a valid PDF.')
      return
    }

    onFileUpload(file)
  }

  const handleDrop = (e) => {
    e.preventDefault()
    e.stopPropagation()
    setIsDragOver(false)

    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      validateAndProcessFile(e.dataTransfer.files[0])
    }
  }

  const handleFileInputChange = (e) => {
    if (e.target.files && e.target.files.length > 0) {
      validateAndProcessFile(e.target.files[0])
    }
  }

  const errorMessage = uploadError || localError

  return (
    <div className="w-full max-w-2xl mx-auto space-y-12">
      {/* ================= Document Intake Area ================= */}
      <div className="space-y-3 animate-entrance-intake">
        <div
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
          onClick={() => !isUploading && fileInputRef.current?.click()}
          className={`relative border rounded-2xl p-8 sm:p-11 text-center transition-all duration-300 cursor-pointer ${
            isDragOver
              ? 'border-legal-accent bg-legal-secondary -translate-y-0.5 shadow-lg'
              : 'border-legal-border bg-legal-surface hover:border-legal-textMuted'
          } ${isUploading ? 'pointer-events-none opacity-80' : ''}`}
        >
          <input
            ref={fileInputRef}
            type="file"
            accept=".pdf,application/pdf"
            onChange={handleFileInputChange}
            className="hidden"
            disabled={isUploading}
          />

          {isUploading ? (
            <div className="flex flex-col items-center justify-center space-y-3.5 py-6">
              <div className="w-8 h-8 border-2 border-legal-border border-t-legal-accent rounded-full animate-spin"></div>
              <div className="space-y-1">
                <p className="text-sm font-medium text-legal-text">
                  Processing legal document...
                </p>
                <p className="text-xs text-legal-textSec">
                  Extracting text, segmenting clauses, and masking sensitive identifiers
                </p>
              </div>
            </div>
          ) : (
            <div className="flex flex-col items-center justify-center space-y-4">
              {/* Document Sheet Motif */}
              <div className="w-16 h-20 rounded border border-legal-border bg-legal-surface flex flex-col justify-between p-2 shadow-inner transition-transform duration-300 group-hover:scale-105">
                <div className="flex items-center justify-between border-b border-legal-border/60 pb-1">
                  <span className="text-[8px] font-mono text-legal-accent font-semibold tracking-wider uppercase">PDF</span>
                  <div className="w-1.5 h-1.5 rounded-full bg-legal-accent/60"></div>
                </div>
                <div className="space-y-1">
                  <div className="h-0.5 w-full bg-legal-border/80 rounded"></div>
                  <div className="h-0.5 w-4/5 bg-legal-border/80 rounded"></div>
                  <div className="h-0.5 w-3/5 bg-legal-border/80 rounded"></div>
                </div>
                <div className="h-0.5 w-2/5 bg-legal-accent/40 rounded"></div>
              </div>

              {/* Title & Instructions */}
              <div className="space-y-1.5">
                <span className="text-[11px] font-semibold text-legal-accentLight uppercase tracking-wider block">
                  Document Intake
                </span>
                <h2 className="text-base sm:text-lg font-semibold text-legal-text">
                  Bring your agreement into LexEase.
                </h2>
                <p className="text-xs text-[#A4AEB9]">
                  {isDragOver ? 'Drop your document here' : 'Drop a PDF anywhere here, or'}
                </p>
              </div>

              {/* Primary Action Button */}
              <div>
                <button
                  type="button"
                  onClick={(e) => {
                    e.stopPropagation()
                    fileInputRef.current?.click()
                  }}
                  className="px-5 py-2.5 text-xs font-semibold rounded-lg text-[#060A0E] bg-[#5E84AC] hover:bg-[#8EABC8] border border-transparent shadow-[0_4px_12px_rgba(94,132,172,0.18)] hover:shadow-[0_6px_16px_rgba(94,132,172,0.28)] hover:-translate-y-[1px] transition-all duration-200 ease-out focus:outline-none focus:ring-2 focus:ring-[#8EABC8] focus:ring-offset-2 focus:ring-offset-legal-bg"
                >
                  Select document
                </button>
              </div>

              {/* Format & Privacy notes */}
              <div className="pt-2 text-[11px] space-y-0.5">
                <p className="text-[#74808D]">PDF · supported format</p>
                <p className="text-[#8EABC8]">Automatic Indian PII protection (Aadhaar, PAN, phone, email)</p>
              </div>
            </div>
          )}
        </div>

        {/* Error Alert */}
        {errorMessage && (
          <div className="p-3.5 rounded-xl bg-[#1C1216] border border-legal-danger/40 text-legal-danger text-xs flex items-start gap-2.5">
            <svg
              className="w-4 h-4 shrink-0 mt-0.5 text-legal-danger"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
              strokeWidth="2"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"
              />
            </svg>
            <div className="flex-1">
              <p className="font-medium text-legal-text">Unable to process document</p>
              <p className="text-legal-danger mt-0.5">{errorMessage}</p>
            </div>
          </div>
        )}
      </div>

      {/* ================= Connected Three-Stage Workflow ================= */}
      <div className="pt-6 border-t border-legal-border/60">
        {/* Subtle Horizontal Connecting Workflow Line on Desktop */}
        <div className="hidden md:block w-full h-px bg-legal-border mb-6 animate-entrance-workflow-line"></div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-8 text-left">
          {/* Stage 01: Understand */}
          <div className="group space-y-2.5 animate-entrance-step-1">
            <div className="flex items-center gap-3">
              <span className="font-mono text-xs font-bold text-legal-textMuted group-hover:text-legal-accent transition-colors">
                01
              </span>
              <span className="text-xs font-semibold uppercase tracking-wider text-legal-text group-hover:text-legal-accent transition-colors">
                Understand
              </span>
            </div>
            <div className="w-8 h-px bg-legal-border group-hover:w-16 group-hover:bg-legal-accent transition-all duration-300"></div>
            <p className="text-xs text-legal-textSec leading-relaxed group-hover:translate-x-1 transition-transform duration-300">
              Get a clear plain-English summary of your agreement.
            </p>
          </div>

          {/* Stage 02: Review */}
          <div className="group space-y-2.5 animate-entrance-step-2">
            <div className="flex items-center gap-3">
              <span className="font-mono text-xs font-bold text-legal-textMuted group-hover:text-legal-accent transition-colors">
                02
              </span>
              <span className="text-xs font-semibold uppercase tracking-wider text-legal-text group-hover:text-legal-accent transition-colors">
                Review
              </span>
            </div>
            <div className="w-8 h-px bg-legal-border group-hover:w-16 group-hover:bg-legal-accent transition-all duration-300"></div>
            <p className="text-xs text-legal-textSec leading-relaxed group-hover:translate-x-1 transition-transform duration-300">
              Identify clauses that may deserve closer attention.
            </p>
          </div>

          {/* Stage 03: Ask */}
          <div className="group space-y-2.5 animate-entrance-step-3">
            <div className="flex items-center gap-3">
              <span className="font-mono text-xs font-bold text-legal-textMuted group-hover:text-legal-accent transition-colors">
                03
              </span>
              <span className="text-xs font-semibold uppercase tracking-wider text-legal-text group-hover:text-legal-accent transition-colors">
                Ask
              </span>
            </div>
            <div className="w-8 h-px bg-legal-border group-hover:w-16 group-hover:bg-legal-accent transition-all duration-300"></div>
            <p className="text-xs text-legal-textSec leading-relaxed group-hover:translate-x-1 transition-transform duration-300">
              Ask questions grounded in your document.
            </p>
          </div>
        </div>
      </div>

      {/* ================= Trust & Disclaimer Note ================= */}
      <div className="pt-2 text-center animate-entrance-disclaimer">
        <p className="text-[11px] text-legal-textMuted leading-relaxed">
          LexEase provides informational assistance and is not a substitute for professional legal advice.
        </p>
      </div>
    </div>
  )
}

export default FileUpload
