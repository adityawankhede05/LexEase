import React from 'react'

const STAGES = [
  { id: 'uploading', label: 'Uploading document' },
  { id: 'reading', label: 'Reading document' },
  { id: 'identifying', label: 'Identifying clauses' },
  { id: 'preparing', label: 'Preparing analysis' },
  { id: 'ready', label: 'Analysis Ready' }
]

function DocumentProcessor({ file, stage, error, onRetry, onCancel }) {
  const getStepStatus = (stepId, currentStage) => {
    if (currentStage === 'ready') {
      return 'completed'
    }
    if (currentStage === 'error') {
      return 'future'
    }

    const stageOrder = ['uploading', 'reading', 'identifying', 'preparing', 'ready']
    const currentIndex = stageOrder.indexOf(currentStage)
    const stepIndex = stageOrder.indexOf(stepId)

    if (stepIndex < currentIndex) return 'completed'
    if (stepIndex === currentIndex) return 'active'
    return 'future'
  }

  const formatFileSize = (bytes) => {
    if (!bytes) return '0 Bytes'
    const k = 1024
    const sizes = ['Bytes', 'KB', 'MB']
    const i = Math.floor(Math.log(bytes) / Math.log(k))
    return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i]
  }

  return (
    <div
      className="w-full max-w-md mx-auto p-5 sm:p-6 rounded-2xl bg-[#111821] border border-[#293542] shadow-xl space-y-6 text-center animate-entrance-intake"
      role="status"
      aria-live="polite"
    >
      {/* Loading Header */}
      <div className="space-y-1">
        <h3 className="text-base sm:text-lg font-semibold text-[#F1F2F3]">
          {stage === 'error'
            ? 'Processing Interrupted'
            : stage === 'ready'
            ? 'Analysis Complete'
            : 'Preparing your document'}
        </h3>
        <p className="text-xs text-[#A4AEB9]">
          {stage === 'error'
            ? 'Document processing couldn\'t be completed.'
            : stage === 'ready'
            ? 'Your document is ready for review.'
            : 'LexEase is reading your agreement and preparing it for review.'}
        </p>
      </div>

      {/* Selected File Details Box */}
      {file && (
        <div className="flex items-center gap-3 p-3 rounded-xl bg-[#151E28] border border-[#293542] text-left">
          <div className="w-9 h-9 rounded bg-[#6E91B5]/10 border border-[#6E91B5]/20 text-[#6E91B5] flex items-center justify-center text-lg shrink-0">
            📄
          </div>
          <div className="min-w-0 flex-1">
            <span
              className="text-xs sm:text-sm font-semibold text-[#F1F2F3] truncate block"
              title={file.name}
            >
              {file.name}
            </span>
            <span className="text-[10px] text-[#A4AEB9] block mt-0.5 font-mono">
              PDF Document • {formatFileSize(file.size)}
            </span>
          </div>
        </div>
      )}

      {/* Steps List */}
      {stage !== 'error' && (
        <div className="space-y-4 pt-2 text-left">
          {STAGES.map((step, idx) => {
            const status = getStepStatus(step.id, stage)

            let icon = null
            let labelColor = 'text-[#74808D]'
            let connectorColor = 'bg-[#293542]'

            if (status === 'completed') {
              icon = (
                <span className="w-5 h-5 rounded-full bg-[#55B18A]/10 border border-[#55B18A]/30 text-[#55B18A] flex items-center justify-center text-xs font-bold font-sans">
                  ✓
                </span>
              )
              labelColor = 'text-[#A4AEB9]'
              connectorColor = 'bg-[#55B18A]'
            } else if (status === 'active') {
              icon = (
                <span className="relative flex h-5 w-5 items-center justify-center">
                  <span className="animate-ping absolute inline-flex h-3.5 w-3.5 rounded-full bg-[#6E91B5] opacity-60"></span>
                  <span className="relative inline-flex rounded-full h-2 w-2 bg-[#6E91B5]"></span>
                </span>
              )
              labelColor = 'text-[#F1F2F3] font-medium'
              connectorColor = 'bg-[#6E91B5]'
            } else {
              icon = (
                <span className="w-5 h-5 rounded-full bg-[#151E28] border border-[#293542] text-[#74808D] flex items-center justify-center text-[10px] font-sans">
                  ○
                </span>
              )
              labelColor = 'text-[#74808D]'
              connectorColor = 'bg-[#293542]'
            }

            return (
              <div key={step.id} className="relative flex items-start gap-4">
                {/* Connector Line */}
                {idx < STAGES.length - 1 && (
                  <div
                    className={`absolute left-2.5 top-5 w-0.5 h-8 -ml-[1px] ${connectorColor} transition-colors duration-500`}
                  />
                )}

                {/* Step Marker */}
                <div className="z-10 flex items-center justify-center shrink-0">
                  {icon}
                </div>

                {/* Step Metadata & Loading Bar */}
                <div className="flex-1 py-0.5 space-y-1">
                  <div className="flex items-center justify-between">
                    <span className={`text-xs sm:text-sm transition-colors duration-500 ${labelColor}`}>
                      {step.label}
                    </span>
                    {status === 'active' && (
                      <span className="text-[10px] text-[#6E91B5] font-mono animate-pulse uppercase tracking-wider">
                        Processing
                      </span>
                    )}
                  </div>

                  {status === 'active' && (
                    <div className="h-0.5 w-24 bg-[#293542] rounded-full overflow-hidden">
                      <div className="h-full bg-[#6E91B5] rounded-full animate-progress-bar"></div>
                    </div>
                  )}
                </div>
              </div>
            )
          })}
        </div>
      )}

      {/* Error state wrapper */}
      {stage === 'error' && (
        <div className="p-4 rounded-xl bg-[#D16B73]/10 border border-[#D16B73]/30 text-left space-y-3">
          <div className="flex items-start gap-2.5">
            <span className="text-lg text-[#D16B73] shrink-0 mt-0.5">⚠️</span>
            <div>
              <h4 className="text-xs sm:text-sm font-semibold text-[#D16B73]">
                Processing Failed
              </h4>
              <p className="text-xs text-[#A4AEB9] mt-1 break-words leading-relaxed">
                {error}
              </p>
            </div>
          </div>
          <div className="flex items-center gap-3 pt-1">
            <button
              type="button"
              onClick={onRetry}
              className="px-3.5 py-1.5 text-xs font-semibold rounded bg-[#D16B73] hover:bg-[#C05A62] text-white transition-colors focus:outline-none focus:ring-1 focus:ring-[#D16B73]"
            >
              Try again
            </button>
            <button
              type="button"
              onClick={onCancel}
              className="px-3.5 py-1.5 text-xs font-semibold rounded bg-transparent hover:bg-legal-surface border border-legal-border text-[#A4AEB9] hover:text-[#F1F2F3] transition-colors focus:outline-none"
            >
              Cancel
            </button>
          </div>
        </div>
      )}

      {/* Footer hint */}
      {stage !== 'error' && stage !== 'ready' && (
        <div className="pt-2 text-[10px] text-[#74808D]">
          Processing speed varies depending on the document length & segment count.
        </div>
      )}
    </div>
  )
}

export default DocumentProcessor
