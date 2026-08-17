import React from 'react'

function DocumentSummary({ summaryData, isLoading, error, onRetry }) {
  if (isLoading) {
    return (
      <div className="p-6 rounded-2xl bg-legal-surface/50 border border-legal-border space-y-6 animate-pulse">
        <div className="flex items-center justify-between">
          <div className="h-6 w-36 bg-legal-elevated rounded-md"></div>
          <div className="h-6 w-24 bg-legal-elevated rounded-full"></div>
        </div>
        <div className="space-y-2.5">
          <div className="h-4 bg-legal-elevated rounded w-full"></div>
          <div className="h-4 bg-legal-elevated rounded w-5/6"></div>
          <div className="h-4 bg-legal-elevated rounded w-4/6"></div>
        </div>
        <div className="space-y-3 pt-4 border-t border-legal-border/80">
          <div className="h-5 w-32 bg-legal-elevated rounded"></div>
          <div className="h-4 bg-legal-elevated rounded w-11/12"></div>
          <div className="h-4 bg-legal-elevated rounded w-4/5"></div>
          <div className="h-4 bg-legal-elevated rounded w-3/4"></div>
        </div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="p-6 rounded-2xl bg-legal-surface/50 border border-legal-danger/30 space-y-4 text-center">
        <div className="w-12 h-12 rounded-full bg-legal-danger/10 border border-legal-danger/20 text-legal-danger flex items-center justify-center mx-auto text-xl">
          ⚠️
        </div>
        <div className="space-y-1">
          <h3 className="text-base font-semibold text-legal-danger">
            Summarization Failed
          </h3>
          <p className="text-xs text-legal-textSec max-w-md mx-auto">{error}</p>
        </div>
        {onRetry && (
          <button
            onClick={onRetry}
            className="px-4 py-1.5 text-xs font-medium rounded-lg bg-legal-surface hover:bg-legal-secondary text-legal-text border border-legal-border transition-colors"
          >
            Retry Summarization
          </button>
        )}
      </div>
    )
  }

  if (!summaryData) {
    return (
      <div className="p-8 rounded-2xl bg-legal-surface/40 border border-legal-border/80 text-center text-legal-textMuted">
        No summary generated yet.
      </div>
    )
  }

  const { summary, key_points, document_type } = summaryData

  return (
    <div className="space-y-6">
      {/* Executive Summary Card */}
      <div className="p-5 rounded-2xl bg-legal-surface border border-legal-border shadow-xl backdrop-blur-sm space-y-6">
        {/* Header */}
        <div className="flex flex-wrap items-center justify-between gap-3 border-b border-legal-border/80 pb-4">
          <div className="flex items-center gap-2.5">
            <span className="text-2xl">📋</span>
            <div>
              <h2 className="text-lg sm:text-xl font-bold text-legal-text">
                Executive Document Summary
              </h2>
              <p className="text-xs text-legal-textSec">
                Plain-English distillation synthesized by AI
              </p>
            </div>
          </div>

          {document_type && (
            <span className="inline-flex items-center px-3 py-1 rounded-full text-xs font-bold bg-legal-info/10 text-legal-info border border-legal-info/30 uppercase tracking-wider text-[10px]">
              🏷️ {document_type}
            </span>
          )}
        </div>

        {/* Plain English Summary */}
        <div className="space-y-2">
          <h3 className="text-xs font-semibold text-legal-textMuted uppercase tracking-wider">
            Overview
          </h3>
          <p className="text-legal-textSec leading-relaxed text-sm whitespace-pre-line">
            {summary}
          </p>
        </div>

        {/* Key Takeaways */}
        {key_points && key_points.length > 0 && (
          <div className="space-y-3 pt-4 border-t border-legal-border/80">
            <h3 className="text-xs font-semibold text-legal-textMuted uppercase tracking-wider flex items-center gap-1.5">
              <span>📌</span> Key Points & Obligations
            </h3>
            <ul className="space-y-2.5">
              {key_points.map((point, index) => (
                <li
                  key={index}
                  className="flex items-start gap-3 p-4 rounded-xl bg-legal-secondary border border-legal-border text-sm text-legal-textSec"
                >
                  <span className="inline-flex items-center justify-center w-5 h-5 rounded-full bg-legal-info/10 text-legal-info border border-legal-info/20 text-xs font-bold shrink-0 mt-0.5">
                    {index + 1}
                  </span>
                  <span className="leading-snug">{point}</span>
                </li>
              ))}
            </ul>
          </div>
        )}
      </div>
    </div>
  )
}

export default DocumentSummary
