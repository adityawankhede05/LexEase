import React from 'react'

function Navbar({ healthStatus, checkingHealth, onCheckHealth, hasDocument, onResetDocument }) {
  const isHealthy = healthStatus?.status === 'healthy'

  const handleUploadClick = () => {
    if (hasDocument) {
      onResetDocument()
    } else {
      const fileInput = document.querySelector('input[type="file"]')
      if (fileInput) {
        fileInput.click()
      }
    }
  }

  return (
    <header className="sticky top-0 z-50 bg-legal-bg/95 backdrop-blur-sm border-b border-legal-border animate-entrance-header">
      <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 h-20 flex items-center justify-between">
        {/* Brand with Silver Sheen Wordmark - Clickable to return Home */}
        <button
          type="button"
          onClick={onResetDocument}
          className="flex items-center space-x-2.5 hover:opacity-90 transition-opacity focus:outline-none focus:ring-1 focus:ring-legal-border rounded-md p-1 group text-left"
          title="Return to home page"
        >
          <div className="flex items-center justify-center w-9 h-9 rounded-lg bg-legal-surface border border-legal-border group-hover:border-legal-textMuted transition-colors">
            <svg
              className="w-5 h-5 text-legal-accent"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
            >
              <path d="m16 16 3-8 3 8c-.87.65-1.92 1-3 1s-2.13-.35-3-1Z" />
              <path d="m2 16 3-8 3 8c-.87.65-1.92 1-3 1s-2.13-.35-3-1Z" />
              <path d="M7 21h10" />
              <path d="M12 3v18" />
              <path d="M3 7h2c2 0 5-1 7-2 2 1 5 2 7 2h2" />
            </svg>
          </div>
          <span className="lexease-wordmark text-[22px] sm:text-[24px] font-semibold tracking-tight cursor-pointer select-none">
            LexEase
          </span>
        </button>

        {/* Right Actions / Navigation */}
        <div className="flex items-center space-x-4 sm:space-x-6">
          {/* Home navigation link */}
          <button
            type="button"
            onClick={onResetDocument}
            className={`text-sm font-medium transition-colors focus:outline-none focus:text-legal-text ${
              !hasDocument
                ? 'text-legal-text border-b-2 border-legal-accent py-1'
                : 'text-legal-textSec hover:text-legal-text py-1 border-b-2 border-transparent'
            }`}
          >
            Home
          </button>

          {/* Upload Button */}
          <button
            type="button"
            onClick={handleUploadClick}
            className="inline-flex items-center px-4 py-2 text-xs font-semibold rounded-lg text-legal-text bg-legal-surface border border-legal-border hover:border-legal-accent hover:bg-legal-secondary transition-colors focus:outline-none focus:ring-1 focus:ring-legal-border"
          >
            {hasDocument ? (
              <>
                <svg
                  className="w-3.5 h-3.5 mr-1.5 text-legal-textSec"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth="2"
                    d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12"
                  />
                </svg>
                Upload New
              </>
            ) : (
              'Upload Document'
            )}
          </button>

          {/* Backend Status Indicator */}
          <button
            type="button"
            onClick={onCheckHealth}
            title="Click to check backend connection status"
            className="flex items-center space-x-2 px-2.5 py-1.5 rounded-lg text-xs font-medium bg-legal-surface border border-legal-border hover:bg-legal-secondary transition-colors focus:outline-none focus:ring-1 focus:ring-legal-border"
          >
            <span
              className={`w-1.5 h-1.5 rounded-full ${
                checkingHealth
                  ? 'bg-legal-warning animate-pulse'
                  : isHealthy
                  ? 'bg-legal-success'
                  : 'bg-legal-danger'
              }`}
            />
            <span className="text-legal-textSec text-[11px] hidden sm:inline">
              {checkingHealth
                ? 'Connecting...'
                : isHealthy
                ? 'Backend Ready'
                : 'Backend Offline'}
            </span>
          </button>
        </div>
      </div>
    </header>
  )
}

export default Navbar
