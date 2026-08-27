import React, { useState, useMemo } from 'react'

function ClauseRiskAnalysis({ analysisData, originalClauses = [], isLoading, error, onRetry }) {
  const [filterRisk, setFilterRisk] = useState('all')
  const [searchQuery, setSearchQuery] = useState('')
  const [expandedClauses, setExpandedClauses] = useState({})

  // Map original clause text by clause_id for rapid lookup
  const clauseTextMap = useMemo(() => {
    const map = new Map()
    if (Array.isArray(originalClauses)) {
      originalClauses.forEach((c) => {
        map.set(c.clause_id, c.text)
      })
    }
    return map
  }, [originalClauses])

  // Count risk levels
  const stats = useMemo(() => {
    const results = analysisData?.results || []
    return {
      total: results.length,
      high: results.filter((r) => r.risk_level === 'high').length,
      medium: results.filter((r) => r.risk_level === 'medium').length,
      low: results.filter((r) => r.risk_level === 'low').length,
    }
  }, [analysisData])

  // Filtered clause list
  const filteredResults = useMemo(() => {
    const results = analysisData?.results || []
    return results.filter((item) => {
      // Risk filter
      if (filterRisk !== 'all' && item.risk_level !== filterRisk) {
        return false
      }
      // Search query
      if (searchQuery.trim()) {
        const q = searchQuery.toLowerCase()
        const text = clauseTextMap.get(item.clause_id) || ''
        const matchesId = item.clause_id.toLowerCase().includes(q)
        const matchesNum = (item.clause_number || '').toLowerCase().includes(q)
        const matchesText = text.toLowerCase().includes(q)
        const matchesExp = item.explanation.toLowerCase().includes(q)
        const matchesRec = item.recommendation.toLowerCase().includes(q)
        return matchesId || matchesNum || matchesText || matchesExp || matchesRec
      }
      return true
    })
  }, [analysisData, filterRisk, searchQuery, clauseTextMap])

  const toggleExpand = (clauseId) => {
    setExpandedClauses((prev) => ({
      ...prev,
      [clauseId]: !prev[clauseId],
    }))
  }

  if (isLoading) {
    return (
      <div className="space-y-6 animate-pulse">
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
          {[1, 2, 3, 4].map((i) => (
            <div key={i} className="h-20 bg-[#111720]/60 rounded-xl border border-[#29313A]"></div>
          ))}
        </div>
        <div className="space-y-4">
          {[1, 2, 3].map((i) => (
            <div key={i} className="h-44 bg-[#111720]/50 rounded-2xl border border-[#29313A]"></div>
          ))}
        </div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="p-6 rounded-2xl bg-[#111720]/50 border border-[#A9676D]/30 space-y-4 text-center">
        <div className="w-12 h-12 rounded-full bg-[#A9676D]/10 border border-[#A9676D]/20 text-[#A9676D] flex items-center justify-center mx-auto text-xl">
          ⚠️
        </div>
        <div className="space-y-1">
          <h3 className="text-base font-semibold text-[#A9676D]">
            Clause Risk Analysis Failed
          </h3>
          <p className="text-xs text-[#A9A69D] max-w-md mx-auto">{error}</p>
        </div>
        {onRetry && (
          <button
            onClick={onRetry}
            className="px-4 py-1.5 text-xs font-medium rounded-lg bg-[#111720] hover:bg-[#171E27] text-[#E8E3D8] border border-[#29313A] transition-colors"
          >
            Retry Analysis
          </button>
        )}
      </div>
    )
  }

  if (!analysisData || !analysisData.results || analysisData.results.length === 0) {
    return (
      <div className="p-8 rounded-2xl bg-[#111720]/40 border border-[#29313A]/80 text-center text-[#64758A]">
        No clause analysis available.
      </div>
    )
  }

  return (
    <div className="space-y-6">
      {/* Risk Metrics Summary Cards */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
        {/* Total */}
        <div
          onClick={() => setFilterRisk('all')}
          className={`p-4 rounded-xl border transition-all cursor-pointer ${
            filterRisk === 'all'
              ? 'bg-legal-surface border-legal-border shadow-md ring-1 ring-legal-border/30'
              : 'bg-legal-surface border-legal-border/50 hover:border-legal-border'
          }`}
        >
          <span className="text-xs font-semibold text-legal-textMuted uppercase tracking-wider">
            Total Clauses
          </span>
          <div className="text-2xl font-bold text-legal-text mt-1">
            {stats.total}
          </div>
        </div>

        {/* High Risk */}
        <div
          onClick={() => setFilterRisk(filterRisk === 'high' ? 'all' : 'high')}
          className={`p-4 rounded-xl border transition-all cursor-pointer ${
            filterRisk === 'high'
              ? 'bg-legal-danger/10 border-legal-danger/60 shadow-md ring-1 ring-legal-danger/30'
              : 'bg-legal-surface border-legal-border/50 hover:border-legal-danger/30'
          }`}
        >
          <span className="text-xs font-semibold text-legal-danger uppercase tracking-wider flex items-center gap-1.5">
            <span className="w-2 h-2 rounded-full bg-legal-danger"></span> High Risk
          </span>
          <div className="text-2xl font-bold text-legal-text mt-1">
            {stats.high}
          </div>
        </div>

        {/* Medium Risk */}
        <div
          onClick={() => setFilterRisk(filterRisk === 'medium' ? 'all' : 'medium')}
          className={`p-4 rounded-xl border transition-all cursor-pointer ${
            filterRisk === 'medium'
              ? 'bg-legal-warning/10 border-legal-warning/60 shadow-md ring-1 ring-legal-warning/30'
              : 'bg-legal-surface border-legal-border/50 hover:border-legal-warning/30'
          }`}
        >
          <span className="text-xs font-semibold text-legal-warning uppercase tracking-wider flex items-center gap-1.5">
            <span className="w-2 h-2 rounded-full bg-legal-warning"></span> Medium Risk
          </span>
          <div className="text-2xl font-bold text-legal-text mt-1">
            {stats.medium}
          </div>
        </div>

        {/* Low Risk */}
        <div
          onClick={() => setFilterRisk(filterRisk === 'low' ? 'all' : 'low')}
          className={`p-4 rounded-xl border transition-all cursor-pointer ${
            filterRisk === 'low'
              ? 'bg-legal-success/10 border-legal-success/60 shadow-md ring-1 ring-legal-success/30'
              : 'bg-legal-surface border-legal-border/50 hover:border-legal-success/30'
          }`}
        >
          <span className="text-xs font-semibold text-legal-success uppercase tracking-wider flex items-center gap-1.5">
            <span className="w-2 h-2 rounded-full bg-legal-success"></span> Low Risk
          </span>
          <div className="text-2xl font-bold text-legal-text mt-1">
            {stats.low}
          </div>
        </div>
      </div>

      {/* Filter Toolbar */}
      <div className="flex flex-col sm:flex-row items-stretch sm:items-center justify-between gap-3 bg-legal-surface/40 p-3 rounded-xl border border-legal-border">
        <div className="flex items-center gap-1.5 overflow-x-auto pb-1 sm:pb-0">
          {['all', 'high', 'medium', 'low'].map((level) => {
            const isActive = filterRisk === level
            let activeClass = 'border-legal-accent/50 text-legal-accent bg-legal-accent/10 shadow-sm'
            if (level === 'high') activeClass = 'border-legal-danger/50 text-legal-danger bg-legal-danger/10 shadow-sm'
            if (level === 'medium') activeClass = 'border-legal-warning/50 text-legal-warning bg-legal-warning/10 shadow-sm'
            if (level === 'low') activeClass = 'border-legal-success/50 text-legal-success bg-legal-success/10 shadow-sm'

            return (
              <button
                key={level}
                onClick={() => setFilterRisk(level)}
                className={`px-3 py-1 text-xs font-medium rounded-lg capitalize border transition-all whitespace-nowrap ${
                  isActive
                    ? activeClass
                    : 'border-transparent bg-legal-secondary text-legal-textMuted hover:bg-legal-surface hover:text-legal-text'
                }`}
              >
                {level === 'all' ? 'All Clauses' : `${level} Risk`}
              </button>
            )
          })}
        </div>

        {/* Search */}
        <div className="relative flex-1 sm:max-w-xs">
          <input
            type="text"
            placeholder="Search clauses..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="w-full pl-8 pr-3 py-1.5 text-xs rounded-lg bg-legal-surface/90 border border-legal-border text-legal-text placeholder-legal-textMuted focus:outline-none focus:border-legal-info/75"
          />
          <span className="absolute left-2.5 top-2.5 text-legal-info text-xs">🔍</span>
          {searchQuery && (
            <button
              onClick={() => setSearchQuery('')}
              className="absolute right-2.5 top-1.5 text-legal-textMuted hover:text-legal-text text-xs"
            >
              ✕
            </button>
          )}
        </div>
      </div>

      {/* Clause Cards List */}
      {filteredResults.length === 0 ? (
        <div className="p-8 rounded-2xl bg-legal-surface/40 border border-legal-border/80 text-center text-legal-textMuted space-y-2">
          <p>No clauses match your active filters.</p>
          <button
            onClick={() => {
              setFilterRisk('all')
              setSearchQuery('')
            }}
            className="text-xs text-legal-action hover:underline font-semibold"
          >
            Clear filters
          </button>
        </div>
      ) : (
        <div className="space-y-6">
          {filteredResults.map((result, idx) => {
            const rawText = clauseTextMap.get(result.clause_id)
            const isExpanded = expandedClauses[result.clause_id]
            const confidencePercent = Math.round((result.confidence ?? 0) * 100)

            const displayClauseNum = result.clause_number
              ? String(result.clause_number).padStart(2, '0')
              : ''

            const cleanIdLabel = result.clause_id
              ? result.clause_id
                  .split('_')
                  .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
                  .join(' ')
              : ''

            const riskTheme = {
              high: {
                badge: 'bg-[#D16B73]/10 text-[#D16B73] border-[#D16B73]/30',
                label: 'High Risk',
                icon: '🚨',
              },
              medium: {
                badge: 'bg-[#D49A4A]/10 text-[#D49A4A] border-[#D49A4A]/30',
                label: 'Medium Risk',
                icon: '⚠️',
              },
              low: {
                badge: 'bg-[#55B18A]/10 text-[#55B18A] border-[#55B18A]/30',
                label: 'Low Risk',
                icon: '✅',
              },
            }[result.risk_level] || {
              badge: 'bg-legal-textMuted/10 text-legal-textMuted border-legal-border',
              label: result.risk_level,
              icon: 'ℹ️',
            }

            return (
              <div
                key={result.clause_id}
                id={`clause-${result.clause_id}`}
                className="reveal-card group p-5 rounded-2xl bg-legal-surface border border-legal-border hover:bg-legal-secondary hover:border-legal-accent shadow-lg hover:shadow-[0_8px_30px_rgba(94,132,172,0.08)] hover:-translate-y-0.5 transition-all duration-200 ease-out space-y-5"
                style={{ animationDelay: `${Math.min(idx * 60, 400)}ms` }}
              >
                {/* Header */}
                <div className="flex flex-wrap items-center justify-between gap-2 border-b border-legal-border/60 pb-3.5">
                  <div className="flex items-center gap-3">
                    <span className="text-sm font-bold text-legal-accentLight group-hover:text-legal-text transition-colors duration-200 tracking-wide uppercase">
                      {displayClauseNum ? `Clause ${displayClauseNum}` : cleanIdLabel}
                    </span>
                    {displayClauseNum && cleanIdLabel && (
                      <span className="text-xs text-legal-textSec font-medium">
                        {cleanIdLabel}
                      </span>
                    )}
                  </div>

                  <div className="flex items-center gap-2">
                    {/* Confidence */}
                    <span className="text-xs text-legal-textMuted bg-legal-surface px-2 py-0.5 rounded border border-legal-border group-hover:text-legal-textSec group-hover:border-legal-border/80 transition-colors duration-200">
                      {confidencePercent}% confidence
                    </span>
                    {/* Risk Badge */}
                    <span
                      className={`inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-[10px] font-bold uppercase tracking-wider border ${riskTheme.badge}`}
                    >
                      <span>{riskTheme.icon}</span>
                      {riskTheme.label}
                    </span>
                  </div>
                </div>

                {/* Original Clause Text */}
                {rawText && (
                  <div className="space-y-1.5">
                    <div className="flex items-center justify-between">
                      <span className="text-[11px] font-semibold text-legal-textMuted uppercase tracking-wider">
                        Clause Text (PII Protected)
                      </span>
                      {rawText.length > 300 && (
                        <button
                          onClick={() => toggleExpand(result.clause_id)}
                          className="text-xs text-legal-action hover:underline font-semibold"
                        >
                          {isExpanded ? 'Show less' : 'Show full text'}
                        </button>
                      )}
                    </div>
                    <p
                      className={`p-5 bg-legal-bg border border-legal-border hover:border-legal-accent/30 focus-within:border-legal-accent/30 text-[15px] text-legal-text leading-[1.65] rounded-xl font-sans transition-colors duration-200 ${
                        rawText.length > 300 && !isExpanded ? 'line-clamp-3' : ''
                      }`}
                    >
                      {rawText}
                    </p>
                  </div>
                )}

                {/* Plain-English Legal Risk Explanation */}
                <div className="space-y-1.5">
                  <span className="text-[11px] font-semibold text-legal-accent uppercase tracking-wider flex items-center gap-1.5">
                    <span className="text-legal-accent">🔍</span> Legal Analysis & Risk Explanation
                  </span>
                  <p className="text-sm text-legal-textSec leading-relaxed pl-1 whitespace-pre-line">
                    {result.explanation}
                  </p>
                </div>

                {/* Actionable Advice / Recommendation */}
                {result.recommendation && (
                  <div className="p-5 rounded-xl bg-legal-secondary border border-legal-border border-l-2 border-l-legal-warning text-sm text-legal-text flex items-start gap-3 transition-colors duration-200">
                    <span className="text-base leading-none shrink-0 mt-0.5 text-legal-warning">💡</span>
                    <div>
                      <span className="font-semibold text-legal-warning">Recommendation before signing: </span>
                      <span>{result.recommendation}</span>
                    </div>
                  </div>
                )}
              </div>
            )
          })}
        </div>
      )}
    </div>
  )
}

export default ClauseRiskAnalysis
