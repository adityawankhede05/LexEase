import React, { useState, useMemo, useEffect, useRef } from 'react'

// Lightweight Animated Counter Hook for smooth metrics count-up (0 -> target)
function useCountUp(target, duration = 850) {
  const [count, setCount] = useState(0)
  const prevTargetRef = useRef(0)
  const hasAnimatedRef = useRef(false)

  useEffect(() => {
    if (target === undefined || target === null) return
    const endVal = Number(target)
    if (isNaN(endVal)) return

    // Respect user's motion preferences
    if (typeof window !== 'undefined' && window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
      setCount(endVal)
      prevTargetRef.current = endVal
      return
    }

    const startVal = hasAnimatedRef.current ? prevTargetRef.current : 0
    if (hasAnimatedRef.current && startVal === endVal) {
      return // Do not re-animate if the value hasn't changed (e.g. filter toggle)
    }

    hasAnimatedRef.current = true
    let start = null
    let rafId = null

    const step = (timestamp) => {
      if (!start) start = timestamp
      const elapsed = timestamp - start
      const progress = Math.min(elapsed / duration, 1)
      // Ease out: smooth deceleration into the target
      const easeOut = 1 - Math.pow(1 - progress, 2.5)
      const current = Math.round(startVal + (endVal - startVal) * easeOut)
      setCount(current)

      if (progress < 1) {
        rafId = requestAnimationFrame(step)
      } else {
        prevTargetRef.current = endVal
      }
    }

    rafId = requestAnimationFrame(step)
    return () => {
      if (rafId) cancelAnimationFrame(rafId)
    }
  }, [target, duration])

  return count
}

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

  // Overall liability score (0-100) and exposure rating
  const { riskScore, riskRating, gaugeColor } = useMemo(() => {
    const total = stats.total
    if (total === 0) {
      return { riskScore: 0, riskRating: 'Low', gaugeColor: '#3E8E63' }
    }
    const score = Math.min(
      100,
      Math.round((stats.high * 90 + stats.medium * 50 + stats.low * 15) / total)
    )
    let rating = 'Low'
    let color = '#3E8E63'
    if (stats.high > 0 || score >= 70) {
      rating = 'High'
      color = '#C94A4A'
    } else if (stats.medium > 0 || score >= 40) {
      rating = 'Medium'
      color = '#D18A24'
    }
    return { riskScore: score, riskRating: rating, gaugeColor: color }
  }, [stats])

  // Animated metric counters
  const animatedScore = useCountUp(riskScore, 1000)
  const animatedTotal = useCountUp(stats.total, 800)
  const animatedHigh = useCountUp(stats.high, 800)
  const animatedMedium = useCountUp(stats.medium, 800)
  const animatedLow = useCountUp(stats.low, 800)

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
        <div className="p-6 rounded-2xl bg-white border border-[#DCE6E0] space-y-4">
          <div className="h-24 bg-[#F3F7F5] rounded-xl"></div>
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 pt-2">
            {[1, 2, 3, 4].map((i) => (
              <div key={i} className="h-16 bg-[#F3F7F5] rounded-xl"></div>
            ))}
          </div>
        </div>
        <div className="space-y-4">
          {[1, 2, 3].map((i) => (
            <div key={i} className="h-44 bg-white rounded-2xl border border-[#DCE6E0]"></div>
          ))}
        </div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="p-6 sm:p-8 rounded-2xl bg-white border border-[#F5C2C2] space-y-4 text-center shadow-sm">
        <div className="w-12 h-12 rounded-full bg-[#FFF2F2] border border-[#F5C2C2] text-[#C94A4A] flex items-center justify-center mx-auto text-xl">
          ⚠️
        </div>
        <div className="space-y-1">
          <h3 className="text-base font-bold text-[#C94A4A]">
            Clause Risk Analysis Failed
          </h3>
          <p className="text-xs text-[#66736C] max-w-md mx-auto">{error}</p>
        </div>
        {onRetry && (
          <button
            onClick={onRetry}
            className="px-4 py-1.5 text-xs font-semibold rounded-xl bg-white hover:bg-[#F3F7F5] text-[#16221C] border border-[#DCE6E0] transition-colors shadow-2xs"
          >
            Retry Analysis
          </button>
        )}
      </div>
    )
  }

  if (!analysisData || !analysisData.results || analysisData.results.length === 0) {
    return (
      <div className="p-8 rounded-2xl bg-white border border-[#DCE6E0] text-center text-[#8C9A92] shadow-sm">
        No clause analysis available.
      </div>
    )
  }

  return (
    <div className="space-y-6">
      {/* Overall Liability Assessment & Risk Metrics Banner */}
      <div className="p-5 sm:p-6 rounded-2xl bg-white border border-[#DCE6E0] shadow-sm space-y-6">
        <div className="flex flex-col lg:flex-row items-center gap-6">
          {/* Circular SVG Gauge Widget */}
          <div className="flex flex-col sm:flex-row items-center gap-5 p-4 rounded-xl bg-[#F8FAF8] border border-[#DCE6E0] shrink-0 w-full lg:w-auto">
            <div className="relative flex items-center justify-center w-28 h-28">
              <svg className="w-full h-full -rotate-90" viewBox="0 0 120 120">
                {/* Background Ring */}
                <circle
                  cx="60"
                  cy="60"
                  r="48"
                  stroke="#E8F1EC"
                  strokeWidth="8"
                  fill="transparent"
                />
                {/* Dynamic Active Arc */}
                <circle
                  cx="60"
                  cy="60"
                  r="48"
                  stroke={gaugeColor}
                  strokeWidth="8"
                  fill="transparent"
                  strokeDasharray="301.6"
                  strokeDashoffset={301.6 - (301.6 * riskScore) / 100}
                  strokeLinecap="round"
                  className="transition-all duration-1000 ease-out"
                />
              </svg>
              <div className="absolute text-center flex flex-col items-center justify-center">
                <span className="text-2xl font-extrabold text-[#16221C] tracking-tight font-mono">
                  {animatedScore}
                </span>
                <span className="text-[10px] text-[#8C9A92] font-mono">
                  / 100
                </span>
              </div>
            </div>

            <div className="text-center sm:text-left space-y-1.5">
              <span className="text-[11px] font-bold uppercase tracking-wider text-[#66736C] block">
                Overall Liability
              </span>
              <div>
                <span
                  className={`inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-bold uppercase tracking-wide border ${
                    riskRating === 'High'
                      ? 'bg-[#FFF2F2] text-[#C94A4A] border-[#F5C2C2]'
                      : riskRating === 'Medium'
                      ? 'bg-[#FEF3E2] text-[#D18A24] border-[#F8DEC0]'
                      : 'bg-[#E8F1EC] text-[#3E8E63] border-[#C8DECE]'
                  }`}
                >
                  <span
                    className={`w-1.5 h-1.5 rounded-full ${
                      riskRating === 'High'
                        ? 'bg-[#C94A4A] animate-pulse'
                        : riskRating === 'Medium'
                        ? 'bg-[#D18A24]'
                        : 'bg-[#3E8E63]'
                    }`}
                  ></span>
                  {riskRating} Exposure
                </span>
              </div>
              <p className="text-[11px] text-[#66736C] max-w-[210px] leading-relaxed pt-0.5">
                Weighted risk index synthesized from clause severity.
              </p>
            </div>
          </div>

          {/* Interactive Animated Metric Cards */}
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 flex-1 w-full">
            {/* Total */}
            <div
              onClick={() => setFilterRisk('all')}
              className={`p-4 rounded-xl border transition-all cursor-pointer ${
                filterRisk === 'all'
                  ? 'bg-[#E8F1EC] border-[#176B4D] shadow-xs ring-1 ring-[#176B4D]/20'
                  : 'bg-[#F8FAF8] border-[#DCE6E0] hover:border-[#176B4D]/40 hover:bg-white'
              }`}
            >
              <span className="text-[11px] font-bold text-[#66736C] uppercase tracking-wider block">
                Total Clauses
              </span>
              <div className="text-2xl font-extrabold text-[#16221C] mt-1 font-mono">
                {animatedTotal}
              </div>
              <span className="text-[10px] text-[#8C9A92]">Click to view all</span>
            </div>

            {/* High Risk */}
            <div
              onClick={() => setFilterRisk(filterRisk === 'high' ? 'all' : 'high')}
              className={`p-4 rounded-xl border transition-all cursor-pointer ${
                filterRisk === 'high'
                  ? 'bg-[#FFF2F2] border-[#C94A4A] shadow-xs ring-1 ring-[#C94A4A]/30'
                  : 'bg-[#F8FAF8] border-[#DCE6E0] hover:border-[#C94A4A]/40 hover:bg-[#FFF2F2]/50'
              }`}
            >
              <span className="text-[11px] font-bold text-[#C94A4A] uppercase tracking-wider flex items-center gap-1.5">
                <span className="w-1.5 h-1.5 rounded-full bg-[#C94A4A]"></span> High Risk
              </span>
              <div className="text-2xl font-extrabold text-[#C94A4A] mt-1 font-mono">
                {animatedHigh}
              </div>
              <span className="text-[10px] text-[#C94A4A]/90">Immediate attention</span>
            </div>

            {/* Medium Risk */}
            <div
              onClick={() => setFilterRisk(filterRisk === 'medium' ? 'all' : 'medium')}
              className={`p-4 rounded-xl border transition-all cursor-pointer ${
                filterRisk === 'medium'
                  ? 'bg-[#FEF3E2] border-[#D18A24] shadow-xs ring-1 ring-[#D18A24]/30'
                  : 'bg-[#F8FAF8] border-[#DCE6E0] hover:border-[#D18A24]/40 hover:bg-[#FEF3E2]/50'
              }`}
            >
              <span className="text-[11px] font-bold text-[#D18A24] uppercase tracking-wider flex items-center gap-1.5">
                <span className="w-1.5 h-1.5 rounded-full bg-[#D18A24]"></span> Medium Risk
              </span>
              <div className="text-2xl font-extrabold text-[#D18A24] mt-1 font-mono">
                {animatedMedium}
              </div>
              <span className="text-[10px] text-[#D18A24]/90">Review terms</span>
            </div>

            {/* Low Risk */}
            <div
              onClick={() => setFilterRisk(filterRisk === 'low' ? 'all' : 'low')}
              className={`p-4 rounded-xl border transition-all cursor-pointer ${
                filterRisk === 'low'
                  ? 'bg-[#E8F1EC] border-[#3E8E63] shadow-xs ring-1 ring-[#3E8E63]/30'
                  : 'bg-[#F8FAF8] border-[#DCE6E0] hover:border-[#3E8E63]/40 hover:bg-[#E8F1EC]/50'
              }`}
            >
              <span className="text-[11px] font-bold text-[#3E8E63] uppercase tracking-wider flex items-center gap-1.5">
                <span className="w-1.5 h-1.5 rounded-full bg-[#3E8E63]"></span> Low Risk
              </span>
              <div className="text-2xl font-extrabold text-[#3E8E63] mt-1 font-mono">
                {animatedLow}
              </div>
              <span className="text-[10px] text-[#3E8E63]/90">Standard clauses</span>
            </div>
          </div>
        </div>

        {/* Filter Toolbar */}
        <div className="flex flex-col sm:flex-row items-stretch sm:items-center justify-between gap-3 pt-4 border-t border-[#DCE6E0]">
          <div className="flex items-center gap-2 overflow-x-auto pb-1 sm:pb-0">
            {[
              { id: 'all', label: 'All Clauses', count: stats.total },
              { id: 'high', label: 'High Risk', count: stats.high },
              { id: 'medium', label: 'Medium Risk', count: stats.medium },
              { id: 'low', label: 'Low Risk', count: stats.low },
            ].map(({ id, label, count }) => {
              const isActive = filterRisk === id
              let activeClass = 'border-[#176B4D] bg-[#E8F1EC] text-[#12372A]'
              if (id === 'high') activeClass = 'border-[#C94A4A] bg-[#FFF2F2] text-[#C94A4A]'
              if (id === 'medium') activeClass = 'border-[#D18A24] bg-[#FEF3E2] text-[#D18A24]'
              if (id === 'low') activeClass = 'border-[#3E8E63] bg-[#E8F1EC] text-[#3E8E63]'

              return (
                <button
                  key={id}
                  onClick={() => setFilterRisk(id)}
                  className={`inline-flex items-center gap-2 px-3 py-1.5 text-xs font-bold rounded-xl border transition-all whitespace-nowrap ${
                    isActive
                      ? `${activeClass} shadow-2xs`
                      : 'border-[#DCE6E0] bg-white text-[#66736C] hover:text-[#16221C] hover:bg-[#F8FAF8]'
                  }`}
                >
                  <span>{label}</span>
                  <span className="text-[10px] font-mono px-1.5 py-0.2 rounded bg-black/5 font-bold">
                    {count}
                  </span>
                </button>
              )
            })}
          </div>

          {/* Search Bar */}
          <div className="relative flex-1 sm:max-w-xs">
            <input
              type="text"
              placeholder="Search clause text, risk issue..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="w-full pl-8 pr-7 py-2 text-xs rounded-xl bg-white border border-[#DCE6E0] text-[#16221C] placeholder-[#8C9A92] focus:outline-none focus:border-[#176B4D] focus:ring-1 focus:ring-[#176B4D]/30 transition-all shadow-2xs"
            />
            <span className="absolute left-2.5 top-2.5 text-[#8C9A92] text-xs">🔍</span>
            {searchQuery && (
              <button
                onClick={() => setSearchQuery('')}
                className="absolute right-2.5 top-2 text-[#8C9A92] hover:text-[#16221C] text-xs"
              >
                ✕
              </button>
            )}
          </div>
        </div>
      </div>

      {/* Clause Cards List */}
      {filteredResults.length === 0 ? (
        <div className="p-8 rounded-2xl bg-white border border-[#DCE6E0] text-center text-[#8C9A92] space-y-2 shadow-sm">
          <p>No clauses match your active filters.</p>
          <button
            onClick={() => {
              setFilterRisk('all')
              setSearchQuery('')
            }}
            className="text-xs text-[#176B4D] hover:underline font-bold"
          >
            Clear filters
          </button>
        </div>
      ) : (
        <div className="space-y-5">
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
                badge: 'bg-[#FFF2F2] text-[#C94A4A] border-[#F5C2C2]',
                borderLeft: 'border-l-4 border-l-[#C94A4A]',
                label: 'High Risk',
                icon: '🚨',
              },
              medium: {
                badge: 'bg-[#FEF3E2] text-[#D18A24] border-[#F8DEC0]',
                borderLeft: 'border-l-4 border-l-[#D18A24]',
                label: 'Medium Risk',
                icon: '⚠️',
              },
              low: {
                badge: 'bg-[#E8F1EC] text-[#3E8E63] border-[#C8DECE]',
                borderLeft: 'border-l-4 border-l-[#3E8E63]',
                label: 'Low Risk',
                icon: '✅',
              },
            }[result.risk_level] || {
              badge: 'bg-[#F3F7F5] text-[#66736C] border-[#DCE6E0]',
              borderLeft: 'border-l-4 border-l-[#DCE6E0]',
              label: result.risk_level,
              icon: 'ℹ️',
            }

            return (
              <div
                key={result.clause_id}
                id={`clause-${result.clause_id}`}
                className={`reveal-card group p-5 sm:p-6 rounded-2xl bg-white border border-[#DCE6E0] hover:border-[#176B4D]/40 shadow-sm hover:shadow-md hover:-translate-y-0.5 transition-all duration-200 ease-out space-y-5 ${riskTheme.borderLeft}`}
                style={{ animationDelay: `${Math.min(idx * 40, 300)}ms` }}
              >
                {/* Header */}
                <div className="flex flex-wrap items-center justify-between gap-3 border-b border-[#DCE6E0] pb-3.5">
                  <div className="flex items-center gap-2.5">
                    <span className="text-sm font-bold text-[#16221C] tracking-tight uppercase">
                      {displayClauseNum ? `Clause ${displayClauseNum}` : cleanIdLabel}
                    </span>
                    {displayClauseNum && cleanIdLabel && (
                      <>
                        <span className="text-[#DCE6E0]">•</span>
                        <span className="text-xs text-[#66736C] font-semibold">
                          {cleanIdLabel}
                        </span>
                      </>
                    )}
                  </div>

                  <div className="flex items-center gap-2.5">
                    {/* Confidence */}
                    <span className="text-[11px] font-mono text-[#66736C] bg-[#F8FAF8] px-2.5 py-0.5 rounded-md border border-[#DCE6E0]">
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
                      <span className="text-[10px] font-bold text-[#8C9A92] uppercase tracking-wider">
                        Original Clause Text (PII Masked)
                      </span>
                      {rawText.length > 300 && (
                        <button
                          onClick={() => toggleExpand(result.clause_id)}
                          className="text-xs text-[#176B4D] hover:underline font-semibold transition-colors cursor-pointer"
                        >
                          {isExpanded ? 'Show less' : 'Show full text'}
                        </button>
                      )}
                    </div>
                    <p
                      className={`p-4 bg-[#F8FAF8] border border-[#DCE6E0] text-sm text-[#66736C] leading-relaxed rounded-xl font-sans italic ${
                        rawText.length > 300 && !isExpanded ? 'line-clamp-3' : ''
                      }`}
                    >
                      "{rawText}"
                    </p>
                  </div>
                )}

                {/* Plain-English Legal Risk Explanation */}
                <div className="space-y-1.5">
                  <span className="text-[10px] font-bold text-[#176B4D] uppercase tracking-wider flex items-center gap-1.5">
                    <span>🔍</span> Plain-English Analysis &amp; Exposure
                  </span>
                  <div className="p-4 rounded-xl bg-[#F4F8F5] border border-[#DCE6E0] text-sm text-[#16221C] leading-relaxed font-normal">
                    {result.explanation}
                  </div>
                </div>

                {/* Actionable Advice / Recommendation */}
                {result.recommendation && (
                  <div className="p-4 rounded-xl bg-[#FFF9F2] border border-[#F8DEC0] text-sm text-[#16221C] flex items-start gap-3">
                    <span className="text-base shrink-0 mt-0.5 text-[#D18A24]">💡</span>
                    <div>
                      <span className="font-bold text-[#D18A24] text-xs uppercase tracking-wider block mb-0.5">
                        Recommendation before signing
                      </span>
                      <span className="text-xs sm:text-sm text-[#16221C] leading-relaxed">
                        {result.recommendation}
                      </span>
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
