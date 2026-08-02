import React from 'react'

function Card({ children, title, subtitle, extraHeader, hoverable = false, className = '' }) {
  return (
    <div
      className={`bg-[#1E293B] border border-slate-800/80 rounded-xl overflow-hidden shadow-lg transition-all duration-300 ${
        hoverable
          ? 'hover:border-slate-700 hover:shadow-xl hover:shadow-slate-950/20 hover:-translate-y-0.5'
          : ''
      } ${className}`}
    >
      {(title || subtitle || extraHeader) && (
        <div className="px-6 py-4 border-b border-slate-800/80 flex items-center justify-between flex-wrap gap-2 bg-[#1E293B]/60">
          <div>
            {title && (
              <h3 className="text-lg font-semibold text-[#F8FAFC] tracking-tight">
                {title}
              </h3>
            )}
            {subtitle && (
              <p className="text-xs text-[#94A3B8] mt-0.5">
                {subtitle}
              </p>
            )}
          </div>
          {extraHeader && <div>{extraHeader}</div>}
        </div>
      )}
      <div className="px-6 py-5 text-[#F8FAFC]">
        {children}
      </div>
    </div>
  )
}

export default Card
