import React from 'react'
import { NavLink } from 'react-router-dom'

function Navbar() {
  return (
    <nav className="sticky top-0 z-50 bg-[#0F172A]/85 backdrop-blur-md border-b border-slate-800">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          {/* Logo Section */}
          <div className="flex items-center">
            <NavLink to="/" className="flex items-center gap-3 group">
              <div className="flex items-center justify-center w-9 h-9 rounded-lg bg-[#2563EB]/10 border border-[#2563EB]/30 text-[#D4AF37] text-xl transition-all duration-300 group-hover:scale-105 group-hover:bg-[#2563EB]/20 group-hover:border-[#D4AF37]/50">
                ⚖️
              </div>
              <span className="text-xl font-bold tracking-tight text-[#F8FAFC] transition-colors duration-300 group-hover:text-white">
                Lex<span className="text-[#D4AF37]">Ease</span>
              </span>
            </NavLink>
          </div>

          {/* Navigation Links */}
          <div className="flex space-x-1 sm:space-x-4">
            <NavLink
              to="/"
              end
              className={({ isActive }) =>
                `px-3 py-2 rounded-md text-sm font-medium transition-all duration-200 ${
                  isActive
                    ? 'text-[#D4AF37] bg-slate-800/40 border-b-2 border-[#D4AF37] rounded-b-none'
                    : 'text-[#94A3B8] hover:text-[#F8FAFC] hover:bg-slate-800/20'
                }`
              }
            >
              Home
            </NavLink>
            <NavLink
              to="/upload"
              className={({ isActive }) =>
                `px-3 py-2 rounded-md text-sm font-medium transition-all duration-200 ${
                  isActive
                    ? 'text-[#D4AF37] bg-slate-800/40 border-b-2 border-[#D4AF37] rounded-b-none'
                    : 'text-[#94A3B8] hover:text-[#F8FAFC] hover:bg-slate-800/20'
                }`
              }
            >
              Upload
            </NavLink>
            <NavLink
              to="/analysis"
              className={({ isActive }) =>
                `px-3 py-2 rounded-md text-sm font-medium transition-all duration-200 ${
                  isActive
                    ? 'text-[#D4AF37] bg-slate-800/40 border-b-2 border-[#D4AF37] rounded-b-none'
                    : 'text-[#94A3B8] hover:text-[#F8FAFC] hover:bg-slate-800/20'
                }`
              }
            >
              Analysis
            </NavLink>
          </div>
        </div>
      </div>
    </nav>
  )
}

export default Navbar
