import React from 'react'

function PageContainer({ children }) {
  return (
    <div className="min-h-[calc(100vh-4rem)] bg-[#0F172A] text-[#F8FAFC] py-8 px-4 sm:px-6 lg:px-8 transition-colors duration-300">
      <div className="max-w-7xl mx-auto w-full animate-fade-in">
        {children}
      </div>
    </div>
  )
}

export default PageContainer
