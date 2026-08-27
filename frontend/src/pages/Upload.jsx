import React, { useState, useRef } from 'react'
import { useNavigate } from 'react-router-dom'
import Card from '../components/Card'
import { uploadDocument } from '../api/client'

function Upload() {
  const [selectedFile, setSelectedFile] = useState(null)
  const [isDragOver, setIsDragOver] = useState(false)
  const [loading, setLoading] = useState(false)
  const [uploadError, setUploadError] = useState(null)
  const fileInputRef = useRef(null)
  const navigate = useNavigate()

  const handleDragOver = (e) => {
    e.preventDefault()
    if (loading) return
    setIsDragOver(true)
  }

  const handleDragLeave = () => {
    if (loading) return
    setIsDragOver(false)
  }

  const handleDrop = (e) => {
    e.preventDefault()
    if (loading) return
    setIsDragOver(false)
    const files = e.dataTransfer.files
    if (files && files.length > 0) {
      const file = files[0]
      if (file.type === 'application/pdf' || file.name.endsWith('.pdf')) {
        setSelectedFile(file)
      } else {
        alert('Please upload a PDF file.')
      }
    }
  }

  const handleFileChange = (e) => {
    if (loading) return
    const files = e.target.files
    if (files && files.length > 0) {
      setSelectedFile(files[0])
    }
  }

  const triggerFileSelect = () => {
    if (loading) return
    fileInputRef.current.click()
  }

  const removeSelectedFile = (e) => {
    e.stopPropagation()
    if (loading) return
    setSelectedFile(null)
    if (fileInputRef.current) {
      fileInputRef.current.value = ''
    }
  }

  const handleAnalyze = async () => {
    if (!selectedFile || loading) return
    setLoading(true)
    setUploadError(null)

    // Calculate size string in KB to pass in state
    const sizeInKB = (selectedFile.size / 1024).toFixed(1) + ' KB'

    try {
      const data = await uploadDocument(selectedFile)
      navigate('/analysis', {
        state: {
          document_id: data.document_id,
          clauses: data.clauses ?? [],
          fileName: selectedFile.name,
          fileSize: sizeInKB,
          fileType: selectedFile.type || 'application/pdf',
        }
      })
    } catch (err) {
      const message =
        err?.response?.data?.detail ??
        err?.response?.data?.message ??
        err?.message ??
        'Upload failed. Please try again.'
      setUploadError(message)
      setLoading(false)
    }
  }

  return (
    <div className="max-w-3xl mx-auto space-y-8 py-4 sm:py-8">
      {/* Page Heading & Header */}
      <div className="text-center space-y-3">
        <h1 className="text-3xl sm:text-4xl font-extrabold tracking-tight text-[#F8FAFC]">
          Upload Legal Document
        </h1>
        <p className="text-[#94A3B8] text-base sm:text-lg max-w-xl mx-auto">
          Upload contracts, agreements, notices, or other legal documents for AI-powered analysis.
        </p>
      </div>

      {/* Main Upload Card */}
      <Card className="p-2">
        <div
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
          onClick={triggerFileSelect}
          className={`border-2 border-dashed rounded-xl p-8 sm:p-12 text-center cursor-pointer transition-all duration-300 ${
            isDragOver
              ? 'border-[#D4AF37] bg-[#2563EB]/5 scale-[0.99]'
              : selectedFile
              ? 'border-emerald-500/50 bg-emerald-500/5'
              : 'border-slate-700/80 hover:border-[#2563EB]/50 hover:bg-slate-800/10'
          } ${loading ? 'opacity-60 cursor-not-allowed' : ''}`}
        >
          <input
            type="file"
            ref={fileInputRef}
            onChange={handleFileChange}
            accept=".pdf"
            className="hidden"
            disabled={loading}
          />

          <div className="flex flex-col items-center justify-center space-y-4">
            {/* PDF/Upload Icon */}
            <div className={`w-16 h-16 rounded-full flex items-center justify-center transition-colors duration-300 ${
              selectedFile ? 'bg-emerald-500/10 text-emerald-400' : 'bg-slate-800 text-[#94A3B8]'
            }`}>
              {selectedFile ? (
                <svg className="w-8 h-8" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
              ) : (
                <svg className="w-8 h-8" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M19.5 14.25v-2.625a3.375 3.375 0 00-3.375-3.375h-1.5A1.125 1.125 0 0113.5 7.125v-1.5a3.375 3.375 0 00-3.375-3.375H8.25m6.75 12H9m1.5-12H5.625c-.621 0-1.125.504-1.125 1.125v17.25c0 .621.504 1.125 1.125 1.125h12.75c.621 0 1.125-.504 1.125-1.125V11.25a9 9 0 00-9-9z" />
                </svg>
              )}
            </div>

            {/* Dynamic Status Text */}
            {selectedFile ? (
              <div className="space-y-2">
                <p className="text-[#F8FAFC] font-semibold text-lg max-w-md break-all mx-auto">
                  {selectedFile.name}
                </p>
                <div className="text-[#94A3B8] text-sm flex flex-col sm:flex-row sm:items-center justify-center gap-x-4 gap-y-1">
                  <span>File Size: {(selectedFile.size / 1024).toFixed(1)} KB</span>
                  <span className="hidden sm:inline text-slate-700">•</span>
                  <span>File Type: {selectedFile.type || 'application/pdf'}</span>
                </div>
                {!loading && (
                  <button
                    onClick={removeSelectedFile}
                    className="mt-3 inline-flex items-center gap-1.5 px-3 py-1 rounded-md text-xs font-medium bg-rose-500/10 text-rose-400 hover:bg-rose-500/20 transition-colors"
                  >
                    <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                      <path strokeLinecap="round" strokeLinejoin="round" d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                    </svg>
                    Remove File
                  </button>
                )}
              </div>
            ) : (
              <div className="space-y-1">
                <p className="text-[#F8FAFC] font-semibold text-lg">
                  Drag & Drop PDF Here
                </p>
                <p className="text-[#94A3B8] text-sm">
                  or <span className="text-[#2563EB] hover:text-[#2563EB]/85 underline font-medium">click to browse</span>
                </p>
                <p className="text-[#94A3B8]/60 text-xs mt-2">
                  Accepted formats: PDF only
                </p>
              </div>
            )}
          </div>
        </div>

        {/* Upload Error Banner */}
        {uploadError && (
          <div className="mt-4 flex items-start gap-3 rounded-lg border border-red-500/30 bg-red-500/10 px-4 py-3">
            <svg className="mt-0.5 h-4 w-4 shrink-0 text-red-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M12 9v2m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
            <p className="text-sm text-red-400 leading-relaxed">{uploadError}</p>
          </div>
        )}

        {/* Submit Section */}
        <div className="mt-6 flex justify-end">
          <button
            onClick={handleAnalyze}
            disabled={!selectedFile || loading}
            className={`w-full sm:w-auto px-8 py-3 rounded-lg font-semibold tracking-wide shadow-md transition-all duration-300 flex items-center justify-center gap-2 ${
              !selectedFile
                ? 'bg-slate-800 text-slate-500 cursor-not-allowed border border-slate-700/50'
                : loading
                ? 'bg-[#2563EB] text-white opacity-80 cursor-wait'
                : 'bg-[#2563EB] hover:bg-[#2563EB]/95 text-white active:scale-95 shadow-[#2563EB]/10 border border-[#2563EB]'
            }`}
          >
            {loading ? (
              <>
                <svg className="animate-spin h-5 w-5 text-white" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                </svg>
                Uploading document...
              </>
            ) : (
              <>
                <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
                </svg>
                Analyze Document
              </>
            )}
          </button>
        </div>
      </Card>
    </div>
  )
}

export default Upload
