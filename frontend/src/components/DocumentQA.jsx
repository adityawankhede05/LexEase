import React, { useState, useRef, useEffect } from 'react'
import { askDocumentQuestion, getErrorMessage } from '../api/client'

function DocumentQA({ documentId, filename, onSelectClause }) {
  const [question, setQuestion] = useState('')
  const [messages, setMessages] = useState([])
  const [isLoading, setIsLoading] = useState(false)
  const [activeError, setActiveError] = useState(null)
  const messagesEndRef = useRef(null)
  const inputRef = useRef(null)
  const threadContainerRef = useRef(null)

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages, isLoading])

  useEffect(() => {
    const container = threadContainerRef.current
    if (!container) return

    let timeoutId = null
    const handleScroll = () => {
      container.classList.add('scrolling')
      clearTimeout(timeoutId)
      timeoutId = setTimeout(() => {
        container.classList.remove('scrolling')
      }, 1000)
    }

    container.addEventListener('scroll', handleScroll, { passive: true })
    return () => {
      container.removeEventListener('scroll', handleScroll)
      clearTimeout(timeoutId)
    }
  }, [])

  const handleAsk = async (qText) => {
    const textToSend = (qText || question).trim()
    if (!textToSend || isLoading) return
    if (!documentId) {
      setActiveError('Document context is not ready. Please re-upload your document.')
      return
    }

    setActiveError(null)
    const userMessage = {
      id: `user-${Date.now()}`,
      sender: 'user',
      text: textToSend,
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
    }

    setMessages((prev) => [...prev, userMessage])
    setQuestion('')
    setIsLoading(true)

    try {
      const response = await askDocumentQuestion(documentId, textToSend)
      const assistantMessage = {
        id: `assistant-${Date.now()}`,
        sender: 'assistant',
        answer: response.answer,
        source_clauses: response.source_clauses || [],
        confidence: response.confidence ?? 0,
        cannot_answer: response.cannot_answer ?? false,
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
      }
      setMessages((prev) => [...prev, assistantMessage])
    } catch (err) {
      const errorMsg = getErrorMessage(err)
      setActiveError(errorMsg)
      const errorMessage = {
        id: `assistant-err-${Date.now()}`,
        sender: 'assistant',
        isError: true,
        answer: `Failed to answer question: ${errorMsg}`,
        source_clauses: [],
        confidence: 0,
        cannot_answer: true,
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
      }
      setMessages((prev) => [...prev, errorMessage])
    } finally {
      setIsLoading(false)
    }
  }

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleAsk()
    }
  }

  return (
    <div className="flex flex-col min-h-[480px] h-[540px] sm:h-[580px] rounded-2xl bg-white border border-[#DCE6E0] shadow-sm overflow-hidden">
      {/* QA Header */}
      <div className="px-6 py-4 border-b border-[#DCE6E0] bg-white flex items-center justify-between flex-wrap gap-2">
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 rounded-xl bg-[#E8F1EC] border border-[#DCE6E0] flex items-center justify-center text-[#176B4D] text-base shadow-2xs">
            💬
          </div>
          <div>
            <h3 className="text-sm sm:text-base font-bold text-[#16221C] tracking-tight">
              Grounded Document Q&amp;A
            </h3>
            <p className="text-xs text-[#66736C]">
              Answers strictly cited from <span className="text-[#16221C] font-semibold">{filename || 'uploaded document'}</span>
            </p>
          </div>
        </div>

        {documentId && (
          <span className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-[#E8F1EC] text-[11px] font-mono text-[#176B4D] border border-[#DCE6E0]">
            <span className="w-1.5 h-1.5 rounded-full bg-[#3E8E63]"></span>
            <span>doc: {documentId.substring(0, 8)}...</span>
          </span>
        )}
      </div>

      {/* Messages Thread */}
      <div ref={threadContainerRef} className="flex-1 p-5 sm:p-6 overflow-y-auto space-y-4 qa-thread-container bg-[#F8FAF8]/50">
        {messages.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-full text-center space-y-4 py-8 px-4 max-w-md mx-auto my-auto select-none">
            <div className="w-12 h-12 rounded-2xl bg-[#E8F1EC] border border-[#DCE6E0] flex items-center justify-center text-[#176B4D] shadow-2xs">
              <svg
                className="w-6 h-6"
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
                strokeWidth="2"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z"
                />
              </svg>
            </div>
            <div className="space-y-1.5">
              <h4 className="text-base sm:text-lg font-bold text-[#12372A] tracking-tight">
                Ask anything about your agreement
              </h4>
              <p className="text-xs sm:text-[13px] text-[#66736C] leading-relaxed max-w-sm mx-auto">
                Ask a question about the document and LexEase will answer using the relevant clauses.
              </p>
            </div>
          </div>
        ) : (
          messages.map((msg) => (
            <div
              key={msg.id}
              className={`flex flex-col ${
                msg.sender === 'user' ? 'items-end' : 'items-start'
              }`}
            >
              {/* Sender Label */}
              <span className="text-[10px] font-bold text-[#8C9A92] uppercase tracking-wider mb-1 px-1">
                {msg.sender === 'user' ? 'You' : 'LexEase AI'}
              </span>

              {/* Message Bubble */}
              <div
                className={`max-w-[88%] sm:max-w-[80%] rounded-2xl p-4 sm:p-5 shadow-2xs space-y-3 ${
                  msg.sender === 'user'
                    ? 'bg-[#E8F1EC] border border-[#C8DECE] text-[#16221C] rounded-tr-xs'
                    : msg.isError
                    ? 'bg-[#FFF2F2] border border-[#F5C2C2] text-[#C94A4A] rounded-tl-xs'
                    : msg.cannot_answer
                    ? 'bg-white border border-[#F8DEC0] text-[#16221C] rounded-tl-xs'
                    : 'bg-white border border-[#DCE6E0] text-[#16221C] rounded-tl-xs shadow-xs'
                }`}
              >
                {/* User Text */}
                {msg.sender === 'user' ? (
                  <p className="text-xs sm:text-sm leading-relaxed font-medium">{msg.text}</p>
                ) : (
                  <>
                    {/* Unanswerable banner */}
                    {msg.cannot_answer && !msg.isError && (
                      <div className="flex items-center gap-2 text-xs font-bold text-[#D18A24] bg-[#FEF3E2] px-3 py-1.5 rounded-lg border border-[#F8DEC0]">
                        <span>⚠️</span> Context Not Present in Agreement
                      </div>
                    )}

                    {/* AI Answer Text */}
                    <p className="text-xs sm:text-sm leading-relaxed whitespace-pre-line text-[#16221C]">
                      {msg.answer}
                    </p>

                    {/* Citations & Confidence Bar */}
                    {!msg.cannot_answer && !msg.isError && (
                      <div className="pt-3 border-t border-[#DCE6E0] flex flex-wrap items-center justify-between gap-2 text-[11px]">
                        {/* Source Clauses */}
                        {msg.source_clauses && msg.source_clauses.length > 0 ? (
                          <div className="flex items-center gap-1.5 flex-wrap">
                            <span className="text-[#66736C] font-semibold">Cited Clauses:</span>
                            {msg.source_clauses.map((clauseId) => (
                              <button
                                key={clauseId}
                                onClick={() => onSelectClause && onSelectClause(clauseId)}
                                className="inline-flex items-center gap-1 px-2.5 py-0.5 rounded-md bg-[#E8F1EC] text-[#176B4D] font-mono font-bold border border-[#C8DECE] hover:bg-[#D4EAE0] transition-all cursor-pointer shadow-2xs"
                                title="Click to view and highlight clause in Risk Analysis"
                              >
                                <span>🔗</span> #{clauseId}
                              </button>
                            ))}
                          </div>
                        ) : (
                          <span className="text-[#8C9A92] italic">No direct clause citation</span>
                        )}

                        {/* Confidence */}
                        <span className="text-[#66736C] font-mono bg-[#F8FAF8] px-2 py-0.5 rounded border border-[#DCE6E0] text-[10px]">
                          {Math.round((msg.confidence ?? 0) * 100)}% match
                        </span>
                      </div>
                    )}
                  </>
                )}
              </div>

              {/* Timestamp */}
              <span className="text-[10px] text-[#8C9A92] px-1 mt-1 font-mono">
                {msg.timestamp}
              </span>
            </div>
          ))
        )}

        {/* Loading Spinner in Thread */}
        {isLoading && (
          <div className="flex flex-col items-start">
            <span className="text-[10px] font-bold text-[#8C9A92] uppercase tracking-wider mb-1 px-1">
              LexEase AI
            </span>
            <div className="bg-white border border-[#DCE6E0] rounded-2xl rounded-tl-xs p-4 shadow-sm flex items-center space-x-3">
              <div className="w-4 h-4 border-2 border-[#176B4D] border-t-transparent rounded-full animate-spin"></div>
              <span className="text-xs text-[#66736C] font-medium">
                Retrieving relevant clauses &amp; synthesizing grounded answer...
              </span>
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Input Area */}
      <div className="p-4 sm:p-5 border-t border-[#DCE6E0] bg-white">
        <form
          onSubmit={(e) => {
            e.preventDefault()
            handleAsk()
          }}
          className="flex items-center gap-2.5"
        >
          <input
            ref={inputRef}
            type="text"
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            onKeyDown={handleKeyDown}
            maxLength={2000}
            disabled={isLoading || !documentId}
            placeholder={
              documentId
                ? 'Ask a question about this agreement (e.g. "What are the termination conditions?")...'
                : 'Upload a document first to ask questions...'
            }
            className="flex-1 px-4 py-3 rounded-xl bg-[#F8FAF8] border border-[#DCE6E0] text-xs sm:text-sm text-[#16221C] placeholder-[#8C9A92] focus:outline-none focus:border-[#176B4D] focus:ring-1 focus:ring-[#176B4D]/30 disabled:opacity-50 disabled:cursor-not-allowed transition-all shadow-2xs"
          />

          <button
            type="submit"
            disabled={!question.trim() || isLoading || !documentId}
            className="px-5 py-3 rounded-xl text-white bg-[#176B4D] hover:bg-[#0F5139] border border-transparent font-bold text-xs sm:text-sm shadow-xs hover:shadow-md active:scale-[0.98] transition-all disabled:opacity-40 disabled:cursor-not-allowed flex items-center gap-2 shrink-0 cursor-pointer"
          >
            <span>Ask</span>
            <svg
              className="w-3.5 h-3.5"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth="2.2"
                d="M14 5l7 7m0 0l-7 7m7-7H3"
              />
            </svg>
          </button>
        </form>

        <div className="flex items-center justify-between px-1 pt-2.5 text-[10px] text-[#8C9A92]">
          <span className="inline-flex items-center gap-1.5">
            <span className="w-1.5 h-1.5 rounded-full bg-[#3E8E63]"></span>
            Grounded lexical retrieval with PII protection
          </span>
          <span className="font-mono">{question.length} / 2000</span>
        </div>
      </div>
    </div>
  )
}

export default DocumentQA
