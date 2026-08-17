import React, { useState, useRef, useEffect } from 'react'
import { askDocumentQuestion, getErrorMessage } from '../api/client'

const SAMPLE_QUESTIONS = [
  'What are the payment terms and due dates?',
  'What is the notice period required for termination?',
  'Are there any penalties, fees, or deposit forfeiture conditions?',
  'What are the landlord inspection and property access rights?',
]

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
    <div className="flex flex-col h-[650px] rounded-2xl bg-legal-surface/60 border border-legal-border shadow-2xl backdrop-blur-sm overflow-hidden">
      {/* QA Header */}
      <div className="px-6 py-4 border-b border-legal-border/80 bg-legal-surface/80 flex items-center justify-between">
        <div className="flex items-center gap-2.5">
          <span className="text-xl">💬</span>
          <div>
            <h3 className="text-sm sm:text-base font-bold text-legal-text">
              Grounded Document Q&A
            </h3>
            <p className="text-xs text-legal-textSec">
              Answers are strictly grounded in clauses from <span className="text-legal-text font-medium">{filename || 'uploaded document'}</span>
            </p>
          </div>
        </div>

        {documentId && (
          <span className="hidden sm:inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md bg-legal-secondary text-[11px] font-mono text-legal-textSec border border-legal-border">
            <span className="w-1.5 h-1.5 rounded-full bg-legal-success"></span>
            doc: {documentId.substring(0, 8)}...
          </span>
        )}
      </div>

      {/* Messages Thread */}
      <div ref={threadContainerRef} className="flex-1 p-5 overflow-y-auto space-y-4 qa-thread-container">
        {messages.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-full text-center space-y-4 py-8">
            <div className="w-14 h-14 rounded-2xl bg-legal-info/10 border border-legal-info/20 text-legal-info flex items-center justify-center text-2xl">
              💡
            </div>
            <div className="space-y-1 max-w-md">
              <h4 className="text-sm font-semibold text-legal-text">
                Ask anything about your document
              </h4>
              <p className="text-xs text-legal-textSec">
                LexEase uses lexical retrieval and legal AI to find specific clauses and provide direct, cited answers.
              </p>
            </div>

            {/* Quick Prompts */}
            <div className="w-full max-w-md space-y-2 pt-2 text-left">
              <span className="text-[11px] font-semibold text-legal-textMuted uppercase tracking-wider block text-center">
                Suggested Questions
              </span>
              <div className="grid grid-cols-1 gap-2">
                {SAMPLE_QUESTIONS.map((sample, idx) => (
                  <button
                    key={idx}
                    onClick={() => handleAsk(sample)}
                    className="p-2.5 rounded-xl bg-legal-bg border border-legal-border hover:border-legal-info/40 text-xs text-legal-textSec hover:text-legal-text transition-all text-left flex items-center justify-between group"
                  >
                    <span>{sample}</span>
                    <span className="text-legal-textMuted group-hover:text-legal-info text-xs">
                      &rarr;
                    </span>
                  </button>
                ))}
              </div>
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
              {/* Message Bubble */}
              <div
                className={`max-w-[85%] sm:max-w-[75%] rounded-2xl p-4 shadow-md space-y-2.5 ${
                  msg.sender === 'user'
                    ? 'bg-legal-secondary border border-legal-border text-legal-text rounded-br-none'
                    : msg.isError
                    ? 'bg-legal-danger/10 border border-legal-danger/30 text-legal-text rounded-bl-none'
                    : msg.cannot_answer
                    ? 'bg-legal-surface border border-legal-warning/35 text-legal-text rounded-bl-none'
                    : 'bg-legal-surface border border-legal-border text-legal-text rounded-bl-none'
                }`}
              >
                {/* User Text */}
                {msg.sender === 'user' ? (
                  <p className="text-xs sm:text-sm leading-relaxed">{msg.text}</p>
                ) : (
                  <>
                    {/* Unanswerable banner */}
                    {msg.cannot_answer && !msg.isError && (
                      <div className="flex items-center gap-2 text-xs font-semibold text-legal-warning bg-legal-warning/10 px-2.5 py-1 rounded-md border border-legal-warning/20">
                        <span>⚠️</span> Unanswerable Context
                      </div>
                    )}

                    {/* AI Answer Text */}
                    <p className="text-xs sm:text-sm leading-relaxed whitespace-pre-line text-legal-text">
                      {msg.answer}
                    </p>

                    {/* Citations & Confidence Bar */}
                    {!msg.cannot_answer && !msg.isError && (
                      <div className="pt-2 border-t border-legal-border/60 flex flex-wrap items-center justify-between gap-2 text-[11px]">
                        {/* Source Clauses */}
                        {msg.source_clauses && msg.source_clauses.length > 0 ? (
                          <div className="flex items-center gap-1.5 flex-wrap">
                            <span className="text-legal-textSec font-medium">Sources:</span>
                            {msg.source_clauses.map((clauseId) => (
                              <button
                                key={clauseId}
                                onClick={() => onSelectClause && onSelectClause(clauseId)}
                                className="px-2 py-0.5 rounded bg-legal-info/10 text-legal-info font-mono border border-legal-info/35 hover:bg-legal-info/20 transition-colors"
                                title="Click to view clause"
                              >
                                #{clauseId}
                              </button>
                            ))}
                          </div>
                        ) : (
                          <span className="text-legal-textSec italic">No direct clause citation</span>
                        )}

                        {/* Confidence */}
                        <span className="text-legal-textSec font-mono">
                          {Math.round((msg.confidence ?? 0) * 100)}% match
                        </span>
                      </div>
                    )}
                  </>
                )}
              </div>

              {/* Timestamp */}
              <span className="text-[10px] text-legal-textMuted px-1 mt-1 font-mono">
                {msg.timestamp}
              </span>
            </div>
          ))
        )}

        {/* Loading Spinner in Thread */}
        {isLoading && (
          <div className="flex flex-col items-start">
            <div className="bg-legal-elevated border border-legal-border rounded-2xl rounded-bl-none p-4 shadow-md flex items-center space-x-3">
              <div className="w-4 h-4 border-2 border-legal-info border-t-transparent rounded-full animate-spin"></div>
              <span className="text-xs text-legal-text">
                Searching document clauses & synthesizing grounded answer...
              </span>
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Input Area */}
      <div className="p-5 border-t border-legal-border/80 bg-legal-surface/80">
        <form
          onSubmit={(e) => {
            e.preventDefault()
            handleAsk()
          }}
          className="flex items-center gap-2"
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
                ? 'Ask a question about this document (e.g. "What is the penalty for late rent?")...'
                : 'Upload a document first to ask questions...'
            }
            className="flex-1 px-4 py-2.5 rounded-xl bg-legal-bg border border-legal-border text-xs sm:text-sm text-legal-text placeholder-legal-textMuted focus:outline-none focus:border-legal-info/75 disabled:opacity-50 disabled:cursor-not-allowed"
          />

          <button
            type="submit"
            disabled={!question.trim() || isLoading || !documentId}
            className="px-4 sm:px-5 py-2.5 rounded-xl text-legal-text bg-legal-surface border border-legal-border hover:border-legal-info hover:bg-legal-secondary font-medium text-xs sm:text-sm shadow-md transition-all disabled:opacity-40 disabled:cursor-not-allowed flex items-center gap-1.5 shrink-0"
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
                strokeWidth="2"
                d="M14 5l7 7m0 0l-7 7m7-7H3"
              />
            </svg>
          </button>
        </form>

        <div className="flex items-center justify-between px-1 pt-2 text-[10px] text-legal-textMuted">
          <span>Grounded retrieval with Indian legal PII protection</span>
          <span>{question.length} / 2000 chars</span>
        </div>
      </div>
    </div>
  )
}

export default DocumentQA
