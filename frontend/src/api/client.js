import axios from 'axios'

// API base URL configured via environment variable (default: http://localhost:8000)
const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL ||
  import.meta.env.VITE_BACKEND_URL ||
  'http://localhost:8000'

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
})

/**
 * Extracts a user-friendly error message from an Axios error response.
 * @param {any} error
 * @returns {string}
 */
export function getErrorMessage(error) {
  if (error.response?.data?.detail) {
    const detail = error.response.data.detail
    if (typeof detail === 'string') {
      return detail
    }
    if (Array.isArray(detail) && detail.length > 0) {
      return detail.map((d) => d.msg || JSON.stringify(d)).join(', ')
    }
    return JSON.stringify(detail)
  }
  return error.message || 'An unexpected error occurred while communicating with the server.'
}

/**
 * Health check endpoint.
 * GET /api/health
 */
export async function checkHealth() {
  const response = await apiClient.get('/api/health')
  return response.data
}

/**
 * Uploads a legal PDF file for validation, text extraction, cleaning, clause segmentation, and PII masking.
 * POST /documents/upload
 * @param {File} file
 * @returns {Promise<{
 *   filename: string,
 *   page_count: number,
 *   character_count: number,
 *   document_id: string,
 *   clauses: Array<{ clause_id: string, clause_number: string | null, text: string }>
 * }>}
 */
export async function uploadDocument(file) {
  const formData = new FormData()
  formData.append('file', file)

  const response = await apiClient.post('/documents/upload', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  })
  return response.data
}

/**
 * Generates an executive summary and key points for a preprocessed document.
 * POST /documents/summarize
 * @param {Array<{ clause_id: string, clause_number: string | null, text: string }>} clauses
 * @returns {Promise<{
 *   summary: string,
 *   key_points: string[],
 *   document_type: string | null
 * }>}
 */
export async function summarizeDocument(clauses, documentId = null) {
  const payload = { clauses }
  if (documentId) {
    payload.document_id = documentId
  }
  const response = await apiClient.post('/documents/summarize', payload)
  return response.data
}

/**
 * Performs legal risk analysis on all preprocessed clause segments in a single AI call.
 * POST /clauses/analyze
 * @param {Array<{ clause_id: string, clause_number: string | null, text: string }>} clauses
 * @param {string | null} documentId
 * @returns {Promise<{
 *   total_clauses: number,
 *   results: Array<{
 *     clause_id: string,
 *     clause_number: string | null,
 *     risk_level: 'low' | 'medium' | 'high',
 *     explanation: string,
 *     recommendation: string,
 *     confidence: number
 *   }>
 * }>}
 */
export async function analyzeClauses(clauses, documentId = null) {
  const payload = { clauses }
  if (documentId) {
    payload.document_id = documentId
  }
  const response = await apiClient.post('/clauses/analyze', payload)
  return response.data
}

/**
 * Grounded Document Q&A against the uploaded document context.
 * POST /documents/ask
 * @param {string} documentId
 * @param {string} question
 * @returns {Promise<{
 *   answer: string,
 *   source_clauses: string[],
 *   confidence: number,
 *   cannot_answer: boolean
 * }>}
 */
export async function askDocumentQuestion(documentId, question) {
  const response = await apiClient.post('/documents/ask', {
    document_id: documentId,
    question,
  })
  return response.data
}

export default apiClient
