import axios from 'axios'

const backendUrl = import.meta.env.VITE_BACKEND_URL || 'http://localhost:8000'

const client = axios.create({
  baseURL: backendUrl,
  headers: {
    'Content-Type': 'application/json',
  },
})

/**
 * Upload a PDF file to the backend for analysis.
 * Uses FormData so the browser sets the correct multipart/form-data boundary automatically.
 * @param {File} file - The PDF File object selected by the user.
 * @returns {Promise<{ document_id: string|null, filename: string, page_count: number, character_count: number, clauses: Array<{ clause_id: string, clause_number: string|null, text: string }> }>}
 */
export async function uploadDocument(file) {
  const formData = new FormData()
  formData.append('file', file)

  const response = await client.post('/documents/upload', formData, {
    // Do NOT set Content-Type manually — let axios/browser set multipart boundary
    headers: {
      'Content-Type': undefined,
    },
  })

  return response.data
}

/**
 * Summarize a legal document using its clause segments.
 * @param {Array<{ clause_id: string, clause_number: string|null, text: string }>} clauses
 * @returns {Promise<{ summary: string, key_points: string[], document_type: string|null }>}
 */
export async function summarizeDocument(clauses) {
  const response = await client.post('/documents/summarize', { clauses })
  return response.data
}

/**
 * Perform clause-level legal risk analysis on a list of clause segments.
 * @param {Array<{ clause_id: string, clause_number: string|null, text: string }>} clauses
 * @returns {Promise<{ total_clauses: number, results: Array<{ clause_id: string, clause_number: string|null, risk_level: string, explanation: string, recommendation: string, confidence: number }> }>}
 */
export async function analyzeClauses(clauses) {
  const response = await client.post('/clauses/analyze', { clauses })
  return response.data
}

export default client
