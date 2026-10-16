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
  if (error.code === 'ERR_NETWORK' || (error.request && !error.response)) {
    return `Cannot reach the LexEase backend at ${API_BASE_URL}. Start or restart the backend, then try again.`
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

// ---------------------------------------------------------------------------
// Feedback and Reviewer Endpoints (Feature 2)
// ---------------------------------------------------------------------------

export async function submitClauseFeedback(payload) {
  const response = await apiClient.post('/adaptive-learning/feedback', payload)
  return response.data
}

export async function submitClauseFeedbackBatch(payload) {
  const response = await apiClient.post('/adaptive-learning/feedback/submissions', payload)
  return response.data
}

export async function getAdaptiveLearningDashboard(adminToken) {
  const headers = adminToken ? { 'X-Adaptive-Admin-Token': adminToken } : {}
  const response = await apiClient.get('/adaptive-learning/dashboard', { headers })
  return response.data
}

export async function getPendingClauseFeedback(adminToken) {
  const response = await apiClient.get('/adaptive-learning/feedback/pending', {
    headers: { 'X-Adaptive-Admin-Token': adminToken },
  })
  return response.data
}

export async function getPendingClauseFeedbackSubmissions(adminToken) {
  const response = await apiClient.get('/adaptive-learning/feedback/submissions/pending', {
    headers: { 'X-Adaptive-Admin-Token': adminToken },
  })
  return response.data
}

export async function reviewClauseFeedback(id, decision, adminToken) {
  const response = await apiClient.post(
    `/adaptive-learning/feedback/${encodeURIComponent(id)}/review`,
    { decision },
    { headers: { 'X-Adaptive-Admin-Token': adminToken } },
  )
  return response.data
}

export async function reviewClauseFeedbackSubmission(id, payload, adminToken) {
  const response = await apiClient.post(
    `/adaptive-learning/feedback/submissions/${encodeURIComponent(id)}/review`,
    payload,
    { headers: { 'X-Adaptive-Admin-Token': adminToken } },
  )
  return response.data
}

// ---------------------------------------------------------------------------
// Document Processing, Summary, Analysis, Q&A
// ---------------------------------------------------------------------------

/**
 * Uploads a legal PDF file for validation, text extraction, cleaning, clause segmentation, and PII masking.
 * POST /documents/upload
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
 */
export async function analyzeClauses(clauses, documentId = null) {
  const payload = { clauses }
  if (documentId) {
    payload.document_id = documentId
  }
  const response = await apiClient.post('/clauses/analyze', payload)
  return response.data
}

// ---------------------------------------------------------------------------
// Proposed Drafting and PDF Export (Feature 1)
// ---------------------------------------------------------------------------

/** Draft alternatives for selected clauses after explicit AI-processing consent. */
export async function rewriteSelectedClauses(payload) {
  const response = await apiClient.post('/clauses/rewrite', payload)
  return response.data
}

/** Download a separate reconstructed DOCX containing user-approved revisions. */
export async function downloadProposedDraft(payload) {
  return apiClient.post('/documents/proposed-draft', payload, { responseType: 'blob' })
}

/** Generate a proposed PDF with selected revisions; original remains untouched. */
export async function downloadProposedPdf(sourceFile, payload) {
  const formData = new FormData()
  formData.append('file', sourceFile, sourceFile.name)
  formData.append('request_json', JSON.stringify(payload))
  return apiClient.post('/documents/proposed-pdf', formData, {
    responseType: 'blob',
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  })
}

/**
 * Grounded Document Q&A against the uploaded document context.
 * POST /documents/ask
 */
export async function askDocumentQuestion(documentId, question) {
  const response = await apiClient.post('/documents/ask', {
    document_id: documentId,
    question,
  })
  return response.data
}

export default apiClient
