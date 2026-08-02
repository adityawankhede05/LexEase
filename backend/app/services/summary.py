import logging
from app.ai.base_provider import BaseAIProvider
from app.ai.exceptions import AIProviderError, AIResponseValidationError
from app.ai.gemini_provider import GeminiProvider
from app.schemas.ai import AITask
from app.schemas.document import ClauseSegment
from app.schemas.summary import DocumentSummaryResponse
from app.services.ai import AIService
from app.services.exceptions import EmptyClauseListError, SummaryGenerationError

logger = logging.getLogger(__name__)

class DocumentSummaryService:
    """
    Service responsible for orchestrating whole document summarization.
    Concatenates ClauseSegment inputs and utilizes AIService for generation.
    """
    def __init__(self, ai_service: AIService | None = None):
        if ai_service is None:
            provider: BaseAIProvider = GeminiProvider()
            self.ai_service = AIService(provider=provider)
        else:
            self.ai_service = ai_service

    async def summarize_document(self, clauses: list[ClauseSegment]) -> DocumentSummaryResponse:
        """
        Concatenates clauses and invokes the AI Foundation layer for whole document summary.
        
        Args:
            clauses: List of ClauseSegment objects to summarize.
            
        Returns:
            DocumentSummaryResponse validated model containing summary, key_points, and document_type.
            
        Raises:
            EmptyClauseListError: If clauses list is empty.
            SummaryGenerationError: If AI generation or schema validation fails.
        """
        if not clauses:
            raise EmptyClauseListError("Clause list cannot be empty for document summarization.")

        formatted_clauses = []
        for clause in clauses:
            if clause.clause_number:
                formatted_clauses.append(f"Clause {clause.clause_number}: {clause.text}")
            else:
                formatted_clauses.append(clause.text)

        full_document_text = "\n\n".join(formatted_clauses)
        payload = {"document_text": full_document_text}

        try:
            summary_response = await self.ai_service.generate(
                task_type=AITask.DOCUMENT_SUMMARY,
                payload=payload,
                response_schema=DocumentSummaryResponse,
            )
            return summary_response
        except (AIProviderError, AIResponseValidationError, ValueError) as e:
            logger.error(f"Failed to generate document summary: {e}")
            raise SummaryGenerationError(f"Document summarization failed: {str(e)}") from e
