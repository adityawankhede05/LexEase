import logging
from app.ai.base_provider import BaseAIProvider
from app.ai.exceptions import AIProviderError, AIResponseValidationError
from app.ai.provider_factory import get_ai_provider
from app.database.document_context_store import (
    DocumentContextStore,
    document_context_store,
)
from app.schemas.ai import AITask
from app.schemas.document import ClauseSegment
from app.schemas.qa import DocumentQAResponse
from app.services.ai import AIService
from app.services.exceptions import (
    DocumentContextNotFoundError,
    QAGenerationError,
)
from app.services.retrieval import ClauseRetrievalService

logger = logging.getLogger(__name__)


class DocumentQAService:
    """
    Service responsible for orchestrating grounded legal document Q&A.

    Flow:
    1. Fetches preprocessed ClauseSegment list from DocumentContextStore.
    2. Runs ClauseRetrievalService to retrieve top relevant clauses.
    3. If no relevant clauses matched, returns cannot_answer=True without calling AI.
    4. Otherwise formats clauses into context payload and calls AIService.
    5. Returns validated DocumentQAResponse.
    """

    def __init__(
        self,
        ai_service: AIService | None = None,
        store: DocumentContextStore | None = None,
        retrieval_service: ClauseRetrievalService | None = None,
    ):
        if ai_service is None:
            provider: BaseAIProvider = get_ai_provider()
            self.ai_service = AIService(provider=provider)
        else:
            self.ai_service = ai_service

        self.store = store if store is not None else document_context_store
        self.retrieval_service = (
            retrieval_service
            if retrieval_service is not None
            else ClauseRetrievalService(top_n=5)
        )

    async def answer_question(
        self, document_id: str, question: str
    ) -> DocumentQAResponse:
        """
        Answers a user question based strictly on the specified document's clauses.

        Args:
            document_id: Stored document context identifier.
            question: User question string.

        Returns:
            DocumentQAResponse containing answer, source_clauses, confidence, cannot_answer.

        Raises:
            DocumentContextNotFoundError: If document_id is not in store.
            QAGenerationError: If AI generation or response parsing fails.
        """
        clauses = self.store.get(document_id)
        if clauses is None:
            raise DocumentContextNotFoundError(
                f"No document context found for document_id: {document_id}"
            )

        relevant_clauses = self.retrieval_service.retrieve(question, clauses)

        if not relevant_clauses:
            return DocumentQAResponse(
                answer="The provided document does not contain relevant information to answer this question.",
                source_clauses=[],
                confidence=0.0,
                cannot_answer=True,
            )

        clauses_context = self._format_clauses(relevant_clauses)
        payload = {"question": question, "clauses_context": clauses_context}

        try:
            response: DocumentQAResponse = await self.ai_service.generate(
                task_type=AITask.DOCUMENT_QA,
                payload=payload,
                response_schema=DocumentQAResponse,
            )
            return response
        except (AIProviderError, AIResponseValidationError, ValueError) as e:
            logger.error(f"Failed to generate document Q&A response: {e}")
            raise QAGenerationError(f"Document Q&A failed: {str(e)}") from e

    @staticmethod
    def _format_clauses(clauses: list[ClauseSegment]) -> str:
        """Formats clause segments into a clear numbered context block for AI prompt."""
        formatted_lines = []
        for clause in clauses:
            header = f"Clause {clause.clause_number}: " if clause.clause_number else ""
            formatted_lines.append(f"[{clause.clause_id}] {header}{clause.text}")
        return "\n\n".join(formatted_lines)
