import asyncio
import logging
from typing import Sequence
from app.ai.base_provider import BaseAIProvider
from app.ai.exceptions import AIProviderError, AIResponseValidationError
from app.ai.provider_factory import get_ai_provider
from app.core.config import settings
from app.schemas.ai import AITask
from app.schemas.document import ClauseSegment
from app.schemas.summary import DocumentSummaryResponse
from app.services.ai import AIService
from app.services.exceptions import EmptyClauseListError, SummaryGenerationError

logger = logging.getLogger(__name__)

# Default maximum estimated tokens per chunk sent to AI provider.
# Keeps each request well below Groq's 8,000 TPM limit (target <= 5,000 tokens including overhead).
DEFAULT_MAX_CHUNK_TOKENS = 3500


def estimate_tokens(text: str) -> int:
    """
    Conservatively estimates the token count for a given text.
    In legal English text, 1 token is approximately 3 to 4 characters.
    Using len(text) / 3.0 provides a safe upper bound.
    """
    if not text:
        return 0
    return max(1, int(len(text) / 3.0))


class DocumentSummaryService:
    """
    Service responsible for orchestrating whole document summarization.
    Implements token-budget-aware chunking, map-reduce summarization, and configurable
    inter-chunk pacing to prevent rate limit (TPM 413/429) errors on large documents.
    """

    def __init__(
        self,
        ai_service: AIService | None = None,
        max_chunk_tokens: int = DEFAULT_MAX_CHUNK_TOKENS,
        chunk_delay: float | None = None,
    ):
        if ai_service is None:
            provider: BaseAIProvider = get_ai_provider()
            self.ai_service = AIService(provider=provider)
        else:
            self.ai_service = ai_service
        self.max_chunk_tokens = max_chunk_tokens
        self.chunk_delay = (
            chunk_delay if chunk_delay is not None else settings.GROQ_REQUEST_DELAY
        )

    @staticmethod
    def _format_clause(clause: ClauseSegment) -> str:
        """Formats a single clause segment for prompt inclusion, preserving clause numbers."""
        if clause.clause_number:
            return f"Clause {clause.clause_number}: {clause.text}"
        return clause.text

    def _chunk_clauses(self, clauses: Sequence[ClauseSegment]) -> list[str]:
        """
        Splits clauses into contiguous chunk strings where each chunk's estimated
        token count is within `self.max_chunk_tokens`.
        """
        chunks: list[str] = []
        current_chunk_clauses: list[str] = []
        current_tokens = 0

        for clause in clauses:
            clause_text = self._format_clause(clause)
            clause_tokens = estimate_tokens(clause_text)

            # If a single clause exceeds max_chunk_tokens, split it by length
            if clause_tokens > self.max_chunk_tokens:
                if current_chunk_clauses:
                    chunks.append("\n\n".join(current_chunk_clauses))
                    current_chunk_clauses = []
                    current_tokens = 0

                char_limit = self.max_chunk_tokens * 3
                for i in range(0, len(clause_text), char_limit):
                    sub_text = clause_text[i : i + char_limit]
                    chunks.append(sub_text)
                continue

            if (current_tokens + clause_tokens > self.max_chunk_tokens) and current_chunk_clauses:
                chunks.append("\n\n".join(current_chunk_clauses))
                current_chunk_clauses = [clause_text]
                current_tokens = clause_tokens
            else:
                current_chunk_clauses.append(clause_text)
                current_tokens += clause_tokens

        if current_chunk_clauses:
            chunks.append("\n\n".join(current_chunk_clauses))

        return chunks

    def _build_combined_summary_prompt_text(
        self, intermediate_summaries: list[DocumentSummaryResponse]
    ) -> str:
        """
        Combines intermediate section summaries and key points into a coherent
        text representation for final synthesis.
        """
        parts = ["Summary of document sections:"]
        for idx, s in enumerate(intermediate_summaries, 1):
            parts.append(f"--- Section {idx} Summary ---")
            parts.append(s.summary)
            if s.key_points:
                parts.append("Key points from this section:")
                for kp in s.key_points:
                    parts.append(f"- {kp}")

        return "\n\n".join(parts)

    def _persist_summary(self, document_id: str, summary_res: DocumentSummaryResponse) -> None:
        """Persists or updates the generated summary for a document in the database."""
        from app.database.models import Document, Summary
        from app.database.session import SessionLocal

        with SessionLocal() as session:
            doc = session.query(Document).filter(Document.id == document_id).first()
            if not doc:
                doc = Document(
                    id=document_id,
                    filename="uploaded_document.pdf",
                    page_count=1,
                    character_count=len(summary_res.summary),
                )
                session.add(doc)
                session.flush()

            existing = session.query(Summary).filter(Summary.document_id == document_id).first()
            if existing:
                existing.summary_text = summary_res.summary
                existing.key_points = summary_res.key_points
                existing.document_type = summary_res.document_type
            else:
                s = Summary(
                    document_id=document_id,
                    summary_text=summary_res.summary,
                    document_type=summary_res.document_type,
                )
                s.key_points = summary_res.key_points
                session.add(s)
            session.commit()

    async def summarize_document(
        self, clauses: list[ClauseSegment], document_id: str | None = None
    ) -> DocumentSummaryResponse:
        """
        Summarizes a legal document represented as a list of ClauseSegment objects.
        Uses single-pass summarization for small documents, and map-reduce chunked
        summarization with pacing for documents exceeding the chunk token budget.
        
        Args:
            clauses: List of ClauseSegment objects to summarize.
            document_id: Optional document ID for database persistence.
            
        Returns:
            DocumentSummaryResponse containing summary, key_points, and document_type.
            
        Raises:
            EmptyClauseListError: If clauses list is empty.
            SummaryGenerationError: If AI generation or schema validation fails.
        """
        if not clauses:
            raise EmptyClauseListError("Clause list cannot be empty for document summarization.")

        chunks = self._chunk_clauses(clauses)
        if not chunks:
            raise EmptyClauseListError("No content available to summarize.")

        try:
            # Case 1: Small document - fits into a single chunk (no pacing delay needed)
            if len(chunks) == 1:
                single_summary = await self.ai_service.generate(
                    task_type=AITask.DOCUMENT_SUMMARY,
                    payload={"document_text": chunks[0]},
                    response_schema=DocumentSummaryResponse,
                )
                if document_id:
                    try:
                        self._persist_summary(document_id, single_summary)
                    except Exception as e:
                        logger.error(f"Failed to persist summary for document {document_id}: {e}", exc_info=True)
                return single_summary

            # Case 2: Large document - Map-Reduce chunked summarization
            logger.info(
                f"Document spans {len(clauses)} clauses and {len(chunks)} chunks. "
                f"Running chunked map-reduce summarization with {self.chunk_delay}s pacing."
            )

            # Map phase: summarize each chunk with inter-chunk pacing
            intermediate_summaries: list[DocumentSummaryResponse] = []
            for chunk_index, chunk in enumerate(chunks, 1):
                if chunk_index > 1 and self.chunk_delay > 0:
                    logger.debug(
                        f"Pacing delay: sleeping {self.chunk_delay}s before chunk {chunk_index}..."
                    )
                    await asyncio.sleep(self.chunk_delay)

                logger.debug(f"Summarizing chunk {chunk_index}/{len(chunks)}")
                chunk_summary = await self.ai_service.generate(
                    task_type=AITask.DOCUMENT_SUMMARY,
                    payload={"document_text": chunk},
                    response_schema=DocumentSummaryResponse,
                )
                intermediate_summaries.append(chunk_summary)

            # Pacing delay before Reduce phase
            if self.chunk_delay > 0:
                logger.debug(
                    f"Pacing delay: sleeping {self.chunk_delay}s before reduce synthesis..."
                )
                await asyncio.sleep(self.chunk_delay)

            # Reduce phase: synthesize intermediate summaries into final response
            combined_text = self._build_combined_summary_prompt_text(intermediate_summaries)
            final_summary = await self.ai_service.generate(
                task_type=AITask.DOCUMENT_SUMMARY,
                payload={"document_text": combined_text},
                response_schema=DocumentSummaryResponse,
            )

            # Preserve document_type if identified during chunking when final is generic
            if not final_summary.document_type or final_summary.document_type == "Legal Document":
                for s in intermediate_summaries:
                    if s.document_type and s.document_type != "Legal Document":
                        final_summary.document_type = s.document_type
                        break

            if document_id:
                try:
                    self._persist_summary(document_id, final_summary)
                except Exception as e:
                    logger.error(f"Failed to persist summary for document {document_id}: {e}", exc_info=True)

            return final_summary

        except (AIProviderError, AIResponseValidationError, ValueError) as e:
            logger.error(f"Failed to generate document summary: {e}")
            raise SummaryGenerationError(f"Document summarization failed: {str(e)}") from e
