import asyncio
import json
import logging
from typing import Sequence
from app.ai.base_provider import BaseAIProvider
from app.ai.exceptions import AIProviderError, AIResponseValidationError
from app.ai.provider_factory import get_ai_provider
from app.core.config import settings
from app.schemas.ai import AITask
from app.schemas.clause_analysis import (
    ClauseAnalysisAIResponse,
    ClauseAnalysisResponse,
    ClauseRiskResult,
)
from app.schemas.document import ClauseSegment
from app.services.ai import AIService
from app.services.exceptions import ClauseAnalysisError, EmptyClauseListError

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


def _serialize_clauses(clauses: Sequence[ClauseSegment]) -> str:
    """Serializes a list of ClauseSegment objects into formatted JSON for prompt insertion."""
    payload = [
        {
            "clause_id": clause.clause_id,
            "clause_number": clause.clause_number,
            "text": clause.text,
        }
        for clause in clauses
    ]
    return json.dumps(payload, ensure_ascii=False, indent=2)


class ClauseAnalysisService:
    """
    Service responsible for orchestrating clause-level legal risk analysis.

    Implements token-budget-aware chunking and inter-chunk pacing to prevent
    rate limit (TPM 413/429) errors on large documents. Results from all chunks
    are aggregated sequentially in Python to maintain original clause order without
    requiring an expensive secondary AI reduction call.
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

    def _chunk_clauses(
        self, clauses: Sequence[ClauseSegment]
    ) -> list[list[ClauseSegment]]:
        """
        Splits clauses into groups where each group's serialized JSON token estimate
        is within `self.max_chunk_tokens`. Never truncates individual clauses.
        """
        chunks: list[list[ClauseSegment]] = []
        current_chunk: list[ClauseSegment] = []

        for clause in clauses:
            if not current_chunk:
                current_chunk.append(clause)
                continue

            candidate_json = _serialize_clauses(current_chunk + [clause])
            candidate_tokens = estimate_tokens(candidate_json)

            if candidate_tokens <= self.max_chunk_tokens:
                current_chunk.append(clause)
            else:
                chunks.append(current_chunk)
                current_chunk = [clause]

        if current_chunk:
            chunks.append(current_chunk)

        return chunks

    async def analyze_clauses(
        self, clauses: list[ClauseSegment]
    ) -> ClauseAnalysisResponse:
        """
        Analyzes a list of clause segments for legal risk.
        Uses a single request for small documents, and sequential chunked requests
        with pacing for documents exceeding the chunk token budget.

        Args:
            clauses: List of ClauseSegment objects to analyze.

        Returns:
            ClauseAnalysisResponse containing total_clauses and a list of
            ClauseRiskResult objects, one per input clause in original order.

        Raises:
            EmptyClauseListError: If the clauses list is empty.
            ClauseAnalysisError: If AI generation or schema validation fails.
        """
        if not clauses:
            raise EmptyClauseListError(
                "Clause list cannot be empty for clause risk analysis."
            )

        chunks = self._chunk_clauses(clauses)
        if not chunks:
            raise EmptyClauseListError("No content available to analyze.")

        try:
            # Case 1: Small document - fits into a single chunk (no pacing delay needed)
            if len(chunks) == 1:
                clauses_json_str = _serialize_clauses(chunks[0])
                payload = {"clauses_json": clauses_json_str}
                ai_response: ClauseAnalysisAIResponse = await self.ai_service.generate(
                    task_type=AITask.CLAUSE_ANALYSIS,
                    payload=payload,
                    response_schema=ClauseAnalysisAIResponse,
                )
                return ClauseAnalysisResponse(
                    total_clauses=len(ai_response.results),
                    results=ai_response.results,
                )

            # Case 2: Large document - Sequential chunked processing with pacing
            logger.info(
                f"Clause analysis spans {len(clauses)} clauses across {len(chunks)} chunks. "
                f"Processing sequentially with {self.chunk_delay}s pacing."
            )

            all_results: list[ClauseRiskResult] = []

            for chunk_index, chunk in enumerate(chunks, 1):
                if chunk_index > 1 and self.chunk_delay > 0:
                    logger.debug(
                        f"Pacing delay: sleeping {self.chunk_delay}s before chunk {chunk_index}..."
                    )
                    await asyncio.sleep(self.chunk_delay)

                logger.debug(
                    f"Analyzing chunk {chunk_index}/{len(chunks)} ({len(chunk)} clauses)"
                )
                clauses_json_str = _serialize_clauses(chunk)
                payload = {"clauses_json": clauses_json_str}
                ai_response: ClauseAnalysisAIResponse = await self.ai_service.generate(
                    task_type=AITask.CLAUSE_ANALYSIS,
                    payload=payload,
                    response_schema=ClauseAnalysisAIResponse,
                )
                all_results.extend(ai_response.results)

            return ClauseAnalysisResponse(
                total_clauses=len(all_results),
                results=all_results,
            )

        except (AIProviderError, AIResponseValidationError, ValueError) as e:
            logger.error(f"Failed to analyze clauses: {e}")
            raise ClauseAnalysisError(
                f"Clause risk analysis failed: {str(e)}"
            ) from e
