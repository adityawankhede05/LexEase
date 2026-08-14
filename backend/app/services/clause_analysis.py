import json
import logging
from app.ai.base_provider import BaseAIProvider
from app.ai.exceptions import AIProviderError, AIResponseValidationError
from app.ai.provider_factory import get_ai_provider
from app.schemas.ai import AITask
from app.schemas.clause_analysis import (
    ClauseAnalysisAIResponse,
    ClauseAnalysisResponse,
)
from app.schemas.document import ClauseSegment
from app.services.ai import AIService
from app.services.exceptions import ClauseAnalysisError, EmptyClauseListError

logger = logging.getLogger(__name__)


class ClauseAnalysisService:
    """
    Service responsible for orchestrating clause-level legal risk analysis.

    All clauses are batched into a single AI request. The AI model returns
    a JSON array of ClauseRiskResult objects, each containing clause_id,
    clause_number, risk_level, explanation, recommendation, and confidence.
    """

    def __init__(self, ai_service: AIService | None = None):
        if ai_service is None:
            provider: BaseAIProvider = get_ai_provider()
            self.ai_service = AIService(provider=provider)
        else:
            self.ai_service = ai_service

    async def analyze_clauses(
        self, clauses: list[ClauseSegment]
    ) -> ClauseAnalysisResponse:
        """
        Analyzes a list of clause segments for legal risk in a single batched AI request.

        Args:
            clauses: List of ClauseSegment objects to analyze.

        Returns:
            ClauseAnalysisResponse containing total_clauses and a list of
            ClauseRiskResult objects, one per input clause.

        Raises:
            EmptyClauseListError: If the clauses list is empty.
            ClauseAnalysisError: If AI generation or schema validation fails.
        """
        if not clauses:
            raise EmptyClauseListError(
                "Clause list cannot be empty for clause risk analysis."
            )

        # Serialize all clauses into a single JSON array for batched processing.
        # Only clause_id, clause_number, and text are passed to the model;
        # the model echoes clause_id and clause_number back for deterministic mapping.
        clauses_payload = [
            {
                "clause_id": clause.clause_id,
                "clause_number": clause.clause_number,
                "text": clause.text,
            }
            for clause in clauses
        ]
        clauses_json_str = json.dumps(clauses_payload, ensure_ascii=False, indent=2)
        payload = {"clauses_json": clauses_json_str}

        try:
            ai_response: ClauseAnalysisAIResponse = await self.ai_service.generate(
                task_type=AITask.CLAUSE_ANALYSIS,
                payload=payload,
                response_schema=ClauseAnalysisAIResponse,
            )
            return ClauseAnalysisResponse(
                total_clauses=len(ai_response.results),
                results=ai_response.results,
            )
        except (AIProviderError, AIResponseValidationError, ValueError) as e:
            logger.error(f"Failed to analyze clauses: {e}")
            raise ClauseAnalysisError(
                f"Clause risk analysis failed: {str(e)}"
            ) from e
