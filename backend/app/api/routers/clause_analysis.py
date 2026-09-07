from fastapi import APIRouter, HTTPException, status
from app.schemas.clause_analysis import ClauseAnalysisRequest, ClauseAnalysisResponse
from app.services.clause_analysis import ClauseAnalysisService
from app.services.exceptions import ClauseAnalysisError, EmptyClauseListError

router = APIRouter(prefix="/clauses", tags=["Clause Analysis"])


@router.post("/analyze", response_model=ClauseAnalysisResponse)
async def analyze_clauses(request: ClauseAnalysisRequest):
    """
    Performs clause-level legal risk analysis on a list of preprocessed clause segments.

    Accepts the same ClauseSegment list produced by POST /documents/upload and
    returns a risk classification (low / medium / high), a plain-English explanation,
    an actionable recommendation, and a confidence score for each clause.
    All clauses are analyzed in a single batched AI request.
    """
    analysis_service = ClauseAnalysisService()
    try:
        return await analysis_service.analyze_clauses(
            request.clauses, document_id=request.document_id
        )
    except EmptyClauseListError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except ClauseAnalysisError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )
