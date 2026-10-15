from datetime import datetime, timezone
import json
import os
from uuid import uuid4

from fastapi import APIRouter, Depends, Header, HTTPException, status
from sqlalchemy.orm import Session

from app.core.config import settings
from app.database.models import ClauseFeedback, ClauseFeedbackSubmission
from app.database.session import get_db
from app.schemas.adaptive_learning import (
    AdaptiveDashboard,
    ClauseFeedbackCreate,
    ClauseFeedbackCreated,
    ClauseFeedbackItem,
    ClauseFeedbackReview,
    ClauseFeedbackSubmissionCreate,
    ClauseFeedbackSubmissionCreated,
    ClauseFeedbackSubmissionReview,
    FeedbackSubmissionClause,
    PendingClauseFeedbackSubmission,
)
from app.services.preprocessing import PreprocessingService

router = APIRouter(prefix="/adaptive-learning", tags=["Reviewed Feedback"])


def require_admin(
    x_adaptive_admin_token: str | None = Header(default=None, alias="X-Adaptive-Admin-Token"),
) -> None:
    expected_token = getattr(settings, "ADAPTIVE_LEARNING_ADMIN_TOKEN", None) or os.getenv(
        "ADAPTIVE_LEARNING_ADMIN_TOKEN", "lexease-review-secret"
    )
    if not x_adaptive_admin_token or x_adaptive_admin_token != expected_token:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="A valid reviewer administrator token is required for this operation.",
        )


def _approved_feedback_counts(db: Session) -> tuple[int, dict[str, int]]:
    records = db.query(ClauseFeedback).filter_by(status="approved").all()
    counts = {"LOW": 0, "MEDIUM": 0, "HIGH": 0}
    for record in records:
        key = record.corrected_risk_level.upper()
        if key in counts:
            counts[key] += 1
    return len(records), counts


@router.get("/dashboard", response_model=AdaptiveDashboard)
def get_dashboard(db: Session = Depends(get_db)) -> AdaptiveDashboard:
    approved_count, approved_class_counts = _approved_feedback_counts(db)
    min_total = 15
    min_per_class = 5
    gate_met = (
        approved_count >= min_total
        and all(approved_class_counts[k] >= min_per_class for k in ["LOW", "MEDIUM", "HIGH"])
    )
    return AdaptiveDashboard(
        active_version="Baseline",
        active_display_label="Production Baseline",
        active_metrics=None,
        pending_feedback_count=db.query(ClauseFeedback).filter_by(status="pending").count(),
        pending_feedback_submission_count=db.query(ClauseFeedbackSubmission).filter_by(status="pending").count(),
        approved_feedback_count=approved_count,
        approved_class_counts=approved_class_counts,
        gate_met=gate_met,
        gate_requirements={"minimum_approved": min_total, "minimum_total": min_total, "minimum_per_class": min_per_class},
        gate_label="Demonstration-only gate (15 approved clauses, ≥5 per class); reviewed clause examples collected for offline evaluation",
        history=[],
    )


@router.post("/feedback", response_model=ClauseFeedbackCreated, status_code=201)
def submit_feedback(
    request: ClauseFeedbackCreate,
    db: Session = Depends(get_db),
) -> ClauseFeedbackCreated:
    if request.consent is not True:
        raise HTTPException(status_code=400, detail="Consent is required to submit clause feedback.")
    if request.predicted_risk_level == request.corrected_risk_level:
        raise HTTPException(status_code=400, detail="Choose a different risk label to submit a correction.")
    text = PreprocessingService.mask_pii(PreprocessingService.clean_text(request.clause_text)).strip()
    if not text:
        raise HTTPException(status_code=400, detail="Clause text is required.")
    group_id = request.source_group_id.strip() if request.source_group_id and request.source_group_id.strip() else f"single_{uuid4()}"
    note = None
    if request.explanation and request.explanation.strip():
        note = f"User explanation: {request.explanation.strip()}"
    record = ClauseFeedback(
        clause_text=text,
        predicted_risk_level=request.predicted_risk_level.value,
        corrected_risk_level=request.corrected_risk_level.value,
        consent=True,
        status="pending",
        review_note=note,
        source_group_id=group_id,
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return ClauseFeedbackCreated(id=record.id, status=record.status)


@router.post("/feedback/submissions", response_model=ClauseFeedbackSubmissionCreated, status_code=201)
def submit_feedback_batch(
    request: ClauseFeedbackSubmissionCreate,
    db: Session = Depends(get_db),
) -> ClauseFeedbackSubmissionCreated:
    if request.consent is not True:
        raise HTTPException(status_code=400, detail="Consent is required to submit document clauses.")

    clauses = []
    for clause in request.clauses:
        text = PreprocessingService.mask_pii(PreprocessingService.clean_text(clause.clause_text)).strip()
        if not text:
            continue
        clauses.append({
            "clause_id": clause.clause_id,
            "clause_number": clause.clause_number,
            "clause_text": text,
            "predicted_risk_level": clause.predicted_risk_level.value,
            "confidence": clause.confidence,
        })
    if not clauses:
        raise HTTPException(status_code=400, detail="No usable clauses were found in the submission.")

    record = ClauseFeedbackSubmission(
        clauses_json=json.dumps(clauses, ensure_ascii=False),
        clause_count=len(clauses),
        consent=True,
        status="pending",
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return ClauseFeedbackSubmissionCreated(
        id=record.id,
        status=record.status,
        clause_count=record.clause_count,
    )


@router.get("/feedback/submissions/pending", response_model=list[PendingClauseFeedbackSubmission], dependencies=[Depends(require_admin)])
def list_pending_feedback_submissions(db: Session = Depends(get_db)) -> list[PendingClauseFeedbackSubmission]:
    records = (
        db.query(ClauseFeedbackSubmission)
        .filter_by(status="pending")
        .order_by(ClauseFeedbackSubmission.created_at.asc())
        .limit(100)
        .all()
    )
    results = []
    for record in records:
        try:
            raw_clauses = json.loads(record.clauses_json or "[]")
        except (TypeError, json.JSONDecodeError):
            raw_clauses = []
        results.append(
            PendingClauseFeedbackSubmission(
                id=record.id,
                status=record.status,
                clause_count=record.clause_count,
                consent=record.consent,
                created_at=record.created_at,
                clauses=[FeedbackSubmissionClause(**c) for c in raw_clauses],
            )
        )
    return results


@router.post("/feedback/submissions/{submission_id}/review", response_model=ClauseFeedbackSubmissionCreated, dependencies=[Depends(require_admin)])
def review_feedback_submission(
    submission_id: str,
    request: ClauseFeedbackSubmissionReview,
    db: Session = Depends(get_db),
) -> ClauseFeedbackSubmissionCreated:
    record = db.query(ClauseFeedbackSubmission).filter_by(id=submission_id).first()
    if record is None:
        raise HTTPException(status_code=404, detail="Feedback submission record was not found.")
    if record.status != "pending":
        raise HTTPException(status_code=409, detail="Feedback submission has already been reviewed.")

    submitted_clauses = json.loads(record.clauses_json or "[]")
    clauses_by_id = {clause["clause_id"]: clause for clause in submitted_clauses}
    unknown_ids = {item.clause_id for item in request.annotations} - clauses_by_id.keys()
    if unknown_ids:
        raise HTTPException(status_code=400, detail="Annotations must refer to clauses in this submission.")

    now = datetime.now(timezone.utc)
    if request.decision == "annotate":
        submission_group = f"submission_{record.id}"
        for annotation in request.annotations:
            clause = clauses_by_id[annotation.clause_id]
            db.add(ClauseFeedback(
                clause_text=clause["clause_text"],
                predicted_risk_level=clause["predicted_risk_level"],
                corrected_risk_level=annotation.corrected_risk_level.value,
                consent=True,
                status="approved",
                review_note=request.review_note or f"Batch annotation from submission {record.id}.",
                source_group_id=submission_group,
                reviewed_at=now,
            ))
        record.status = "annotated"
    else:
        record.status = "rejected"
    record.review_note = request.review_note
    record.reviewed_at = now
    db.commit()
    return ClauseFeedbackSubmissionCreated(
        id=record.id,
        status=record.status,
        clause_count=len(request.annotations) if request.decision == "annotate" else 0,
    )


@router.get("/feedback/pending", response_model=list[ClauseFeedbackItem], dependencies=[Depends(require_admin)])
def list_pending_feedback(db: Session = Depends(get_db)) -> list[ClauseFeedbackItem]:
    return (
        db.query(ClauseFeedback)
        .filter_by(status="pending")
        .order_by(ClauseFeedback.created_at.asc())
        .limit(200)
        .all()
    )


@router.post("/feedback/{feedback_id}/review", response_model=ClauseFeedbackItem, dependencies=[Depends(require_admin)])
def review_feedback(
    feedback_id: str,
    request: ClauseFeedbackReview,
    db: Session = Depends(get_db),
) -> ClauseFeedbackItem:
    record = db.query(ClauseFeedback).filter_by(id=feedback_id).first()
    if record is None:
        raise HTTPException(status_code=404, detail="Feedback record was not found.")
    if record.status != "pending":
        raise HTTPException(status_code=409, detail="Feedback has already been reviewed.")
    record.status = "approved" if request.decision == "approve" else "rejected"
    if request.review_note and request.review_note.strip():
        if record.review_note:
            record.review_note = f"{record.review_note} | Reviewer: {request.review_note.strip()}"
        else:
            record.review_note = request.review_note.strip()
    record.reviewed_at = datetime.now(timezone.utc)
    db.commit()
    db.refresh(record)
    return record
