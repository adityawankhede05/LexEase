import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pytest
from fastapi import HTTPException

from app.core.config import settings
from app.database.models import ClauseFeedback, ClauseFeedbackSubmission
import app.database.session as session_module
from app.api.routers.adaptive_learning import (
    get_dashboard,
    list_pending_feedback,
    list_pending_feedback_submissions,
    require_admin,
    review_feedback,
    review_feedback_submission,
    submit_feedback,
    submit_feedback_batch,
)
from app.schemas.adaptive_learning import (
    ClauseFeedbackCreate,
    ClauseFeedbackReview,
    ClauseFeedbackSubmissionCreate,
    ClauseFeedbackSubmissionReview,
)


def test_feedback_requires_consent_and_stores_clause_only():
    with session_module.SessionLocal() as db:
        with pytest.raises(HTTPException) as error:
            submit_feedback(
                ClauseFeedbackCreate(
                    clause_text="The supplier may terminate at any time.",
                    predicted_risk_level="low",
                    corrected_risk_level="high",
                    consent=False,
                ),
                db,
            )
        assert getattr(error.value, "status_code", None) == 400

        response = submit_feedback(
            ClauseFeedbackCreate(
                clause_text="Contact test@example.com at 9876543210. Supplier has unlimited liability.",
                predicted_risk_level="low",
                corrected_risk_level="high",
                consent=True,
            ),
            db,
        )
        assert response.status == "pending"
    with session_module.SessionLocal() as db:
        record = db.query(ClauseFeedback).filter_by(id=response.id).one()
        assert record.status == "pending"
        assert record.consent is True
        assert "test@example.com" not in record.clause_text
        assert "9876543210" not in record.clause_text


def test_feedback_requires_admin_review(monkeypatch):
    monkeypatch.setenv("ADAPTIVE_LEARNING_ADMIN_TOKEN", "test-admin-secret")
    with session_module.SessionLocal() as db:
        response = submit_feedback(
            ClauseFeedbackCreate(
                clause_text="The tenant may terminate this lease without notice.",
                predicted_risk_level="low",
                corrected_risk_level="high",
                consent=True,
            ),
            db,
        )
        with pytest.raises(HTTPException) as error:
            require_admin(x_adaptive_admin_token=None)
        assert getattr(error.value, "status_code", None) == 403

        require_admin(x_adaptive_admin_token="test-admin-secret")
        pending = list_pending_feedback(db)
        assert any(item.id == response.id for item in pending)

        approved = review_feedback(
            response.id,
            ClauseFeedbackReview(decision="approve"),
            db,
        )
        assert approved.status == "approved"

    with session_module.SessionLocal() as db:
        record = db.query(ClauseFeedback).filter_by(id=response.id).one()
        assert record.status == "approved"
        assert record.reviewed_at is not None


def test_dashboard_reflects_approved_counts_and_gate():
    with session_module.SessionLocal() as db:
        dashboard = get_dashboard(db)
        assert dashboard.active_version == "Baseline"
        assert dashboard.gate_requirements["minimum_total"] == 15
        assert dashboard.gate_requirements["minimum_per_class"] == 5

        # Insert approved feedback to meet gate
        for i in range(5):
            db.add(ClauseFeedback(
                clause_text=f"Low risk clause {i}",
                predicted_risk_level="high",
                corrected_risk_level="low",
                consent=True,
                status="approved",
            ))
            db.add(ClauseFeedback(
                clause_text=f"Medium risk clause {i}",
                predicted_risk_level="low",
                corrected_risk_level="medium",
                consent=True,
                status="approved",
            ))
            db.add(ClauseFeedback(
                clause_text=f"High risk clause {i}",
                predicted_risk_level="low",
                corrected_risk_level="high",
                consent=True,
                status="approved",
            ))
        db.commit()

        updated_dashboard = get_dashboard(db)
        assert updated_dashboard.approved_feedback_count >= 15
        assert updated_dashboard.approved_class_counts["LOW"] >= 5
        assert updated_dashboard.approved_class_counts["MEDIUM"] >= 5
        assert updated_dashboard.approved_class_counts["HIGH"] >= 5
        assert updated_dashboard.gate_met is True


def test_document_batch_submission_and_reviewer_annotation():
    with session_module.SessionLocal() as db:
        batch = submit_feedback_batch(
            ClauseFeedbackSubmissionCreate(
                clauses=[
                    {
                        "clause_id": "batch-c1",
                        "clause_number": "1",
                        "clause_text": "Email legal@example.com. Provider may suspend service.",
                        "predicted_risk_level": "medium",
                        "confidence": 0.85,
                    },
                    {
                        "clause_id": "batch-c2",
                        "clause_number": "2",
                        "clause_text": "Either party may terminate on 30 days notice.",
                        "predicted_risk_level": "low",
                        "confidence": 0.95,
                    },
                ],
                consent=True,
            ),
            db,
        )
        assert batch.status == "pending"
        assert batch.clause_count == 2

        pending_subs = list_pending_feedback_submissions(db)
        target_sub = next(s for s in pending_subs if s.id == batch.id)
        assert "legal@example.com" not in target_sub.clauses[0].clause_text

        reviewed = review_feedback_submission(
            batch.id,
            ClauseFeedbackSubmissionReview(
                decision="annotate",
                annotations=[{"clause_id": "batch-c1", "corrected_risk_level": "high"}],
                review_note="Verified high risk suspension clause.",
            ),
            db,
        )
        assert reviewed.status == "annotated"
        assert reviewed.clause_count == 1

        approved = db.query(ClauseFeedback).filter_by(
            predicted_risk_level="medium",
            corrected_risk_level="high",
            status="approved",
        ).all()
        assert any("Provider may suspend service" in item.clause_text for item in approved)


def test_feedback_with_explanation_and_live_model_isolation():
    with session_module.SessionLocal() as db:
        resp = submit_feedback(
            ClauseFeedbackCreate(
                clause_text="The customer shall indemnify the vendor for all losses exceeding contract value.",
                predicted_risk_level="low",
                corrected_risk_level="high",
                consent=True,
                explanation="Uncapped indemnity clause poses significant liability risk.",
            ),
            db,
        )
        assert resp.status == "pending"
        item_id = resp.id

        # Check pending queue displays user explanation
        pending_items = list_pending_feedback(db)
        target = next(item for item in pending_items if item.id == item_id)
        assert "Uncapped indemnity clause" in target.review_note

        # Reviewer approves with comment
        reviewed = review_feedback(
            item_id,
            ClauseFeedbackReview(decision="approve", review_note="Verified high risk indemnity."),
            db,
        )
        assert reviewed.status == "approved"
        assert "Uncapped indemnity clause" in reviewed.review_note
        assert "Verified high risk indemnity" in reviewed.review_note

        # Verify persisted record
        record = db.query(ClauseFeedback).filter_by(id=item_id).one()
        assert record.status == "approved"
        assert record.reviewed_at is not None
