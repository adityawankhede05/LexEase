import os
import sys
import json
import logging
from app.ai.base_provider import BaseAIProvider
from app.ai.exceptions import AIProviderError, AIResponseValidationError
from app.ai.provider_factory import get_ai_provider
from app.schemas.ai import AITask
from app.schemas.clause_analysis import (
    ClauseAnalysisAIResponse,
    ClauseAnalysisResponse,
    ClauseRiskResult,
    RiskLevel
)
from app.schemas.document import ClauseSegment
from app.services.ai import AIService
from app.services.exceptions import ClauseAnalysisError, EmptyClauseListError

logger = logging.getLogger(__name__)

class ClauseAnalysisService:
    """
    Service responsible for orchestrating clause-level legal risk analysis.
    Integrates the local ML models (clause classifier and risk safety signal)
    with Groq contextual analysis using the RiskReconciler logic.
    """
    
    # Class-level lazy singletons for local models
    _device = None
    _domain_detector = None
    _clause_classifier = None
    _risk_engine = None
    _router = None

    @classmethod
    def _init_models(cls):
        """Initializes the local ML singletons once."""
        if cls._clause_classifier is None:
            logger.info("Initializing local ML models for hybrid pipeline...")
            import torch
            
            # Resolve root directory to import ml package
            current_file = os.path.abspath(__file__)
            # backend/app/services/clause_analysis.py -> project root
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_file))))
            if project_root not in sys.path:
                sys.path.insert(0, project_root)
                
            from ml.final_pipeline.domain_detector import LegalDomainDetector
            from ml.final_pipeline.clause_classifier import HierarchicalClauseClassifier
            from ml.final_pipeline.risk_engine import LegalRiskEngine
            from ml.final_pipeline.hybrid_router import HybridRiskRouter
            
            cls._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            cls._domain_detector = LegalDomainDetector()
            cls._clause_classifier = HierarchicalClauseClassifier(cls._device)
            cls._risk_engine = LegalRiskEngine(cls._device)
            cls._router = HybridRiskRouter()
            logger.info(f"Local ML models loaded successfully on device: {cls._device}")

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
        Analyzes a list of clause segments using the hybrid local ML + Groq pipeline.
        Only clauses needing deep analysis are routed to Groq.
        """
        if not clauses:
            raise EmptyClauseListError(
                "Clause list cannot be empty for clause risk analysis."
            )

        # 1. Initialize local ML models
        self._init_models()
        from ml.final_pipeline.reconciler import RiskReconciler

        # Combine text to detect overall document domain
        combined_text = "\n\n".join([c.text for c in clauses])
        domain = self._domain_detector.detect_domain(combined_text)
        logger.info(f"Detected document domain: {domain}")

        # 2. Local Clause Classification and Risk Scoring
        local_results = {}
        clauses_to_route = []
        
        for clause in clauses:
            # Predict clause type
            clause_type, confidence = self._clause_classifier.classify_clause(clause.text, domain)
            # Predict local risk safety signal
            local_risk = self._risk_engine.predict_risk(clause.text)
            
            local_results[clause.clause_id] = {
                "clause_type": clause_type,
                "confidence": confidence,
                "local_risk": local_risk
            }
            
            # Check if this clause needs to be routed to Groq
            should_route = self._router.should_route_to_groq(
                domain, clause_type, confidence, local_risk
            )
            
            if should_route:
                clauses_to_route.append(clause)
                
        logger.info(f"Hybrid Router: Routing {len(clauses_to_route)}/{len(clauses)} clauses to Groq for contextual analysis.")

        # 3. Call Groq for the routed clauses (batched in a single request)
        groq_results = {}
        if clauses_to_route:
            clauses_payload = [
                {
                    "clause_id": clause.clause_id,
                    "clause_number": clause.clause_number,
                    "text": clause.text,
                }
                for clause in clauses_to_route
            ]
            clauses_json_str = json.dumps(clauses_payload, ensure_ascii=False, indent=2)
            payload = {"clauses_json": clauses_json_str}
            
            try:
                ai_response: ClauseAnalysisAIResponse = await self.ai_service.generate(
                    task_type=AITask.CLAUSE_ANALYSIS,
                    payload=payload,
                    response_schema=ClauseAnalysisAIResponse,
                )
                
                # Index Groq results by clause_id
                for res in ai_response.results:
                    groq_results[res.clause_id] = {
                        "final_risk": res.risk_level.lower(),
                        "explanation": res.explanation,
                        "recommendation": res.recommendation,
                        "confidence": getattr(res, "confidence", 1.0),
                        "risk_score": getattr(res, "risk_score", None),
                        "risk_label": getattr(res, "risk_label", None),
                        "reasons": getattr(res, "reasons", []),
                        "issues": getattr(res, "issues", []),
                        "recommended_action": getattr(res, "recommended_action", None)
                    }
                    
            except (AIProviderError, AIResponseValidationError, ValueError) as e:
                logger.error(f"Groq batch query failed: {e}. Falling back to local risk values.")
                # We will handle fallbacks dynamically in the reconciliation loop

        # 4. Reconciliation and Output Formatting
        final_results = []
        
        for clause in clauses:
            local_data = local_results[clause.clause_id]
            groq_res = groq_results.get(clause.clause_id)
            
            # Reconcile local prediction with Groq (handles groq_res=None automatically)
            reconciled = RiskReconciler.reconcile(local_data["local_risk"], groq_res)
            
            # Map finalized level to enum
            final_level_str = reconciled["final_risk_level"].lower()
            if final_level_str == "high":
                final_level = RiskLevel.HIGH
            elif final_level_str == "medium":
                final_level = RiskLevel.MEDIUM
            else:
                final_level = RiskLevel.LOW
                
            # Construct standard pydantic result object
            # (includes both standard fields for FE compatibility and detailed hybrid fields)
            result = ClauseRiskResult(
                clause_id=clause.clause_id,
                clause_number=clause.clause_number,
                risk_level=final_level,
                explanation=reconciled["explanation"],
                recommendation=reconciled["recommendation"],
                confidence=groq_res.get("confidence", 1.0) if groq_res else 1.0,
                risk_score=reconciled["final_risk_score"],
                risk_label=reconciled["final_risk_level"].upper(),
                reasons=groq_res.get("reasons", []) if groq_res else [],
                issues=groq_res.get("issues", []) if groq_res else [],
                recommended_action=reconciled["recommendation"]
            )
            
            final_results.append(result)
            
        return ClauseAnalysisResponse(
            total_clauses=len(final_results),
            results=final_results
        )
