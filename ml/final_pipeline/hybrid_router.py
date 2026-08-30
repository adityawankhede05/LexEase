import os
import sys
import asyncio
from dotenv import load_dotenv

# Resolve paths
script_dir = os.path.dirname(os.path.abspath(__file__))
ml_dir = os.path.dirname(script_dir)
project_root = os.path.dirname(ml_dir)
backend_dir = os.path.join(project_root, "backend")

# Explicitly load backend environment variables before importing app modules
load_dotenv(os.path.join(backend_dir, ".env"))

if project_root not in sys.path:
    sys.path.insert(0, project_root)
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from app.schemas.document import ClauseSegment
from app.services.clause_analysis import ClauseAnalysisService

class HybridRiskRouter:
    def __init__(self):
        self.api_service = ClauseAnalysisService()

    def should_route_to_groq(self, domain: str, clause_type: str, clause_confidence: float, risk_data: dict) -> bool:
        # Routing logic
        # 1. Primary Risk is Medium or High
        if risk_data["risk_level"] in ["medium", "high"]:
            return True
        # 2. 6A warning is triggered
        if risk_data["high_risk_warning"]:
            return True
        # 3. Local models disagree
        if risk_data["requires_contextual_review"]:
            return True
        # 4. Unknown/Other domain or low classification confidence
        if clause_type == "UNKNOWN/OTHER" or clause_confidence < 0.25:
            return True
            
        return False

    async def get_groq_prediction(self, text: str, clause_id: str = "c_demo", clause_number: str = "Clause Demo") -> dict:
        try:
            segment = ClauseSegment(clause_id=clause_id, clause_number=clause_number, text=text)
            response = await self.api_service.analyze_clauses([segment])
            if response and response.results:
                res = response.results[0]
                return {
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
        except Exception as e:
            # Fallback in case of API error/offline
            print(f"Groq API error: {e}", file=sys.stderr)
            
        return {
            "final_risk": "low",
            "explanation": "Automatic low risk classification by local hybrid filter (API offline).",
            "recommendation": "No remediation required.",
            "confidence": 0.5,
            "risk_score": 25.0,
            "risk_label": "LOW",
            "reasons": ["API Offline"],
            "issues": [],
            "recommended_action": "No remediation required."
        }
