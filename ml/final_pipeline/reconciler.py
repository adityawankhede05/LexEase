import logging

logger = logging.getLogger(__name__)

class RiskReconciler:
    @staticmethod
    def reconcile(local_risk: dict, groq_risk: dict | None) -> dict:
        """
        Reconciles the local ML model's predictions and high-risk safety warning
        with Groq's contextual analysis using a deterministic rule with Groq priority.
        
        Args:
            local_risk: dict containing local risk_score, risk_level, high_risk_warning, requires_contextual_review
            groq_risk: dict containing final_risk (or risk_label), risk_score, confidence, reasons, issues, recommended_action
            
        Returns:
            dict containing:
              - final_risk_level: "low", "medium", "high"
              - final_risk_score: float (0.0 to 100.0)
              - explanation: str
              - recommendation: str
              - local_ml_contribution: dict (scores/labels of local model)
              - groq_contribution: dict
        """
        local_level = local_risk["risk_level"].lower()
        local_score = local_risk["risk_score"]
        has_warning = local_risk["high_risk_warning"]
        
        # If Groq was not queried or is unavailable, default entirely to the local ML model fallback
        if groq_risk is None or "API Offline" in groq_risk.get("reasons", []):
            reconcile_note = "[Reconciliation] Groq is unavailable. Using local ML fallback."
            return {
                "final_risk_level": local_level,
                "final_risk_score": local_score,
                "explanation": f"{reconcile_note}\nEvaluated locally by LexEase hybrid filter (LLM routing skipped).",
                "recommendation": "No remediation required." if local_level == "low" else "Recommend reviewing local risk indicators.",
                "local_ml_contribution": {
                    "score": round(local_score, 2),
                    "level": local_level,
                    "warning": has_warning
                },
                "groq_contribution": None
            }
            
        groq_level = (groq_risk.get("risk_label") or groq_risk.get("final_risk")).lower()
        groq_score = groq_risk.get("risk_score")
        if groq_score is None:
            # Fallback mapping if risk_score not present in raw Groq response
            map_score = {"low": 25.0, "medium": 50.0, "high": 75.0}
            groq_score = map_score.get(groq_level, 25.0)
            
        groq_conf = groq_risk.get("confidence", 1.0)
        
        # Reconciliation Logic (Groq Priority):
        reconciliation_notes = []
        final_level = groq_level
        
        if groq_level == "high":
            # A. Groq is HIGH: never downgrade it
            final_level = "high"
            reconciliation_notes.append("Groq identified severe contractual risks.")
            
        elif groq_level == "medium":
            # B. Groq is MEDIUM: escalate to HIGH only if has_warning or score >= 80
            if has_warning or local_score >= 80.0:
                final_level = "high"
                reconciliation_notes.append("Escalated to HIGH: Local safety signal is strong enough to escalate.")
            else:
                reconciliation_notes.append("Groq identified moderate risks; local model agrees within safety bounds.")
                
        elif groq_level == "low":
            # C. Groq is LOW:
            if has_warning and local_score >= 90.0:
                # Exceptional local safety escalation to HIGH
                final_level = "high"
                reconciliation_notes.append("Exceptional local safety escalation due to an extremely strong risk signal.")
            elif has_warning and local_score >= 70.0:
                # Escalate to MEDIUM
                final_level = "medium"
                reconciliation_notes.append("Groq assessed the clause as low risk, but the local safety model detected a strong risk signal, so the result was conservatively escalated to MEDIUM.")
            else:
                # Trust Groq's LOW
                final_level = "low"
                if local_level == "high":
                    reconciliation_notes.append("Groq is the primary contextual assessment and local signal was not strong enough to override it.")
                else:
                    reconciliation_notes.append("Groq and local model both confirm terms are standard and low-risk.")
                    
        # Calculate final continuous risk score (80% Groq + 20% Local)
        final_score = 0.8 * groq_score + 0.2 * local_score
        
        # Ensure consistency between score and label
        if final_level == "low":
            final_score = min(final_score, 40.0)
        elif final_level == "medium":
            final_score = max(41.0, min(final_score, 59.9))
        elif final_level == "high":
            final_score = max(60.0, final_score)
            
        # Construct explanation combining Groq reasons & reconciliation notes
        explanation_parts = []
        if reconciliation_notes:
            explanation_parts.append(f"[Reconciliation] {' '.join(reconciliation_notes)}")
        
        reasons_list = groq_risk.get("reasons", [])
        if reasons_list:
            explanation_parts.append("Analysis: " + " ".join(reasons_list))
        else:
            explanation_parts.append(groq_risk.get("explanation", ""))
            
        issues_list = groq_risk.get("issues", [])
        if issues_list:
            explanation_parts.append("Issues identified: " + ", ".join(issues_list))
            
        final_explanation = "\n".join(explanation_parts)
        
        # Recommendation
        final_rec = groq_risk.get("recommended_action") or groq_risk.get("recommendation") or "No remediation required."
        
        return {
            "final_risk_level": final_level,
            "final_risk_score": round(final_score, 2),
            "explanation": final_explanation,
            "recommendation": final_rec,
            "local_ml_contribution": {
                "score": round(local_score, 2),
                "level": local_level,
                "warning": has_warning
            },
            "groq_contribution": {
                "score": round(groq_score, 2),
                "level": groq_level,
                "confidence": round(groq_conf, 2)
            }
        }
