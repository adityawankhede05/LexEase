import re
import sys
import os
import json
import logging
from typing import Dict, Any, List, Optional

# Resolve paths
script_dir = os.path.dirname(os.path.abspath(__file__))
ml_dir = os.path.dirname(script_dir)
project_root = os.path.dirname(ml_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import model inference helper
from ml.inference.predict import predict_clause

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class LegalRiskEngine:
    """
    A rule-based legal risk reasoning engine that operates independently of the
    ML classification layer. It processes the ML predictions and applies a separate
    rule layer to extract risk signals, protective signals, required context,
    external legal references, and uncertainty metrics.
    """
    
    def __init__(self) -> None:
        # Grounded Indian legal references established in project documentation/PDF
        # (e.g. Section 27 of Indian Contract Act for non-compete, Rent Control Acts for leases)
        self.grounded_statutes = {
            "non_compete": {
                "statute": "Section 27 of the Indian Contract Act, 1872",
                "rules": "Under Section 27, any agreement in restraint of trade or profession is void, except in cases of sale of goodwill. Post-employment non-compete covenants are generally legally unenforceable in India.",
                "source": "Grounded in Indian Contract Law"
            },
            "rent_control": {
                "statute": "State-specific Rent Control Acts (e.g., Maharashtra Rent Control Act, 1999)",
                "rules": "Standard lease increases are governed by state-specific rent ceilings and tenancy rules. Unilateral rent increases without notice are restricted.",
                "source": "Grounded in Indian Tenancy Law"
            },
            "arbitration_cost": {
                "statute": "Section 31A of the Arbitration and Conciliation Act, 1996",
                "rules": "Normally, costs follow the event (losing party pays). A clause forcing the weaker party (tenant/employee) to bear all costs upfront regardless of success is highly asymmetric and may be challenged.",
                "source": "Grounded in Indian ADR Law"
            }
        }

    def _extract_risk_signals(self, text: str, clause_type: str) -> List[str]:
        """Extracts specific vocabulary patterns that indicate potential legal hazards."""
        signals = []
        normalized = text.lower()

        if "without notice" in normalized or "without prior notice" in normalized:
            signals.append("unilateral_action_no_notice")
        if "sole discretion" in normalized or "unilateral right" in normalized:
            signals.append("asymmetric_discretionary_rights")
        if "non-compete" in normalized or "shall not work for" in normalized or "shall not join" in normalized:
            signals.append("post_employment_restraint")
        if "indemnify" in normalized and "harmless" in normalized:
            signals.append("indemnity_obligations")
        if "uncapped" in normalized or "no limit" in normalized:
            signals.append("uncapped_financial_liability")
        if "forfeiture" in normalized or "forfeit" in normalized:
            signals.append("asset_or_deposit_forfeiture_risk")
        if "lock-in" in normalized:
            signals.append("lock_in_period_obligation")
            
        return signals

    def _extract_protective_signals(self, text: str) -> List[str]:
        """Extracts vocabulary patterns indicating protective clauses or mitigations."""
        signals = []
        normalized = text.lower()

        if "mutual agreement" in normalized or "mutually agreed" in normalized:
            signals.append("mutual_consent_requirement")
        if "notice of" in normalized or "days notice" in normalized or "days' notice" in normalized:
            signals.append("notice_period_provision")
        if "cap of" in normalized or "capped at" in normalized or "maximum liability" in normalized:
            signals.append("liability_cap_limit")
        if "cure period" in normalized or "days to remedy" in normalized or "days to cure" in normalized:
            signals.append("default_cure_period")
            
        return signals

    def _get_required_context(self, clause_type: str) -> List[str]:
        """Defines what other parts of the contract must be checked to assess this clause type."""
        if "rent" in clause_type or "payment" in clause_type:
            return ["security_deposit_refund_terms", "maintenance_charges_clause", "late_payment_grace_period"]
        elif "termination" in clause_type:
            return ["cure_period_for_default", "security_deposit_forfeiture_rules", "consequences_of_termination"]
        elif "non_compete" in clause_type or "confidentiality" in clause_type:
            return ["geographic_scope_definition", "duration_of_restriction", "definition_of_confidential_info"]
        return ["governing_law_clause", "dispute_resolution_mechanism"]

    def _get_external_legal_info(self, text: str, clause_type: str) -> Dict[str, Any]:
        """Provides external legal annotations grounded in Indian law, or flags as ungrounded."""
        normalized = text.lower()
        clause_type_norm = clause_type.lower()

        if "non_compete" in clause_type_norm or "competitor" in normalized:
            return self.grounded_statutes["non_compete"]
        elif "rent" in clause_type_norm or "lease" in clause_type_norm or "landlord" in normalized:
            return self.grounded_statutes["rent_control"]
        elif "arbitration" in normalized or "dispute" in normalized:
            if "solely" in normalized or "entirely" in normalized:
                return self.grounded_statutes["arbitration_cost"]

        # Default fallback for rules/topics not grounded in uploaded sources
        return {
            "statute": "Not established by uploaded sources",
            "rules": "No specific statutory rule mapping is established for this clause type by the currently uploaded project documentation sources.",
            "source": "None"
        }

    def assess_clause(self, clause_text: str, available_document_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Main entry point for hybrid risk reasoning.
        1. Calls the ML transformer classifier for predictions.
        2. Evaluates risk signals, protective signals, required context, and legal statutes.
        3. Computes uncertainty/escalation flags.
        
        Args:
            clause_text: Raw text of the legal clause.
            available_document_context: Optional metadata about the full contract.

        Returns:
            A structured legal risk report separating ML classification from rule reasoning.
        """
        # Step 1: Query ML model layer
        ml_prediction = predict_clause(clause_text)
        
        predicted_clause_type = ml_prediction["predicted_clause_type"]
        predicted_risk_level = ml_prediction["predicted_risk_level"]
        ml_confidence = ml_prediction["confidence"]

        # Step 2: Extract Rule Signals (Independent Layer)
        risk_signals = self._extract_risk_signals(clause_text, predicted_clause_type)
        protective_signals = self._extract_protective_signals(clause_text)
        required_context = self._get_required_context(predicted_clause_type)
        external_legal_info = self._get_external_legal_info(clause_text, predicted_clause_type)

        # Step 3: Heuristic Reasoning for Final Risk and Uncertainty
        # Flag uncertainty if:
        # - ML confidence is low (< 0.6)
        # - There is a high risk signal with no protective signal
        # - The ML model predicted "low" but rule engine found high risk signals (e.g. unilateral no-notice termination)
        uncertainty_reasons = []
        if ml_confidence < 0.6:
            uncertainty_reasons.append(f"Low ML model joint confidence score: {ml_confidence:.4f}")
        
        has_high_risk_signal = any(sig in ["unilateral_action_no_notice", "post_employment_restraint", "uncapped_financial_liability"] for sig in risk_signals)
        has_protective_limit = len(protective_signals) > 0
        
        if has_high_risk_signal and not has_protective_limit:
            uncertainty_reasons.append("High risk signals detected without accompanying protective provisions/caps.")

        is_escalation_required = len(uncertainty_reasons) > 0

        # Calculate final risk assessment level based on hybrid indicators
        final_assessment_level = predicted_risk_level
        if has_high_risk_signal:
            final_assessment_level = "high"
        elif len(risk_signals) > 0 and len(protective_signals) == 0:
            final_assessment_level = "medium"

        return {
            "clause_text": clause_text,
            "ml_classification": {
                "predicted_clause_type": predicted_clause_type,
                "dataset_associated_risk_level": predicted_risk_level,
                "clause_type_confidence": ml_prediction["clause_type_confidence"],
                "risk_level_confidence": ml_prediction["risk_level_confidence"],
                "joint_confidence": ml_prediction["confidence"],
                "is_fallback": ml_prediction.get("is_fallback", False)
            },
            "rule_reasoning": {
                "extracted_risk_signals": risk_signals,
                "extracted_protective_signals": protective_signals,
                "required_surrounding_context": required_context,
                "external_legal_grounding": external_legal_info,
                "final_risk_assessment": final_assessment_level,
                "uncertainty_escalation": {
                    "is_escalation_required": is_escalation_required,
                    "reasons": uncertainty_reasons
                }
            }
        }

if __name__ == "__main__":
    engine = LegalRiskEngine()
    
    # Test
    sample_clause = "The landlord reserves the right to terminate this lease agreement at any time without notice."
    assessment = engine.assess_clause(sample_clause)
    print(json.dumps(assessment, indent=2))
