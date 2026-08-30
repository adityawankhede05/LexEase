import os
import sys
import torch
import asyncio

# Resolve paths
script_dir = os.path.dirname(os.path.abspath(__file__))
ml_dir = os.path.dirname(script_dir)
project_root = os.path.dirname(ml_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from ml.final_pipeline.domain_detector import LegalDomainDetector
from ml.final_pipeline.clause_classifier import HierarchicalClauseClassifier
from ml.final_pipeline.risk_engine import LegalRiskEngine
from ml.final_pipeline.hybrid_router import HybridRiskRouter
from ml.final_pipeline.reconciler import RiskReconciler

# Initialize pipeline components lazily or globally
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_domain_detector = LegalDomainDetector()
_clause_classifier = HierarchicalClauseClassifier(_device)
_risk_engine = LegalRiskEngine(_device)
_router = HybridRiskRouter()

def split_into_clauses(text: str):
    # Splits by double newlines or typical Clause headers
    paragraphs = text.split("\n\n")
    cleaned_clauses = []
    for p in paragraphs:
        p_clean = p.strip()
        if p_clean:
            cleaned_clauses.append(p_clean)
    return cleaned_clauses

async def analyze_legal_document_async(document_text: str) -> dict:
    # 1. Detect Domain
    domain = _domain_detector.detect_domain(document_text)
    
    # Split text into clauses
    clause_texts = split_into_clauses(document_text)
    results = []
    
    for idx, text in enumerate(clause_texts):
        clause_num = f"Clause {idx + 1}"
        
        # 2. Classify Clause Type
        clause_type, confidence = _clause_classifier.classify_clause(text, domain)
        
        # 3. Predict Local Risk (Baseline/Safety Signal)
        local_risk_data = _risk_engine.predict_risk(text)
        
        # 4. Route to Groq if required
        should_route = _router.should_route_to_groq(domain, clause_type, confidence, local_risk_data)
        
        groq_res = None
        if should_route:
            groq_res = await _router.get_groq_prediction(text, clause_id=f"c_{idx+1}", clause_number=clause_num)
            
        # 5. Reconcile Local ML with Groq
        reconciled = RiskReconciler.reconcile(local_risk_data, groq_res)
            
        results.append({
            "clause_text": text,
            "clause_type": clause_type,
            "clause_confidence": confidence,
            "risk_score": reconciled["final_risk_score"],
            "risk_level": reconciled["final_risk_level"],
            "high_risk_warning": local_risk_data["high_risk_warning"],
            "explanation": reconciled["explanation"],
            "recommendation": reconciled["recommendation"],
            "local_ml_contribution": reconciled["local_ml_contribution"],
            "groq_contribution": reconciled["groq_contribution"]
        })
        
    return {
        "document_domain": domain,
        "clauses": results
    }

def analyze_legal_document(document_text: str) -> dict:
    # Synchronous wrapper for async function
    loop = asyncio.get_event_loop()
    if loop.is_running():
        # Running in Jupyter or an active loop
        import nest_asyncio
        nest_asyncio.apply()
    return asyncio.run(analyze_legal_document_async(document_text))
