import os
import sys
import json
import asyncio
from fastapi.testclient import TestClient
from dotenv import load_dotenv
from unittest.mock import patch

# Resolve paths
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Load env variables
load_dotenv(os.path.join(project_root, ".env"))

# Ensure console supports utf-8 prints on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from app.main import app
from app.services.clause_analysis import ClauseAnalysisService

def run_end_to_end_verification():
    print("================================================================================")
    print("STARTING PRODUCTION PIPELINE END-TO-END VERIFICATION")
    print("================================================================================\n")
    
    client = TestClient(app)
    
    # 5 different representative legal clauses
    test_clauses = [
        {
            "clause_id": "clause_deposit",
            "clause_number": "1",
            "text": (
                "The Tenant shall pay a security deposit of Rs. 50,000, which shall be "
                "refunded in full within 30 days of vacating the premises, subject only "
                "to deduction for actual physical damage."
            )
        },
        {
            "clause_id": "clause_termination",
            "clause_number": "2",
            "text": (
                "The Company shall have the right to terminate this agreement immediately "
                "at any time without notice and without payment of any pending dues if it "
                "is dissatisfied with the Employee's performance."
            )
        },
        {
            "clause_id": "clause_non_compete",
            "clause_number": "3",
            "text": (
                "The Employee shall not engage in any competing business anywhere in India "
                "for a period of 3 years after the termination of employment."
            )
        },
        {
            "clause_id": "clause_indemnity",
            "clause_number": "4",
            "text": (
                "The Vendor shall indemnify, defend, and hold harmless the Company from and "
                "against any and all losses, claims, damages, liabilities, costs, and expenses "
                "of any nature whatsoever, without any cap or limitation."
            )
        },
        {
            "clause_id": "clause_governing_law",
            "clause_number": "5",
            "text": (
                "This Agreement shall be governed by and construed in accordance with the "
                "laws of India, and the courts of New Delhi shall have exclusive jurisdiction "
                "over any disputes arising hereunder."
            )
        }
    ]
    
    payload = {"clauses": test_clauses}
    
    # Send request to production endpoint
    response = client.post("/clauses/analyze", json=payload)
    assert response.status_code == 200, f"Request failed: {response.text}"
    response_data = response.json()
    
    results = {res["clause_id"]: res for res in response_data["results"]}
    
    # Trace models internally to report full metrics
    ClauseAnalysisService._init_models()
    local_classifier = ClauseAnalysisService._clause_classifier
    local_risk_engine = ClauseAnalysisService._risk_engine
    local_router = ClauseAnalysisService._router
    
    print("--- INDIVIDUAL CLAUSE TRACES ---")
    
    for idx, c in enumerate(test_clauses, start=1):
        c_id = c["clause_id"]
        res = results[c_id]
        
        # Internal ML models execution for trace report
        c_type, c_conf = local_classifier.classify_clause(c["text"], "Employment")
        l_risk = local_risk_engine.predict_risk(c["text"])
        routed = local_router.should_route_to_groq("Employment", c_type, c_conf, l_risk)
        
        print(f"\n[Clause {idx}] {c_id.upper()}")
        print(f"  Clause Text:   {c['text']}")
        print(f"  Local predicted type: {c_type} (Conf: {c_conf:.2f})")
        print(f"  Local ML risk score:  {l_risk['risk_score']:.2f}% (Label: {l_risk['risk_level'].upper()})")
        print(f"  Local safety warning: {l_risk['high_risk_warning']}")
        print(f"  Routed to Groq:       {routed}")
        
        if routed:
            # Reconciled final score is 0.7 * groq_score + 0.3 * local_score
            # We can calculate back the groq score: groq_score = (final_score - 0.3 * local_score) / 0.7
            final_score = res["risk_score"]
            groq_score = round((final_score - 0.3 * l_risk["risk_score"]) / 0.7, 2)
            print(f"  Groq score (inferred): {groq_score}% (Label: {res.get('risk_label')})")
        else:
            print("  Groq score:           N/A (Bypassed)")
            
        print(f"  Final risk level:     {res['risk_level'].upper()} (Score: {res['risk_score']}%)")
        print(f"  Final Explanation:    {res['explanation']}")
        print(f"  Final Recommendation: {res['recommendation']}")
        print("-" * 80)
        
    # --- FALLBACK AND RATE LIMIT TEST ---
    print("\n--- Fallback / Error Handling Verification ---")
    
    # Mock AIService.generate to throw AIProviderError (simulating rate limit / 429)
    from app.ai.exceptions import AIProviderError
    
    with patch("app.services.clause_analysis.AIService.generate", side_effect=AIProviderError("Simulated 429 Rate Limit Error")):
        print("Sending request with simulated Groq failure/429...")
        response_fallback = client.post("/clauses/analyze", json=payload)
        
        # Verify that the system did NOT crash and gracefully completed
        assert response_fallback.status_code == 200, f"Fallback failed: {response_fallback.text}"
        fallback_data = response_fallback.json()
        assert fallback_data["total_clauses"] == len(test_clauses)
        
        print("Success: API completed with fallback values during mock rate limit (no crashes).")
        
        # Print a sample fallback result
        fb_res = fallback_data["results"][1]
        print(f"Sample Fallback (clause_termination):")
        print(f"  Risk Level:     {fb_res['risk_level'].upper()} (Score: {fb_res['risk_score']}%)")
        print(f"  Explanation:    {fb_res['explanation']}")
        print("--------------------------------------------------------------------------------")
        
    print("\nE2E Verification successfully completed!")

if __name__ == "__main__":
    run_end_to_end_verification()
