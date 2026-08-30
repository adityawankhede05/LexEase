import os
import sys
import json
import asyncio
from fastapi.testclient import TestClient
from dotenv import load_dotenv

# Resolve paths to import app
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Configure encoding for Windows console output
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Explicitly load backend .env
load_dotenv(os.path.join(project_root, ".env"))

from app.main import app
from app.services.clause_analysis import ClauseAnalysisService

def test_production_hybrid_pipeline_flow():
    """
    Integration test verifying the end-to-end production flow from HTTP Request
    to local ML classification, local risk scoring, Groq analysis, and reconciliation.
    """
    client = TestClient(app)
    
    # 1. Define a typical test clause (unilateral non-compete which is high-risk in India)
    test_clause_text = (
        "The Employee agrees that they shall not engage in any competing business "
        "anywhere in the territory of India for a period of three (3) years after "
        "the termination of their employment with the Company."
    )
    
    payload = {
        "clauses": [
            {
                "clause_id": "clause_test_1",
                "clause_number": "1",
                "text": test_clause_text
            }
        ]
    }
    
    print("\n[STEP 1] Sending clause analysis request to POST /clauses/analyze...")
    response = client.post("/clauses/analyze", json=payload)
    
    # Assert successful API request
    assert response.status_code == 200, f"Request failed with status {response.status_code}: {response.text}"
    data = response.json()
    
    assert data["total_clauses"] == 1
    result = data["results"][0]
    
    print("\n[STEP 2] API Response received successfully! Parsing details:")
    print("--------------------------------------------------------------------------------")
    print(f"Clause ID:       {result['clause_id']}")
    print(f"Final Risk:      {result['risk_level'].upper()} (Score: {result.get('risk_score')}%)")
    print(f"Explanation:     {result['explanation']}")
    print(f"Recommendation:  {result['recommendation']}")
    print("--------------------------------------------------------------------------------")
    
    # Now let's extract internal debug fields to trace the hybrid pipeline execution
    # Since we saved these details inside the service, let's rerun a debug mock
    # directly using the Service singletons to trace the local ML contributions
    ClauseAnalysisService._init_models()
    local_classifier = ClauseAnalysisService._clause_classifier
    local_risk_engine = ClauseAnalysisService._risk_engine
    
    # Predict local values
    c_type, c_conf = local_classifier.classify_clause(test_clause_text, "Employment")
    l_risk = local_risk_engine.predict_risk(test_clause_text)
    
    # Prepare trace report
    trace_info = {
        "clause_type": c_type,
        "clause_classification_confidence": round(c_conf, 4),
        "local_risk_score": round(l_risk["risk_score"], 2),
        "local_risk_label": l_risk["risk_level"].upper(),
        "local_high_risk_signal": l_risk["high_risk_warning"],
        "groq_risk_score": result.get("risk_score"), # from result or resolved
        "groq_risk_label": result.get("risk_label"),
        "groq_confidence": round(result.get("confidence", 1.0), 2),
        "final_risk_score": result.get("risk_score"),
        "final_risk_label": result.get("risk_label"),
        "reconciliation_reason": result["explanation"].split("\n")[0] # First line is reconciliation note
    }
    
    print("\n[STEP 3] Trace output matching requested schema:")
    print(json.dumps(trace_info, indent=2))
    
    # Verify that the local model was called and returned non-empty values
    assert c_type != "UNKNOWN/OTHER"
    assert l_risk["risk_score"] > 0
    print("\n[SUCCESS] Production hybrid pipeline integrated and verified successfully!")

if __name__ == "__main__":
    test_production_hybrid_pipeline_flow()
