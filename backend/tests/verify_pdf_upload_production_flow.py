import os
import sys
import json
from fastapi.testclient import TestClient
from dotenv import load_dotenv

# Resolve paths
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Load env variables
load_dotenv(os.path.join(project_root, ".env"))

# Set encoding for Windows terminals
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from app.main import app

def run_pdf_upload_production_verification():
    print("================================================================================")
    print("RUNNING PDF UPLOAD TO RISK ANALYSIS PRODUCTION FLOW VERIFICATION")
    print("================================================================================\n")
    
    client = TestClient(app)
    
    # 1. Target real user-uploaded PDF file in AppData folder
    pdf_path = r"C:\Users\Sourish Bhandakkar\.gemini\antigravity\brain\d726b9d7-32ea-4ed6-9dd8-106d5e1f07d8\.user_uploaded\media_1788016402231.pdf"
    if not os.path.exists(pdf_path):
        print(f"Error: PDF file not found at: {pdf_path}")
        return
        
    print(f"[STEP 1] Uploading raw PDF document: {pdf_path}")
    with open(pdf_path, "rb") as f:
        response_upload = client.post(
            "/documents/upload",
            files={"file": ("Final Year Project Details.pdf", f, "application/pdf")}
        )
        
    assert response_upload.status_code == 200, f"PDF upload failed: {response_upload.text}"
    upload_data = response_upload.json()
    
    print("\n[STEP 2] PDF text extraction and clause segmentation completed successfully!")
    print(f"  Filename:        {upload_data['filename']}")
    print(f"  Page Count:      {upload_data['page_count']}")
    print(f"  Character Count: {upload_data['character_count']}")
    print(f"  Clauses Found:   {len(upload_data['clauses'])}")
    print(f"  Document ID:     {upload_data['document_id']}")
    
    assert len(upload_data["clauses"]) > 0, "No clauses segmented in the document!"
    
    # Take a sample of the first 4 clauses to run the risk analysis pipeline
    sample_clauses = upload_data["clauses"][:4]
    
    print(f"\n[STEP 3] Sending {len(sample_clauses)} segmented clauses to POST /clauses/analyze...")
    analyze_payload = {"clauses": sample_clauses}
    
    response_analyze = client.post("/clauses/analyze", json=analyze_payload)
    assert response_analyze.status_code == 200, f"Clause analysis failed: {response_analyze.text}"
    analyze_data = response_analyze.json()
    
    print("\n[STEP 4] Risk Analysis and Hybrid Reconciliation completed successfully!")
    print(f"  Total analyzed: {analyze_data['total_clauses']} clauses.")
    
    # Compact trace of results for the analyzed clauses
    print("\n--- DETAILED RISK ANALYSIS TRACE FOR UI DISPLAY ---")
    for idx, res in enumerate(analyze_data["results"], start=1):
        clause_id = res["clause_id"]
        # Find original text from upload
        orig_text = next(c["text"] for c in sample_clauses if c["clause_id"] == clause_id)
        
        print(f"\n[UI Card {idx}] ID: {clause_id}")
        print(f"  Raw Clause Text:  {orig_text[:140]}...")
        # Since we populated these in the new ClauseRiskResult model, let's verify they exist and trace them:
        print(f"  Risk Level (UI):  {res['risk_level'].upper()}")
        print(f"  Risk Score (UI):  {res.get('risk_score')}%")
        print(f"  Explanation (UI): {res['explanation']}")
        print(f"  Recommendation:   {res['recommendation']}")
        print(f"  Confidence:       {res['confidence'] * 100:.1f}%")
        print("-" * 80)
        
    print("\n[SUCCESS] E2E PDF upload to hybrid risk UI presentation flow verified successfully!")

if __name__ == "__main__":
    run_pdf_upload_production_verification()
