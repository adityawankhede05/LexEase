import os
import sys
import json
import httpx

# Resolve paths
script_dir = os.path.dirname(os.path.abspath(__file__))
backend_dir = os.path.dirname(script_dir)
workspace_root = os.path.dirname(backend_dir)
if workspace_root not in sys.path:
    sys.path.insert(0, workspace_root)

# Set stdout encoding
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

def run_loopback_verification():
    print("================================================================================")
    print("RUNNING LIVE LOOPBACK NETWORK FLOW VERIFICATION")
    print("================================================================================\n")
    
    # 1. Query Frontend Dev Server
    print("[STEP 1] Querying Frontend Server at http://localhost:5173/...")
    try:
        r_fe = httpx.get("http://localhost:5173/")
        print(f"  Frontend Status: {r_fe.status_code} OK")
        print(f"  HTML Content Preview: {r_fe.text[:120].strip()}...")
        fe_ok = r_fe.status_code == 200
    except Exception as e:
        print(f"  Error contacting frontend: {e}")
        fe_ok = False
        
    # 2. Query Backend Server
    print("\n[STEP 2] Querying Backend Docs at http://127.0.0.1:8000/docs...")
    try:
        r_be = httpx.get("http://127.0.0.1:8000/docs")
        print(f"  Backend Docs Status: {r_be.status_code} OK")
        be_ok = r_be.status_code == 200
    except Exception as e:
        print(f"  Error contacting backend: {e}")
        be_ok = False
        
    # 3. Simulate E2E upload and analysis requests over TCP network
    print("\n[STEP 3] Uploading real PDF over loopback TCP network...")
    pdf_path = r"C:\Users\Sourish Bhandakkar\.gemini\antigravity\brain\d726b9d7-32ea-4ed6-9dd8-106d5e1f07d8\.user_uploaded\media_1788016402231.pdf"
    if not os.path.exists(pdf_path):
        print(f"  Error: PDF not found at {pdf_path}")
        return
        
    try:
        with open(pdf_path, "rb") as f:
            r_upload = httpx.post(
                "http://127.0.0.1:8000/documents/upload",
                files={"file": ("media_1788016402231.pdf", f, "application/pdf")}
            )
        print(f"  Upload HTTP Status: {r_upload.status_code}")
        upload_data = r_upload.json()
        print(f"  Clauses Extracted:  {len(upload_data.get('clauses', []))}")
        upload_ok = r_upload.status_code == 200
    except Exception as e:
        print(f"  Error uploading PDF: {e}")
        upload_ok = False
        
    print("\n[STEP 4] Analyzing clauses over loopback TCP network...")
    try:
        sample_clauses = upload_data.get("clauses", [])[:3]
        payload = {"clauses": sample_clauses}
        r_analyze = httpx.post("http://127.0.0.1:8000/clauses/analyze", json=payload, timeout=60.0)
        print(f"  Analyze HTTP Status: {r_analyze.status_code}")
        analyze_data = r_analyze.json()
        print(f"  Total Results:       {analyze_data.get('total_clauses')}")
        analyze_ok = r_analyze.status_code == 200
        
        # Print E2E trace details for verification
        print("\n--- LOOPBACK PIPELINE TRACE REPORT ---")
        for idx, res in enumerate(analyze_data.get("results", []), start=1):
            print(f"\n[Clause {idx}] {res['clause_id']}")
            print(f"  Risk level:           {res['risk_level'].upper()} (Score: {res.get('risk_score')}%)")
            print(f"  Explanation (UI bind): {res['explanation'].split(chr(10))[0]}")
            print(f"  Recommendation:       {res['recommendation']}")
            print(f"  Confidence:           {res['confidence'] * 100:.1f}%")
            print("-" * 80)
            
    except Exception as e:
        print(f"  Error analyzing clauses: {e}")
        analyze_ok = False
        
    print("\n================================================================================")
    if fe_ok and be_ok and upload_ok and analyze_ok:
        print("VERIFICATION RESULT: ALL LOOPBACK LIVE SOCKET TESTS PASSED!")
    else:
        print("VERIFICATION RESULT: FAILURE (Some socket tests failed).")
    print("================================================================================\n")
    
    assert fe_ok and be_ok and upload_ok and analyze_ok

if __name__ == "__main__":
    run_loopback_verification()
