import os
import sys

# Resolve paths
script_dir = os.path.dirname(os.path.abspath(__file__))
backend_dir = os.path.dirname(script_dir)
workspace_root = os.path.dirname(backend_dir)
if workspace_root not in sys.path:
    sys.path.insert(0, workspace_root)
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

# Set stdout encoding
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from ml.final_pipeline.reconciler import RiskReconciler

def test_reconciliation_rules():
    print("================================================================================")
    print("VERIFYING PRIORITY RECONCILIATION RULES")
    print("================================================================================\n")
    
    # Define test cases mapping to the examples in the prompt
    test_cases = [
        {
            "name": "Example 1: Groq LOW + local HIGH (no safety warning)",
            "local": {"risk_level": "high", "risk_score": 72.0, "high_risk_warning": False},
            "groq": {"risk_label": "LOW", "risk_score": 20.0, "confidence": 0.9, "reasons": ["Standard terms"], "recommended_action": "None"},
            "expected_level": "low"
        },
        {
            "name": "Example 2: Groq LOW + local HIGH (with safety warning)",
            "local": {"risk_level": "high", "risk_score": 85.0, "high_risk_warning": True},
            "groq": {"risk_label": "LOW", "risk_score": 20.0, "confidence": 0.9, "reasons": ["Standard terms"], "recommended_action": "None"},
            "expected_level": "medium"
        },
        {
            "name": "Example 3: Groq LOW + local HIGH (extremely strong safety warning)",
            "local": {"risk_level": "high", "risk_score": 92.0, "high_risk_warning": True},
            "groq": {"risk_label": "LOW", "risk_score": 15.0, "confidence": 0.9, "reasons": ["Standard terms"], "recommended_action": "None"},
            "expected_level": "high"
        },
        {
            "name": "Example 4: Groq HIGH + local LOW (never downgrade)",
            "local": {"risk_level": "low", "risk_score": 25.0, "high_risk_warning": False},
            "groq": {"risk_label": "HIGH", "risk_score": 85.0, "confidence": 0.95, "reasons": ["Severe liability"], "recommended_action": "Negotiate"},
            "expected_level": "high"
        },
        {
            "name": "Example 5: Groq MEDIUM + local HIGH (no safety warning)",
            "local": {"risk_level": "high", "risk_score": 70.0, "high_risk_warning": False},
            "groq": {"risk_label": "MEDIUM", "risk_score": 55.0, "confidence": 0.85, "reasons": ["Unclear language"], "recommended_action": "Clarify"},
            "expected_level": "medium"
        },
        {
            "name": "Example 6: Groq MEDIUM + local HIGH (with safety warning)",
            "local": {"risk_level": "high", "risk_score": 85.0, "high_risk_warning": True},
            "groq": {"risk_label": "MEDIUM", "risk_score": 55.0, "confidence": 0.85, "reasons": ["Unclear language"], "recommended_action": "Clarify"},
            "expected_level": "high"
        },
        {
            "name": "Example 7: Groq unavailable (fallback to local)",
            "local": {"risk_level": "high", "risk_score": 75.0, "high_risk_warning": True},
            "groq": None,
            "expected_level": "high"
        }
    ]
    
    print(f"{'Local Risk':<15} | {'Groq Risk':<15} | {'Safety Signal':<13} | {'Final Risk':<10} | {'Status':<6} | Reason")
    print("-" * 110)
    
    all_pass = True
    for case in test_cases:
        local_desc = f"{case['local']['risk_level'].upper()} ({case['local']['risk_score']})"
        groq_desc = f"{case['groq']['risk_label'].upper()} ({case['groq']['risk_score']})" if case["groq"] else "Unavailable"
        safety_desc = "TRUE" if case['local']['high_risk_warning'] else "FALSE"
        
        # Run reconciler
        res = RiskReconciler.reconcile(case["local"], case["groq"])
        final_desc = f"{res['final_risk_level'].upper()} ({res['final_risk_score']})"
        
        is_pass = res["final_risk_level"].lower() == case["expected_level"].lower()
        if not is_pass:
            all_pass = False
            
        status = "PASS" if is_pass else "FAIL"
        reason = res["explanation"].split("\n")[0].replace("[Reconciliation] ", "")
        
        print(f"{local_desc:<15} | {groq_desc:<15} | {safety_desc:<13} | {final_desc:<10} | {status:<6} | {reason}")
        
    print("\n================================================================================")
    if all_pass:
        print("VERIFICATION RESULT: ALL TESTS PASSED SUCCESSFULLY!")
    else:
        print("VERIFICATION RESULT: FAILURE (Some test cases failed).")
    print("================================================================================\n")
    
    assert all_pass

if __name__ == "__main__":
    test_reconciliation_rules()
