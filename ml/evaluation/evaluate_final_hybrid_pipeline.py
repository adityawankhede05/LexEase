import os
import sys
import json
import torch
import asyncio
import time
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, confusion_matrix

# Resolve paths
script_dir = os.path.dirname(os.path.abspath(__file__))
ml_dir = os.path.dirname(script_dir)
project_root = os.path.dirname(ml_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from ml.final_pipeline.analyze_document import analyze_legal_document_async
from ml.evaluation.evaluate_final_risk_model import predict_risk, load_final_model, CONTEXT_TEST_CASES
from ml.final_pipeline.reconciler import RiskReconciler

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

async def run_contextual_tests_hybrid(tokenizer, model, risk_engine):
    print("\nRunning Contextual Risk Separation Tests on Hybrid Pipeline...")
    results = []
    pass_count = 0
    total_count = len(CONTEXT_TEST_CASES)
    
    from ml.final_pipeline.hybrid_router import HybridRiskRouter
    router = HybridRiskRouter()
    
    for case in CONTEXT_TEST_CASES:
        # Get local risk score
        local_low = risk_engine.predict_risk(case["low_text"])
        local_high = risk_engine.predict_risk(case["high_text"])
        
        # Get Groq predictions with retries
        groq_low = await get_prediction_with_retry(router, case["low_text"], "low_test")
        groq_high = await get_prediction_with_retry(router, case["high_text"], "high_test")
        
        # Reconcile
        rec_low = RiskReconciler.reconcile(local_low, groq_low)
        rec_high = RiskReconciler.reconcile(local_high, groq_high)
        
        low_score = rec_low["final_risk_score"]
        low_level = rec_low["final_risk_level"]
        high_score = rec_high["final_risk_score"]
        high_level = rec_high["final_risk_level"]
        
        diff = high_score - low_score
        is_pass = diff > 5.0 and (low_score < high_score)
        if is_pass:
            pass_count += 1
            
        results.append({
            "category": case["category"],
            "low_text": case["low_text"],
            "low_score": round(low_score, 2),
            "low_level": low_level,
            "high_text": case["high_text"],
            "high_score": round(high_score, 2),
            "high_level": high_level,
            "diff": round(diff, 2),
            "result": "PASS" if is_pass else "FAIL"
        })
        
    print(f"Hybrid Contextual Test Summary: {pass_count}/{total_count} PASSED.")
    
    # Save text report
    out_dir = os.path.join(ml_dir, "evaluation", "results")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "final_hybrid_contextual_test.txt"), "w") as f:
        f.write("========================================================================================\n")
        f.write("FINAL HYBRID PIPELINE CONTEXTUAL RISK SEPARATION TEST REPORT\n")
        f.write("========================================================================================\n\n")
        for res in results:
            f.write(f"Category: {res['category']}\n")
            f.write(f"  Safer (LOW) Wording:  {res['low_text']}\n")
            f.write(f"    Score: {res['low_score']}% | Level: {res['low_level'].upper()}\n")
            f.write(f"  Riskier (HIGH) Wording: {res['high_text']}\n")
            f.write(f"    Score: {res['high_score']}% | Level: {res['high_level'].upper()}\n")
            f.write(f"  Risk Separation Margin: {res['diff']} points | Result: {res['result']}\n")
            f.write("----------------------------------------------------------------------------------------\n")
        f.write(f"\nFinal Verdict: {'PASS' if pass_count >= 5 else 'FAIL'} (Passed {pass_count} out of {total_count} tests)\n")
        
    return results, (pass_count >= 5)

async def get_prediction_with_retry(router, text, clause_id, max_retries=6, delay=3.0):
    """
    Calls get_groq_prediction and retries if fallback is triggered due to rate limit/429.
    """
    for attempt in range(max_retries):
        res = await router.get_groq_prediction(text, clause_id=clause_id)
        # Check if fallback was triggered by looking for 'API Offline' in reasons
        if "API Offline" not in res.get("reasons", []):
            return res
        # If rate limited, backoff and retry
        backoff = delay * (1.5 ** attempt)
        print(f"Rate limited or API error. Retrying in {backoff:.1f}s (Attempt {attempt+1}/{max_retries})...")
        await asyncio.sleep(backoff)
    # Return last fallback result if all retries failed
    return res

async def evaluate_real_world_hybrid(risk_engine):
    print("\nRunning real-world validation evaluation of hybrid pipeline on 150 clauses (sequential with rate-limit pacing)...")
    val_path = os.path.join(ml_dir, "data", "real_world_validation", "validation_clauses.csv")
    val_df = pd.read_csv(val_path)
    
    from ml.final_pipeline.hybrid_router import HybridRiskRouter
    router = HybridRiskRouter()
    
    processed_results = []
    start_time = time.time()
    
    for idx, row in val_df.iterrows():
        text = str(row["clause_text"])
        
        # 1. Local risk
        local_res = risk_engine.predict_risk(text)
        
        # 2. Call Groq with pacing (2.0s sleep) and active retry handling
        groq_res = await get_prediction_with_retry(router, text, clause_id=f"vc_{idx}")
        
        # 3. Reconcile
        rec = RiskReconciler.reconcile(local_res, groq_res)
        
        processed_results.append({
            "idx": idx,
            "clause_text": text,
            "document_type": row["document_type"],
            "correct_risk_level": row["correct_risk_level"],
            "final_risk_level": rec["final_risk_level"],
            "final_risk_score": rec["final_risk_score"],
            "local_risk_level": local_res["risk_level"],
            "local_risk_score": local_res["risk_score"]
        })
        
        # Print progress every 15 records
        if (idx + 1) % 15 == 0:
            elapsed = time.time() - start_time
            print(f"Processed {idx + 1}/150 clauses... Elapsed: {elapsed:.1f}s")
            
        # Pacing delay to avoid TPM rate limit exhaustion
        await asyncio.sleep(1.8)
        
    results_df = pd.DataFrame(processed_results)
    
    y_true = results_df["correct_risk_level"].tolist()
    y_pred = results_df["final_risk_level"].tolist()
    
    # Compute Hybrid Metrics
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    
    labels = ["low", "medium", "high"]
    p_class, r_class, f1_class, s_class = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    pred_dist = pd.Series(y_pred).value_counts().to_dict()
    
    class_metrics = {}
    for i, label in enumerate(labels):
        class_metrics[label] = {
            "precision": round(p_class[i], 4),
            "recall": round(r_class[i], 4),
            "f1_score": round(f1_class[i], 4),
            "support": int(s_class[i])
        }
        
    high_recall = class_metrics["high"]["recall"]
    
    # Per-domain metrics
    domain_results = {}
    for domain, group in results_df.groupby("document_type"):
        d_true = group["correct_risk_level"].tolist()
        d_pred = group["final_risk_level"].tolist()
        d_acc = accuracy_score(d_true, d_pred)
        d_f1 = f1_score(d_true, d_pred, average="macro", zero_division=0)
        domain_results[domain] = {
            "accuracy": round(d_acc, 4),
            "macro_f1": round(d_f1, 4),
            "support": len(group)
        }
        
    metrics = {
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "high_risk_recall": round(high_recall, 4),
        "prediction_distribution": pred_dist,
        "class_metrics": class_metrics,
        "domain_metrics": domain_results,
        "confusion_matrix": cm.tolist()
    }
    
    # Save JSON metrics
    out_dir = os.path.join(ml_dir, "evaluation", "results")
    with open(os.path.join(out_dir, "final_hybrid_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)
        
    # Generate text report comparing baselines
    with open(os.path.join(out_dir, "final_hybrid_report.txt"), "w") as f:
        f.write("========================================================================================\n")
        f.write("FINAL HYBRID PIPELINE REAL-WORLD EVALUATION REPORT (150 CLAUSES)\n")
        f.write("========================================================================================\n\n")
        f.write(f"Hybrid Accuracy:  {acc*100:.2f}%\n")
        f.write(f"Hybrid Macro F1:  {macro_f1*100:.2f}%\n")
        f.write(f"HIGH-risk Recall:  {high_recall*100:.2f}%\n\n")
        
        f.write("Class-level Metrics:\n")
        f.write(f"  LOW    -> Precision: {p_class[0]:.4f}, Recall: {r_class[0]:.4f}, F1: {f1_class[0]:.4f} (Support: {s_class[0]})\n")
        f.write(f"  MEDIUM -> Precision: {p_class[1]:.4f}, Recall: {r_class[1]:.4f}, F1: {f1_class[1]:.4f} (Support: {s_class[1]})\n")
        f.write(f"  HIGH   -> Precision: {p_class[2]:.4f}, Recall: {r_class[2]:.4f}, F1: {f1_class[2]:.4f} (Support: {s_class[2]})\n\n")
        
        f.write("Confusion Matrix:\n")
        f.write(f"          | Pred Low | Pred Med | Pred High\n")
        f.write(f"---------------------------------------------\n")
        f.write(f"Act Low   | {cm[0][0]:<8} | {cm[0][1]:<8} | {cm[0][2]:<8}\n")
        f.write(f"Act Med   | {cm[1][0]:<8} | {cm[1][1]:<8} | {cm[1][2]:<8}\n")
        f.write(f"Act High  | {cm[2][0]:<8} | {cm[2][1]:<8} | {cm[2][2]:<8}\n\n")
        
        f.write("Prediction Distribution vs Ground Truth:\n")
        f.write(f"  LOW    -> Pred: {pred_dist.get('low', 0):<4} (GT: {s_class[0]})\n")
        f.write(f"  MEDIUM -> Pred: {pred_dist.get('medium', 0):<4} (GT: {s_class[1]})\n")
        f.write(f"  HIGH   -> Pred: {pred_dist.get('high', 0):<4} (GT: {s_class[2]})\n\n")
        
        f.write("Per-Domain Performance:\n")
        for dom, d_met in domain_results.items():
            f.write(f"  {dom:<35} -> Accuracy: {d_met['accuracy']*100:.2f}%, Macro F1: {d_met['macro_f1']*100:.2f}% (Support: {d_met['support']})\n")
            
        f.write("\n========================================================================================\n")
        f.write("COMPARATIVE PERFORMANCE TABLE\n")
        f.write("========================================================================================\n")
        f.write(f"{'Model / Pipeline':<38} | {'Accuracy':<10} | {'Macro F1':<10} | {'HIGH Recall':<11}\n")
        f.write("-" * 78 + "\n")
        f.write(f"{'Experiment 3 (Legal-BERT Multi-Task)':<38} | {'28.00%':<10} | {'29.43%':<10} | {'78.50%':<11}\n")
        f.write(f"{'Pretrained OOF Baseline':<38} | {'58.67%':<10} | {'36.35%':<10} | {'11.11%':<11}\n")
        f.write(f"{'Experiment 6A (Fine-Tuned Risk)':<38} | {'42.67%':<10} | {'42.79%':<10} | {'96.30%':<11}\n")
        f.write(f"{'Groq API-Based Baseline':<38} | {'52.00%':<10} | {'41.62%':<10} | {'96.30%':<11}\n")
        f.write(f"{'Local ML Risk Model (final_risk_model)':<38} | {'42.67%':<10} | {'41.78%':<10} | {'92.59%':<11}\n")
        f.write(f"{'FINAL HYBRID PIPELINE (this run)':<38} | {acc*100:.2f}%     | {macro_f1*100:.2f}%     | {high_recall*100:.2f}%\n")
        
    return metrics

async def main():
    tokenizer, model = load_final_model()
    
    from ml.final_pipeline.risk_engine import LegalRiskEngine
    risk_engine = LegalRiskEngine(DEVICE)
    
    # 1. Run contextual tests
    context_results, pass_verdict = await run_contextual_tests_hybrid(tokenizer, model, risk_engine)
    
    # 2. Run real-world evaluation
    metrics = await evaluate_real_world_hybrid(risk_engine)
    
    print("\nHybrid Evaluation complete. Results saved to ml/evaluation/results/")
    print(f"Overall Hybrid Accuracy: {metrics['accuracy']*100:.2f}%")
    print(f"Overall Hybrid Macro F1: {metrics['macro_f1']*100:.2f}%")
    print(f"Overall Hybrid HIGH Recall: {metrics['high_risk_recall']*100:.2f}%")
    print(f"Contextual Test Verdict: {'PASS' if pass_verdict else 'FAIL'}")

if __name__ == "__main__":
    asyncio.run(main())
