import os
import sys
import json
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from transformers import AutoTokenizer
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, confusion_matrix

# Resolve paths
script_dir = os.path.dirname(os.path.abspath(__file__))
ml_dir = os.path.dirname(script_dir)
project_root = os.path.dirname(ml_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from ml.evaluation.evaluate_pretrained_risk_models import MultiTaskLegalModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Contextual paired test cases
CONTEXT_TEST_CASES = [
    {
        "category": "Termination",
        "low_text": "Either party may terminate this agreement at any time by providing at least 30 days prior written notice to the other party.",
        "high_text": "Company may terminate this agreement immediately at any time without notice and without payment of any pending dues."
    },
    {
        "category": "Indemnity",
        "low_text": "Vendor shall indemnify the Company for direct losses arising from breach of contract, capped at the total contract value.",
        "high_text": "Vendor shall indemnify Company for all losses, damages, costs and expenses of any nature whatsoever without any cap or limit."
    },
    {
        "category": "Confidentiality",
        "low_text": "The receiving party shall keep all confidential information secret for a period of 3 years following the termination of this Agreement.",
        "high_text": "The receiving party shall keep all information confidential forever under penalty of immediate liquidated damages of 50 lakhs."
    },
    {
        "category": "Liability",
        "low_text": "Except for fraud or willful misconduct, either party's maximum aggregate liability shall be limited to the total fees paid under this contract.",
        "high_text": "Neither party limits its liability for any breach under this Agreement, exposing both parties to unlimited exposure."
    },
    {
        "category": "Non-compete",
        "low_text": "During the term of employment, the Employee shall not work for any direct competitor of the Employer.",
        "high_text": "Employee shall not compete in any capacity anywhere in India for 3 years after termination."
    },
    {
        "category": "Payment/Penalty",
        "low_text": "Payments shall be made within 45 days of receipt of a valid invoice, in compliance with the MSMED Act, 2006.",
        "high_text": "Company may withhold payment at its sole discretion if it is dissatisfied with work quality, without any interest on late payments."
    },
    {
        "category": "Deposit/Lease",
        "low_text": "The security deposit shall be refundable in full within 30 days of vacating the premises, subject only to actual physical damage.",
        "high_text": "The security deposit is non-refundable and shall be forfeited in full if the Tenant terminates before the 24-month lock-in period."
    }
]

def load_final_model():
    model_dir = os.path.join(ml_dir, "models", "final_risk_model")
    with open(os.path.join(model_dir, "metadata.json")) as f:
        meta = json.load(f)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = MultiTaskLegalModel("nlpaueb/legal-bert-base-uncased", meta["num_labels"])
    model.load_state_dict(
        torch.load(os.path.join(model_dir, "pytorch_model.bin"), map_location=DEVICE)
    )
    model.to(DEVICE).eval()
    return tokenizer, model

def predict_risk(text, tokenizer, model):
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=256, padding=True)
    with torch.no_grad():
        _, risk_raw = model(enc["input_ids"].to(DEVICE), enc["attention_mask"].to(DEVICE))
        score = float(risk_raw.item()) * 100
    
    # Frozen calibrated thresholds: LOW < 41, MEDIUM < 60, HIGH >= 60
    if score < 41.0:
        level = "low"
    elif score < 60.0:
        level = "medium"
    else:
        level = "high"
    return score, level

def run_contextual_tests(tokenizer, model):
    print("\nRunning Contextual Risk Separation Tests...")
    results = []
    pass_count = 0
    total_count = len(CONTEXT_TEST_CASES)
    
    for case in CONTEXT_TEST_CASES:
        low_score, low_level = predict_risk(case["low_text"], tokenizer, model)
        high_score, high_level = predict_risk(case["high_text"], tokenizer, model)
        
        diff = high_score - low_score
        is_pass = diff > 5.0 # Context separation: HIGH risk should score at least 5 points higher than LOW
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
        
    print(f"Contextual Test Summary: {pass_count}/{total_count} PASSED.")
    
    # Save text report
    out_dir = os.path.join(ml_dir, "evaluation", "results")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "final_contextual_risk_test.txt"), "w") as f:
        f.write("========================================================================================\n")
        f.write("FINAL CONTEXTUAL RISK SEPARATION TEST REPORT\n")
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

def run_real_world_evaluation(tokenizer, model):
    print("\nRunning real-world validation evaluation on 150 clauses...")
    val_path = os.path.join(ml_dir, "data", "real_world_validation", "validation_clauses.csv")
    val_df = pd.read_csv(val_path)
    
    y_true = val_df["correct_risk_level"].tolist()
    y_pred = []
    scores = []
    
    for idx, row in val_df.iterrows():
        score, level = predict_risk(row["clause_text"], tokenizer, model)
        y_pred.append(level)
        scores.append(score)
        
    # Standard metrics
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    
    labels = ["low", "medium", "high"]
    p_class, r_class, f1_class, s_class = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    pred_dist = pd.Series(y_pred).value_counts().to_dict()
    
    # Class-level support
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
    val_df["pred_level"] = y_pred
    for domain, group in val_df.groupby("document_type"):
        d_true = group["correct_risk_level"].tolist()
        d_pred = group["pred_level"].tolist()
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
    
    # Save metrics JSON
    out_dir = os.path.join(ml_dir, "evaluation", "results")
    with open(os.path.join(out_dir, "final_risk_model_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)
        
    # Generate text report
    with open(os.path.join(out_dir, "final_risk_model_report.txt"), "w") as f:
        f.write("========================================================================================\n")
        f.write("FINAL RISK MODEL REAL-WORLD EVALUATION REPORT (150 CLAUSES)\n")
        f.write("========================================================================================\n\n")
        f.write(f"Overall Accuracy:  {acc*100:.2f}%\n")
        f.write(f"Overall Macro F1:  {macro_f1*100:.2f}%\n")
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
        f.write(f"{'Previous final_risk_model':<38} | {'37.33%':<10} | {'37.68%':<10} | {'96.30%':<11}\n")
        f.write(f"{'New FINAL RISK MODEL (this run)':<38} | {acc*100:.2f}%     | {macro_f1*100:.2f}%     | {high_recall*100:.2f}%\n")
        
    return metrics

def main():
    tokenizer, model = load_final_model()
    
    # 1. Run contextual tests
    context_results, pass_verdict = run_contextual_tests(tokenizer, model)
    
    # 2. Run real-world evaluation
    metrics = run_real_world_evaluation(tokenizer, model)
    
    print("\nEvaluation complete. Results saved to ml/evaluation/results/")
    print(f"Overall Real-World Accuracy: {metrics['accuracy']*100:.2f}%")
    print(f"Overall Real-World Macro F1: {metrics['macro_f1']*100:.2f}%")
    print(f"Contextual Test Verdict: {'PASS' if pass_verdict else 'FAIL'}")

if __name__ == "__main__":
    main()
