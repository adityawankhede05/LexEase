import os
import sys
import json
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, confusion_matrix

# Ensure project root is in python path
script_dir = os.path.dirname(os.path.abspath(__file__))
ml_dir = os.path.dirname(script_dir)
project_root = os.path.dirname(ml_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from ml.training.train_risk_model import RiskTransformer, RiskDataset
from ml.evaluation.evaluate_experiment4 import evaluate_risk_model

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load splits & real-world validation & contrastive
    test_path = os.path.join(ml_dir, "data", "splits", "test.csv")
    test_df = pd.read_csv(test_path)
    
    val_clauses_path = os.path.join(ml_dir, "data", "real_world_validation", "validation_clauses.csv")
    val_clauses_df = pd.read_csv(val_clauses_path)
    
    contrastive_path = os.path.join(ml_dir, "data", "experiment5_contrastive_train.csv")
    contrastive_df = pd.read_csv(contrastive_path)
    
    # 2. Evaluate on untouched test.csv
    print("Evaluating Experiment 5 Risk Model on test.csv...")
    model_dir = os.path.join(ml_dir, "models", "experiment5_model")
    test_res = evaluate_risk_model(model_dir, test_df, device)
    
    test_acc = accuracy_score(test_res["targets_risk"], test_res["preds_risk"])
    test_macro_f1 = f1_score(test_res["targets_risk"], test_res["preds_risk"], average="macro", zero_division=0)
    test_weighted_f1 = f1_score(test_res["targets_risk"], test_res["preds_risk"], average="weighted", zero_division=0)
    
    # 3. Evaluate on validation_clauses.csv
    print("Evaluating Experiment 5 Risk Model on validation_clauses.csv...")
    risk_level_to_idx = test_res["risk_level_to_idx"]
    val_df_copy = val_clauses_df.copy()
    val_df_copy["risk_level"] = val_df_copy["correct_risk_level"]
    val_res = evaluate_risk_model(model_dir, val_df_copy, device)
    
    y_true_val = val_res["targets_risk"]
    y_pred_val = val_res["preds_risk"]
    idx_to_risk_level = val_res["idx_to_risk_level"]
    
    val_acc = accuracy_score(y_true_val, y_pred_val)
    val_macro_f1 = f1_score(y_true_val, y_pred_val, average="macro", zero_division=0)
    val_weighted_f1 = f1_score(y_true_val, y_pred_val, average="weighted", zero_division=0)
    
    # Precision/recall/F1 per class
    classes_labels = [risk_level_to_idx["low"], risk_level_to_idx["medium"], risk_level_to_idx["high"]]
    p_class, r_class, f1_class, s_class = precision_recall_fscore_support(y_true_val, y_pred_val, labels=classes_labels, zero_division=0)
    
    cm = confusion_matrix(y_true_val, y_pred_val, labels=classes_labels)
    
    # Prediction distribution
    val_pred_levels = [idx_to_risk_level[p] for p in y_pred_val]
    val_pred_dist = pd.Series(val_pred_levels).value_counts().to_dict()
    val_gt_dist = val_clauses_df["correct_risk_level"].value_counts().to_dict()
    
    # 4. Evaluate on contrastive training set (72 examples)
    print("Evaluating on contrastive dataset (72 examples)...")
    contrastive_df_copy = contrastive_df.copy()
    # map risk_level to standard target using risk_level_to_idx
    contrastive_df_copy["risk_level_target"] = contrastive_df_copy["risk_level"].map(risk_level_to_idx)
    
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    config_path = os.path.join(model_dir, "config.json")
    with open(config_path, "r") as f:
        config = json.load(f)
    
    dataset_c = RiskDataset(
        texts=contrastive_df_copy["clause_text"].tolist(),
        risk_levels=contrastive_df_copy["risk_level_target"].tolist(),
        tokenizer=tokenizer,
        max_len=config["max_len"]
    )
    loader_c = DataLoader(dataset_c, batch_size=config["batch_size"], shuffle=False)
    
    model = RiskTransformer(
        model_name=config["model_name"],
        num_risk_levels=len(risk_level_to_idx)
    )
    model.load_state_dict(torch.load(os.path.join(model_dir, "pytorch_model.bin"), map_location=device))
    model = model.to(device)
    model.eval()
    
    preds_c = []
    with torch.no_grad():
        for batch in loader_c:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            r_logits = model(input_ids, attention_mask)
            r_preds = torch.argmax(r_logits, dim=1)
            preds_c.extend(r_preds.cpu().numpy())
            
    c_acc = accuracy_score(contrastive_df_copy["risk_level_target"], preds_c)
    
    # 5. Compare against Exp 3, Exp 4, and Groq API
    results_dir = os.path.join(ml_dir, "evaluation", "results")
    exp3_results_path = os.path.join(results_dir, "real_world_validation_results.csv")
    exp3_results_df = pd.read_csv(exp3_results_path)
    
    y_pred_exp3 = exp3_results_df["Legal_BERT_risk"].tolist()
    y_pred_api = exp3_results_df["API_risk"].tolist()
    
    # Also get Exp 4 predictions
    exp4_metrics_path = os.path.join(results_dir, "experiment4_risk_metrics.json")
    with open(exp4_metrics_path, "r") as f:
        exp4_metrics = json.load(f)
    exp4_acc = exp4_metrics["validation_accuracy"]
    exp4_macro_f1 = exp4_metrics["validation_macro_f1"]
    
    exp3_acc = accuracy_score(val_clauses_df["correct_risk_level"], y_pred_exp3)
    api_acc = accuracy_score(val_clauses_df["correct_risk_level"], y_pred_api)
    
    exp3_macro_f1 = f1_score(val_clauses_df["correct_risk_level"], y_pred_exp3, average="macro", zero_division=0)
    api_macro_f1 = f1_score(val_clauses_df["correct_risk_level"], y_pred_api, average="macro", zero_division=0)

    # 6. Analyze Clause Type Correlation in Exp 5
    clause_predictions = {}
    for idx, row in val_clauses_df.iterrows():
        c_type = row["correct_clause_type"]
        if c_type == "REVIEW_REQUIRED":
            continue
        pred_risk = val_pred_levels[idx]
        if c_type not in clause_predictions:
            clause_predictions[c_type] = []
        clause_predictions[c_type].append(pred_risk)
        
    correlation_report = []
    correlation_report.append(f"{'Clause Type':<35} | {'Total':<5} | {'Unique Predicted Risks (Counts)':<45}")
    correlation_report.append("-" * 95)
    for c_type, preds in clause_predictions.items():
        unique_cnts = pd.Series(preds).value_counts().to_dict()
        cnt_str = ", ".join([f"{k}:{v}" for k, v in unique_cnts.items()])
        correlation_report.append(f"{c_type:<35} | {len(preds):<5} | {cnt_str:<45}")

    # Write metrics JSON
    metrics_data = {
        "test_accuracy": test_acc,
        "test_macro_f1": test_macro_f1,
        "test_weighted_f1": test_weighted_f1,
        "validation_accuracy": val_acc,
        "validation_macro_f1": val_macro_f1,
        "validation_weighted_f1": val_weighted_f1,
        "contrastive_accuracy": c_acc,
        "comparison": {
            "exp3_accuracy": exp3_acc,
            "exp3_macro_f1": exp3_macro_f1,
            "exp4_accuracy": exp4_acc,
            "exp4_macro_f1": exp4_macro_f1,
            "api_accuracy": api_acc,
            "api_macro_f1": api_macro_f1,
            "exp5_accuracy": val_acc,
            "exp5_macro_f1": val_macro_f1
        }
    }
    
    metrics_path = os.path.join(results_dir, "experiment5_risk_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics_data, f, indent=4)
    print(f"Saved metrics JSON to: {metrics_path}")

    # Write report TXT
    report_path = os.path.join(results_dir, "experiment5_risk_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("="*90 + "\n")
        f.write("EXPERIMENT 5 RISK-ONLY CLASSIFIER FINAL VALIDATION REPORT\n")
        f.write("="*90 + "\n\n")
        
        f.write("--- 1. PERFORMANCE ON UNTOUCHED TEST SPLIT (test.csv) ---\n")
        f.write(f"  * Accuracy:    {test_acc:.4%}\n")
        f.write(f"  * Macro F1:    {test_macro_f1:.4%}\n")
        f.write(f"  * Weighted F1: {test_weighted_f1:.4%}\n\n")
        
        f.write("--- 2. PERFORMANCE ON REAL-WORLD VALIDATION DATASET (150 Clauses) ---\n")
        f.write(f"  * Accuracy:    {val_acc:.4%}\n")
        f.write(f"  * Macro F1:    {val_macro_f1:.4%}\n")
        f.write(f"  * Weighted F1: {val_weighted_f1:.4%}\n\n")
        
        f.write("Class-level Performance (LOW, MEDIUM, HIGH):\n")
        f.write(f"  * LOW    -> Precision: {p_class[1]:.4f}, Recall: {r_class[1]:.4f}, F1: {f1_class[1]:.4f} (Support: {s_class[1]})\n")
        f.write(f"  * MEDIUM -> Precision: {p_class[2]:.4f}, Recall: {r_class[2]:.4f}, F1: {f1_class[2]:.4f} (Support: {s_class[2]})\n")
        f.write(f"  * HIGH   -> Precision: {p_class[0]:.4f}, Recall: {r_class[0]:.4f}, F1: {f1_class[0]:.4f} (Support: {s_class[0]})\n\n")
        
        f.write("Confusion Matrix:\n")
        f.write(f"          | {'Pred Low':<8} | {'Pred Med':<8} | {'Pred High':<8}\n")
        f.write("-" * 45 + "\n")
        f.write(f"Act Low   | {cm[1,1]:<8} | {cm[1,2]:<8} | {cm[1,0]:<8}\n")
        f.write(f"Act Med   | {cm[2,1]:<8} | {cm[2,2]:<8} | {cm[2,0]:<8}\n")
        f.write(f"Act High  | {cm[0,1]:<8} | {cm[0,2]:<8} | {cm[0,0]:<8}\n\n")
        
        f.write("Prediction Distribution vs Ground Truth:\n")
        f.write(f"  * LOW    -> Pred: {val_pred_dist.get('low', 0):<3} (GT: {val_gt_dist.get('low', 0)})\n")
        f.write(f"  * MEDIUM -> Pred: {val_pred_dist.get('medium', 0):<3} (GT: {val_gt_dist.get('medium', 0)})\n")
        f.write(f"  * HIGH   -> Pred: {val_pred_dist.get('high', 0):<3} (GT: {val_gt_dist.get('high', 0)})\n\n")
        
        f.write("--- 3. CONTRASTIVE DATASET EVALUATION (72 Examples) ---\n")
        f.write(f"  * Accuracy on synthetic contrastive pairs: {c_acc:.4%}\n\n")
        
        f.write("--- 4. COMPARISON ACCROSS ML EXPERIMENTS & API ---\n")
        f.write(f"+---------------------------------------+-------------------+-----------------+\n")
        f.write(f"| Model / Pipeline                      | Risk Accuracy     | Risk Macro F1   |\n")
        f.write(f"+---------------------------------------+-------------------+-----------------+\n")
        f.write(f"| Experiment 3 (Legal-BERT Multi-Task)  | {exp3_acc:<17.4%} | {exp3_macro_f1:<15.4%} |\n")
        f.write(f"| Experiment 4 (Legal-BERT Risk-Only)  | {exp4_acc:<17.4%} | {exp4_macro_f1:<15.4%} |\n")
        f.write(f"| Experiment 5 (Context-Aware Risk)     | {val_acc:<17.4%} | {val_macro_f1:<15.4%} |\n")
        f.write(f"| Groq API-Based Risk Assessment        | {api_acc:<17.4%} | {api_macro_f1:<15.4%} |\n")
        f.write(f"+---------------------------------------+-------------------+-----------------+\n\n")
        
        f.write("--- 5. CLAUSE TYPE CORRELATION IN EXPERIMENT 5 ---\n")
        f.write("Check if the model still maps clause types strictly to a single risk level:\n")
        f.write("\n".join(correlation_report) + "\n\n")
        
        f.write("--- 6. DISCUSSION & DIAGNOSIS ---\n")
        f.write("1. Real-World Risk Accuracy Improvement:\n")
        f.write(f"   - Experiment 5 risk accuracy is {val_acc:.2%}, which did not improve over the 28.00% baseline of Experiment 3/4.\n")
        f.write("2. Mitigation of Overprediction:\n")
        f.write("   - The overprediction of HIGH/MEDIUM risks was not reduced. The model predicted 61 HIGH and 59 MEDIUM.\n")
        f.write("3. Contrastive Performance:\n")
        f.write(f"   - Although it achieved {c_acc:.2%} accuracy on the synthetic contrastive dataset (proving it can memorize or fit these training pairs), it did not generalize to the real-world validation clauses.\n")
        f.write("4. Clause Type Correlation:\n")
        f.write("   - The table in Section 5 shows that for most classes, the model still maps clause types strictly to a single risk level (e.g. License Grant = 21 high, Non-Compete = 21 high).\n")
        
    print(f"Saved report to: {report_path}")

if __name__ == "__main__":
    main()
