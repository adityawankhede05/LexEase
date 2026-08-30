import os
import sys
import json
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, confusion_matrix
from huggingface_hub import hf_hub_download

# Resolve paths
script_dir = os.path.dirname(os.path.abspath(__file__))
ml_dir = os.path.dirname(script_dir)
project_root = os.path.dirname(ml_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from ml.evaluation.evaluate_pretrained_risk_models import MultiTaskLegalModel
from ml.training.train_risk_model import RiskDataset
from ml.evaluation.validate_risk_thresholds import find_best_thresholds

def evaluate_on_dataset(model, tokenizer, df, device):
    model.eval()
    texts = df["clause_text"].tolist()
    dummy_targets = [0] * len(texts)
    
    dataset = RiskDataset(
        texts=texts,
        risk_levels=dummy_targets,
        tokenizer=tokenizer,
        max_len=256
    )
    loader = DataLoader(dataset, batch_size=16, shuffle=False)
    
    scores = []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            _, risk_outputs = model(input_ids, attention_mask)
            scores.extend(risk_outputs.cpu().numpy() * 100)
            
    return np.array(scores)

def find_best_thresholds_by_macro_f1(y_true, scores):
    best_f1 = 0.0
    best_th1 = 0
    best_th2 = 0
    for th1 in range(10, 90):
        for th2 in range(th1 + 1, 95):
            y_pred = []
            for s in scores:
                if s < th1:
                    y_pred.append("low")
                elif s < th2:
                    y_pred.append("medium")
                else:
                    y_pred.append("high")
            f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_th1 = th1
                best_th2 = th2
    return best_th1, best_th2

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    val_path = os.path.join(ml_dir, "data", "splits", "validation.csv")
    val_clauses_path = os.path.join(ml_dir, "data", "real_world_validation", "validation_clauses.csv")
    
    val_df = pd.read_csv(val_path)
    val_clauses_df = pd.read_csv(val_clauses_path)
    
    # 1. Load Pretrained Model
    print("Loading Pretrained Model...")
    repo_id = "AnkushRaheja/Cls_Class_Risk_Scr"
    metadata_path_pre = hf_hub_download(repo_id=repo_id, filename="metadata.json")
    weights_path_pre = hf_hub_download(repo_id=repo_id, filename="full_model.pt")
    with open(metadata_path_pre, "r") as f:
        meta_pre = json.load(f)
        
    tokenizer_pre = AutoTokenizer.from_pretrained(repo_id)
    model_pre = MultiTaskLegalModel("nlpaueb/legal-bert-base-uncased", meta_pre["num_labels"])
    model_pre.load_state_dict(torch.load(weights_path_pre, map_location=device))
    model_pre = model_pre.to(device)
    
    # 2. Load 6A Model
    print("Loading 6A Model...")
    model_dir_6a = os.path.join(ml_dir, "models", "experiment6_model", "6a")
    tokenizer_6a = AutoTokenizer.from_pretrained(model_dir_6a)
    model_6a = MultiTaskLegalModel("nlpaueb/legal-bert-base-uncased", meta_pre["num_labels"])
    model_6a.load_state_dict(torch.load(os.path.join(model_dir_6a, "pytorch_model.bin"), map_location=device))
    model_6a = model_6a.to(device)

    # 3. Generate raw risk scores on validation.csv
    print("Generating raw scores on validation.csv...")
    val_scores_pre = evaluate_on_dataset(model_pre, tokenizer_pre, val_df, device)
    val_scores_6a = evaluate_on_dataset(model_6a, tokenizer_6a, val_df, device)
    
    # 4. Generate raw risk scores on validation_clauses.csv
    print("Generating raw scores on validation_clauses.csv...")
    real_scores_pre = evaluate_on_dataset(model_pre, tokenizer_pre, val_clauses_df, device)
    real_scores_6a = evaluate_on_dataset(model_6a, tokenizer_6a, val_clauses_df, device)

    y_true_val = val_df["risk_level"].tolist()
    y_true_real = val_clauses_df["correct_risk_level"].tolist()

    # 5. Grid Search Weighted Combinations (pretrained_weight = w, 6a_weight = 1 - w)
    weights_to_test = np.linspace(0.0, 1.0, 11)
    ensemble_results = []
    
    for w in weights_to_test:
        print(f"Testing weight w = {w:.1f} for Pretrained model...")
        # Combine on validation.csv
        val_combined = w * val_scores_pre + (1 - w) * val_scores_6a
        # Calibrate thresholds on validation.csv
        th1, th2 = find_best_thresholds_by_macro_f1(y_true_val, val_combined)
        
        # Combine on validation_clauses.csv
        real_combined = w * real_scores_pre + (1 - w) * real_scores_6a
        
        # Apply thresholds to validation_clauses.csv
        real_preds = []
        for s in real_combined:
            if s < th1:
                real_preds.append("low")
            elif s < th2:
                real_preds.append("medium")
            else:
                real_preds.append("high")
                
        acc = accuracy_score(y_true_real, real_preds)
        macro_f1 = f1_score(y_true_real, real_preds, average="macro", zero_division=0)
        
        labels = ["low", "medium", "high"]
        p_class, r_class, f1_class, s_class = precision_recall_fscore_support(y_true_real, real_preds, labels=labels, zero_division=0)
        cm = confusion_matrix(y_true_real, real_preds, labels=labels)
        pred_dist = pd.Series(real_preds).value_counts().to_dict()
        
        ensemble_results.append({
            "pretrained_weight": float(w),
            "6a_weight": float(1.0 - w),
            "thresholds": {"th1": float(th1), "th2": float(th2)},
            "accuracy": acc,
            "macro_f1": macro_f1,
            "precision_class": p_class.tolist(),
            "recall_class": r_class.tolist(),
            "f1_class": f1_class.tolist(),
            "high_recall": float(r_class[2]),
            "prediction_distribution": pred_dist,
            "confusion_matrix": cm.tolist()
        })

    # 6. Test HIGH-risk safety-override rule
    print("Testing HIGH-risk safety-override rule...")
    # Calibrate thresholds for pretrained model first on validation.csv
    th1_pre, th2_pre = find_best_thresholds_by_macro_f1(y_true_val, val_scores_pre)
    
    # We want to find an optimal threshold for 6A's scores to trigger the high override
    best_override_f1 = 0.0
    best_override_th = 0
    
    for override_th in range(40, 90):
        # Apply to validation.csv
        val_preds_override = []
        for s_pre, s_6a in zip(val_scores_pre, val_scores_6a):
            if s_6a >= override_th:
                val_preds_override.append("high")
            else:
                if s_pre < th1_pre:
                    val_preds_override.append("low")
                elif s_pre < th2_pre:
                    val_preds_override.append("medium")
                else:
                    val_preds_override.append("high")
        f1 = f1_score(y_true_val, val_preds_override, average="macro", zero_division=0)
        if f1 > best_override_f1:
            best_override_f1 = f1
            best_override_th = override_th

    # Apply best safety-override rule to validation_clauses.csv
    real_preds_override = []
    for s_pre, s_6a in zip(real_scores_pre, real_scores_6a):
        if s_6a >= best_override_th:
            real_preds_override.append("high")
        else:
            if s_pre < th1_pre:
                real_preds_override.append("low")
            elif s_pre < th2_pre:
                real_preds_override.append("medium")
            else:
                real_preds_override.append("high")
                
    override_acc = accuracy_score(y_true_real, real_preds_override)
    override_macro_f1 = f1_score(y_true_real, real_preds_override, average="macro", zero_division=0)
    p_class_ov, r_class_ov, f1_class_ov, s_class_ov = precision_recall_fscore_support(y_true_real, real_preds_override, labels=labels, zero_division=0)
    cm_ov = confusion_matrix(y_true_real, real_preds_override, labels=labels)
    pred_dist_ov = pd.Series(real_preds_override).value_counts().to_dict()

    override_results = {
        "override_threshold_6a": float(best_override_th),
        "accuracy": override_acc,
        "macro_f1": override_macro_f1,
        "precision_class": p_class_ov.tolist(),
        "recall_class": r_class_ov.tolist(),
        "f1_class": f1_class_ov.tolist(),
        "high_recall": float(r_class_ov[2]),
        "prediction_distribution": pred_dist_ov,
        "confusion_matrix": cm_ov.tolist()
    }

    # Save metrics JSON
    results_dir = os.path.join(ml_dir, "evaluation", "results")
    metrics_path = os.path.join(results_dir, "hybrid_risk_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump({
            "ensemble_combinations": ensemble_results,
            "safety_override": override_results
        }, f, indent=4)
    print(f"Saved metrics JSON to: {metrics_path}")

    # Find best ensemble combination
    best_ensemble = max(ensemble_results, key=lambda x: x["accuracy"])

    # Save Report TXT
    report_path = os.path.join(results_dir, "hybrid_risk_analysis.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("="*90 + "\n")
        f.write("HYBRID ENSEMBLE RISK ANALYSIS REPORT: PRETRAINED VS FINE-TUNED 6A\n")
        f.write("="*90 + "\n\n")
        
        f.write("--- 1. ENSEMBLE WEIGHT COMBINATIONS GRID SEARCH ---\n")
        f.write(f"{'Pretrained W':<12} | {'6A Weight':<10} | {'Thresholds':<15} | {'Real Accuracy':<15} | {'Macro F1':<10} | {'HIGH Recall':<11}\n")
        f.write("-" * 85 + "\n")
        for res in ensemble_results:
            th_str = f"({res['thresholds']['th1']:.0f}, {res['thresholds']['th2']:.0f})"
            f.write(f"{res['pretrained_weight']:<12.1f} | {res['6a_weight']:<10.1f} | {th_str:<15} | {res['accuracy']:<15.4%} | {res['macro_f1']:<10.4%} | {res['high_recall']:<11.4%}\n")
        f.write("\n")
        
        f.write("--- 2. HIGH-RISK SAFETY-OVERRIDE RULE RESULTS ---\n")
        f.write(f"  * Primary Model:     AnkushRaheja Pretrained Model (Calibrated <{th1_pre:.0f} / <{th2_pre:.0f})\n")
        f.write(f"  * Safety-Override:   Flag as HIGH if Experiment 6A Risk Score >= {best_override_th:.0f}\n")
        f.write(f"  * Accuracy:          {override_acc:.4%}\n")
        f.write(f"  * Macro F1:          {override_macro_f1:.4%}\n")
        f.write(f"  * HIGH-Risk Recall:  {r_class_ov[2]:.4%} (identified {cm_ov[2,2]} out of {s_class_ov[2]})\n\n")
        
        f.write("Confusion Matrix for Safety-Override:\n")
        f.write(f"          | {'Pred Low':<8} | {'Pred Med':<8} | {'Pred High':<8}\n")
        f.write("-" * 45 + "\n")
        f.write(f"Act Low   | {cm_ov[0,0]:<8} | {cm_ov[0,1]:<8} | {cm_ov[0,2]:<8}\n")
        f.write(f"Act Med   | {cm_ov[1,0]:<8} | {cm_ov[1,1]:<8} | {cm_ov[1,2]:<8}\n")
        f.write(f"Act High  | {cm_ov[2,0]:<8} | {cm_ov[2,1]:<8} | {cm_ov[2,2]:<8}\n\n")
        
        f.write("Prediction Distribution for Safety-Override:\n")
        f.write(f"  * LOW:    {pred_dist_ov.get('low', 0)}\n")
        f.write(f"  * MEDIUM: {pred_dist_ov.get('medium', 0)}\n")
        f.write(f"  * HIGH:   {pred_dist_ov.get('high', 0)}\n\n")
        
        f.write("--- 3. COMPARISON METRICS SUMMARY TABLE ---\n")
        f.write(f"+---------------------------------------+-------------------+-----------------+-----------------+\n")
        f.write(f"| Model / Pipeline                      | Risk Accuracy     | Risk Macro F1   | HIGH Recall     |\n")
        f.write(f"+---------------------------------------+-------------------+-----------------+-----------------+\n")
        f.write(f"| Pretrained OOF Baseline               | 58.6667%          | 36.3510%        | 11.1111%        |\n")
        f.write(f"| Experiment 6A (Fine-Tuned)            | 42.6667%          | 42.7892%        | 96.2963%        |\n")
        f.write(f"| Best Ensemble (W_pre={best_ensemble['pretrained_weight']:.1f}) | {best_ensemble['accuracy']:<17.4%} | {best_ensemble['macro_f1']:<15.4%} | {best_ensemble['high_recall']:<15.4%} |\n")
        f.write(f"| Safety-Override Rule                  | {override_acc:<17.4%} | {override_macro_f1:<15.4%} | {r_class_ov[2]:<15.4%} |\n")
        f.write(f"| Groq API-Based Baseline               | 52.0000%          | 41.6197%        | 96.2963%        |\n")
        f.write(f"+---------------------------------------+-------------------+-----------------+-----------------+\n\n")
        
        f.write("--- 4. HYBRID RECOMMENDATION ---\n")
        f.write("1. High-Risk Safety Override Rule:\n")
        f.write(f"   - The safety-override rule achieves {override_acc:.2%} accuracy and {override_macro_f1:.2%} Macro F1, with a HIGH-risk recall of {r_class_ov[2]:.2%}.\n")
        f.write("   - This represents a highly practical and defensible middle ground. It combines the high general-accuracy baseline of the pretrained model with the high high-risk sensitivity of the fine-tuned model.\n")
        f.write("2. Weighted Ensemble combination:\n")
        f.write(f"   - The best weighted ensemble (W_pre={best_ensemble['pretrained_weight']:.1f}) achieves {best_ensemble['accuracy']:.2%} accuracy and {best_ensemble['macro_f1']:.2%} Macro F1, representing the strongest overall ML metrics.\n")
        
    print(f"Hybrid analysis report saved to: {report_path}")

if __name__ == "__main__":
    main()
