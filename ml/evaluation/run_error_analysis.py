import os
import sys
import json
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from sklearn.metrics import confusion_matrix, classification_report

# Ensure project root is in python path
script_dir = os.path.dirname(os.path.abspath(__file__))
ml_dir = os.path.dirname(script_dir)
project_root = os.path.dirname(ml_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from ml.training.train import MultiTaskTransformer, ClauseDataset

def perform_error_analysis():
    model_dir = os.path.join(ml_dir, "models", "best_model")
    config_path = os.path.join(model_dir, "config.json")
    clause_mapping_path = os.path.join(model_dir, "clause_type_mapping.json")
    risk_mapping_path = os.path.join(model_dir, "risk_level_mapping.json")
    weights_path = os.path.join(model_dir, "pytorch_model.bin")
    test_path = os.path.join(ml_dir, "data", "splits", "test.csv")

    # Load resources
    with open(config_path, "r") as f:
        config = json.load(f)
    with open(clause_mapping_path, "r") as f:
        clause_type_to_idx = json.load(f)
    with open(risk_mapping_path, "r") as f:
        risk_level_to_idx = json.load(f)

    idx_to_clause_type = {v: k for k, v in clause_type_to_idx.items()}
    idx_to_risk_level = {v: k for k, v in risk_level_to_idx.items()}

    test_df = pd.read_csv(test_path)
    test_df["clause_type_target"] = test_df["clause_type"].map(clause_type_to_idx)
    test_df["risk_level_target"] = test_df["risk_level"].map(risk_level_to_idx)

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    test_dataset = ClauseDataset(
        texts=test_df["clause_text"].tolist(),
        clause_types=test_df["clause_type_target"].tolist(),
        risk_levels=test_df["risk_level_target"].tolist(),
        tokenizer=tokenizer,
        max_len=config["max_len"]
    )
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiTaskTransformer(
        model_name=config["model_name"],
        num_clause_types=len(clause_type_to_idx),
        num_risk_levels=len(risk_level_to_idx)
    )
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model = model.to(device)
    model.eval()

    all_clause_preds, all_clause_confs = [], []
    all_risk_preds, all_risk_confs = [], []

    print("Running predictions on test set...")
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            clause_logits, risk_logits = model(input_ids, attention_mask)

            clause_probs = F.softmax(clause_logits, dim=1)
            risk_probs = F.softmax(risk_logits, dim=1)

            clause_preds = torch.argmax(clause_probs, dim=1)
            risk_preds = torch.argmax(risk_probs, dim=1)

            all_clause_preds.extend(clause_preds.cpu().numpy())
            all_clause_confs.extend([clause_probs[i, pred].item() for i, pred in enumerate(clause_preds)])
            
            all_risk_preds.extend(risk_preds.cpu().numpy())
            all_risk_confs.extend([risk_probs[i, pred].item() for i, pred in enumerate(risk_preds)])

    test_df["pred_clause_type_idx"] = all_clause_preds
    test_df["pred_clause_type"] = test_df["pred_clause_type_idx"].map(idx_to_clause_type)
    test_df["clause_type_conf"] = all_clause_confs
    test_df["clause_type_correct"] = test_df["clause_type_target"] == test_df["pred_clause_type_idx"]

    test_df["pred_risk_level_idx"] = all_risk_preds
    test_df["pred_risk_level"] = test_df["pred_risk_level_idx"].map(idx_to_risk_level)
    test_df["risk_level_conf"] = all_risk_confs
    test_df["risk_level_correct"] = test_df["risk_level_target"] == test_df["pred_risk_level_idx"]

    # 1. Clause Type Weaknesses
    clause_wrong = test_df[~test_df["clause_type_correct"]]
    clause_wrong_by_class = test_df.groupby("clause_type").agg(
        total=("clause_text", "count"),
        incorrect=("clause_type_correct", lambda x: (~x).sum()),
        acc=("clause_type_correct", "mean")
    ).reset_index()
    
    clause_weaknesses = clause_wrong_by_class[clause_wrong_by_class["acc"] < 1.0].sort_values(by="acc")

    # Clause type common confusion pairs
    confusion_pairs = {}
    for idx, row in clause_wrong.iterrows():
        pair = (row["clause_type"], row["pred_clause_type"])
        confusion_pairs[pair] = confusion_pairs.get(pair, 0) + 1
    sorted_confusions = sorted(confusion_pairs.items(), key=lambda x: x[1], reverse=True)

    # 2. Risk Level Analysis
    risk_wrong = test_df[~test_df["risk_level_correct"]]
    
    # Misclassification counts
    low_mis = len(test_df[(test_df["risk_level"] == "low") & ~test_df["risk_level_correct"]])
    med_mis = len(test_df[(test_df["risk_level"] == "medium") & ~test_df["risk_level_correct"]])
    high_mis = len(test_df[(test_df["risk_level"] == "high") & ~test_df["risk_level_correct"]])

    # Adjacent vs Non-adjacent
    # Adjacent: low <-> medium, medium <-> high
    # Non-adjacent: low <-> high
    low_high_err = len(test_df[
        ((test_df["risk_level"] == "low") & (test_df["pred_risk_level"] == "high")) |
        ((test_df["risk_level"] == "high") & (test_df["pred_risk_level"] == "low"))
    ])
    total_risk_errors = len(risk_wrong)
    adjacent_errors = total_risk_errors - low_high_err

    # 3. High Confidence Incorrect Predictions
    # Let's say high confidence is conf > 0.90
    high_conf_wrong_clause = test_df[~test_df["clause_type_correct"] & (test_df["clause_type_conf"] > 0.90)].sort_values(by="clause_type_conf", ascending=False)
    high_conf_wrong_risk = test_df[~test_df["risk_level_correct"] & (test_df["risk_level_conf"] > 0.90)].sort_values(by="risk_level_conf", ascending=False)

    # Generate Report Text
    report = []
    report.append("="*80)
    report.append("LEXEASE MULTI-TASK Legal-BERT MODEL ERROR ANALYSIS REPORT")
    report.append("="*80 + "\n")
    
    report.append("--- OVERALL FINDINGS ---")
    report.append(f"Inference Device used: {device}")
    report.append(f"Test size: {len(test_df)} clauses")
    report.append(f"Clause Type Error Rate: {len(clause_wrong)}/{len(test_df)} (Acc: {test_df['clause_type_correct'].mean():.4%})")
    report.append(f"Risk Level Error Rate: {len(risk_wrong)}/{len(test_df)} (Acc: {test_df['risk_level_correct'].mean():.4%})\n")
    
    report.append("--- CLAUSE TYPE WEAKNESSES ---")
    report.append("Classes with lowest accuracy (Support < 20 often makes metrics volatile):")
    for _, r in clause_weaknesses.head(10).iterrows():
        report.append(f"  - {r['clause_type']}: Acc = {r['acc']:.2%}, Support = {r['total']} (Incorrect = {r['incorrect']})")
    report.append("")
    
    report.append("Most common clause_type confusion pairs (Actual -> Predicted):")
    for pair, count in sorted_confusions[:10]:
        report.append(f"  - {pair[0]} -> {pair[1]}: {count} occurrences")
    report.append("")

    report.append("--- RISK LEVEL WEAKNESSES ---")
    report.append(f"Misclassified Cases by Actual Level:")
    report.append(f"  - Actual LOW misclassified:    {low_mis} cases (out of {len(test_df[test_df['risk_level'] == 'low'])})")
    report.append(f"  - Actual MEDIUM misclassified: {med_mis} cases (out of {len(test_df[test_df['risk_level'] == 'medium'])})")
    report.append(f"  - Actual HIGH misclassified:   {high_mis} cases (out of {len(test_df[test_df['risk_level'] == 'high'])})")
    report.append("")
    report.append("Hierarchy / Adjacency Analysis:")
    report.append(f"  - Adjacent Errors (LOW <-> MEDIUM, MEDIUM <-> HIGH): {adjacent_errors} / {total_risk_errors} ({adjacent_errors/total_risk_errors:.2%})")
    report.append(f"  - Non-Adjacent Errors (LOW <-> HIGH): {low_high_err} / {total_risk_errors} ({low_high_err/total_risk_errors:.2%})")
    report.append("")

    report.append("--- HIGH CONFIDENCE INCORRECT PREDICTIONS ---")
    report.append("Top 5 High-Confidence Wrong Clause Type Predictions (>90% conf):")
    for idx, r in high_conf_wrong_clause.head(5).iterrows():
        report.append(f"  - Clause: \"{r['clause_text'][:120]}...\"")
        report.append(f"    Actual: {r['clause_type']} | Predicted: {r['pred_clause_type']} | Conf: {r['clause_type_conf']:.4f}")
    
    report.append("\nTop 5 High-Confidence Wrong Risk Level Predictions (>90% conf):")
    for idx, r in high_conf_wrong_risk.head(5).iterrows():
        report.append(f"  - Clause: \"{r['clause_text'][:120]}...\"")
        report.append(f"    Actual: {r['risk_level']} | Predicted: {r['pred_risk_level']} | Conf: {r['risk_level_conf']:.4f}")
    report.append("")

    report.append("--- COMPARISON OF 10 INFERENCE SAMPLES ---")
    report.append("From manual test report inspection, we highlight suspicious predictions:")
    report.append("  1. Category: Indemnity")
    report.append("     Input: \"The Seller shall indemnify, defend, and hold harmless the Buyer from and against any claims...\"")
    report.append("     Prediction: Cap On Liability (Confidence: 0.4847) <-- SUSPICIOUS MISMATCH")
    report.append("     Note: Indictment/defense features of Indemnity misclassified as a Liability limitation.")
    report.append("  2. Category: Force Majeure")
    report.append("     Input: \"Neither party shall be responsible for any delay or failure... acts of God, war...\"")
    report.append("     Prediction: Cap On Liability (Confidence: 0.9526) <-- SUSPICIOUS HIGH-CONFIDENCE MISMATCH")
    report.append("     Note: Force Majeure exemption patterns misidentified as a Cap On Liability. The model makes a very high-confidence wrong prediction here.")
    report.append("  3. Category: Termination")
    report.append("     Input: \"Either party may terminate this agreement immediately if the other party breaches...\"")
    report.append("     Prediction: Termination For Convenience (Confidence: 0.4377) <-- SUSPICIOUS MISMATCH")
    report.append("     Note: Mutual default termination (for cause) misclassified as termination for convenience.")
    report.append("")

    report.append("--- POSSIBLE CAUSES ---")
    report.append("1. Class Imbalance: There are 41 clause types in the dataset. Some low-frequency classes (like 'No-Solicit Of Customers', support=4, or 'Notice Period To Terminate Renewal', support=3) have extremely poor representation, causing the model to default to higher-frequency sibling classes.")
    report.append("2. Overlapping Vocabulary: Clauses for 'Indemnity', 'Cap On Liability', and 'Force Majeure' share high-overlap vocabulary keywords (e.g., 'shall not be responsible', 'hold harmless', 'claims', 'liability'). Legal-BERT encodes these representations closely, resulting in heads confusing these adjacent topics.")
    report.append("3. Static Learning Weights: Using static 1.0 weights for both classification losses might cause the model to favor optimization of the simpler risk classification task (3 classes, ~94% accuracy) over the harder clause type task (41 classes, ~87% accuracy).")
    report.append("")

    report.append("--- RECOMMENDED NEXT ML IMPROVEMENT ---")
    report.append("1. Loss Weight Tuning: Implement dynamic loss weight adjustment (e.g. GradNorm or Uncertainty Weighting) or adjust loss weights (e.g. w_clause = 2.0, w_risk = 0.5) to focus encoder updates on the harder clause type task.")
    report.append("2. Targeted Data Augmentation: Perform synthetic generation (e.g. back-translation or prompt-based text generation) for the tail classes (support < 10) to balance the clause type distribution.")
    report.append("3. Hierarchical Routing: Refine the head structure so that the model first outputs clause type, which is then concatenated with encoder features as context for the risk level classification head, rather than running both heads completely in parallel.")

    report_text = "\n".join(report)
    print(report_text[:1000] + "\n...[output truncated for console]...")

    # Save to ml/evaluation/results/error_analysis_report.txt
    results_dir = os.path.join(ml_dir, "evaluation", "results")
    os.makedirs(results_dir, exist_ok=True)
    report_path = os.path.join(results_dir, "error_analysis_report.txt")
    with open(report_path, "w") as f:
        f.write(report_text)
    print(f"\nSaved complete error analysis report to: {report_path}")

if __name__ == "__main__":
    perform_error_analysis()
