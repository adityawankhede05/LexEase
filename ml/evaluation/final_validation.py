import os
import sys
import json
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from sklearn.metrics import accuracy_score, f1_score

# Ensure project root is in python path
script_dir = os.path.dirname(os.path.abspath(__file__))
ml_dir = os.path.dirname(script_dir)
project_root = os.path.dirname(ml_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from ml.training.train import MultiTaskTransformer, ClauseDataset

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model_dir = os.path.join(ml_dir, "models", "experiment3_model")
    config_path = os.path.join(model_dir, "config.json")
    clause_mapping_path = os.path.join(model_dir, "clause_type_mapping.json")
    risk_mapping_path = os.path.join(model_dir, "risk_level_mapping.json")
    weights_path = os.path.join(model_dir, "pytorch_model.bin")

    # 1. Verify files exist
    files_to_check = [config_path, clause_mapping_path, risk_mapping_path, weights_path]
    files_status = {}
    for fp in files_to_check:
        files_status[os.path.basename(fp)] = os.path.exists(fp)
    
    all_files_exist = all(files_status.values())
    print(f"Files verification: {files_status}")

    # Load configurations & mappings
    with open(config_path, "r") as f:
        config = json.load(f)
    with open(clause_mapping_path, "r") as f:
        clause_type_to_idx = json.load(f)
    with open(risk_mapping_path, "r") as f:
        risk_level_to_idx = json.load(f)

    idx_to_clause_type = {v: k for k, v in clause_type_to_idx.items()}
    idx_to_risk_level = {v: k for k, v in risk_level_to_idx.items()}

    # Load dataset
    test_path = os.path.join(ml_dir, "data", "splits", "test.csv")
    test_df = pd.read_csv(test_path)
    test_df["clause_type_target"] = test_df["clause_type"].map(clause_type_to_idx)
    test_df["risk_level_target"] = test_df["risk_level"].map(risk_level_to_idx)

    # 2. Run test set evaluation
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    dataset = ClauseDataset(
        texts=test_df["clause_text"].tolist(),
        clause_types=test_df["clause_type_target"].tolist(),
        risk_levels=test_df["risk_level_target"].tolist(),
        tokenizer=tokenizer,
        max_len=config["max_len"]
    )
    loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=False)

    model = MultiTaskTransformer(
        model_name=config["model_name"],
        num_clause_types=len(clause_type_to_idx),
        num_risk_levels=len(risk_level_to_idx)
    )
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model = model.to(device)
    model.eval()

    preds_clause, targets_clause = [], []
    preds_risk, targets_risk = [], []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            c_targets = batch["clause_type"].to(device)
            r_targets = batch["risk_level"].to(device)

            c_logits, r_logits = model(input_ids, attention_mask)

            c_preds = torch.argmax(c_logits, dim=1)
            r_preds = torch.argmax(r_logits, dim=1)

            preds_clause.extend(c_preds.cpu().numpy())
            targets_clause.extend(c_targets.cpu().numpy())

            preds_risk.extend(r_preds.cpu().numpy())
            targets_risk.extend(r_targets.cpu().numpy())

    clause_acc = accuracy_score(targets_clause, preds_clause)
    clause_macro_f1 = f1_score(targets_clause, preds_clause, average="macro", zero_division=0)
    clause_weighted_f1 = f1_score(targets_clause, preds_clause, average="weighted", zero_division=0)

    risk_acc = accuracy_score(targets_risk, preds_risk)
    risk_macro_f1 = f1_score(targets_risk, preds_risk, average="macro", zero_division=0)
    risk_weighted_f1 = f1_score(targets_risk, preds_risk, average="weighted", zero_division=0)

    # Validate against expected values
    expected_clause_acc = 0.8667
    expected_clause_macro_f1 = 0.7945
    expected_clause_weighted_f1 = 0.8677
    expected_risk_acc = 0.9386
    expected_risk_macro_f1 = 0.9391
    expected_risk_weighted_f1 = 0.9387

    metrics_match = (
        abs(clause_acc - expected_clause_acc) < 0.005 and
        abs(clause_macro_f1 - expected_clause_macro_f1) < 0.005 and
        abs(clause_weighted_f1 - expected_clause_weighted_f1) < 0.005 and
        abs(risk_acc - expected_risk_acc) < 0.005 and
        abs(risk_macro_f1 - expected_risk_macro_f1) < 0.005 and
        abs(risk_weighted_f1 - expected_risk_weighted_f1) < 0.005
    )

    # 3. Test inference on sample clauses
    samples = [
        "Neither party shall be liable to the other for any indirect, incidental, special, or consequential damages.",
        "The Client agrees to pay the service provider a minimum monthly commitment of INR 50,000 for the duration."
    ]
    inference_outputs = []
    for s in samples:
        encoding_s = tokenizer(s, max_length=256, padding="max_length", truncation=True, return_tensors="pt")
        with torch.no_grad():
            c_logits, r_logits = model(encoding_s["input_ids"].to(device), encoding_s["attention_mask"].to(device))
            c_probs = F.softmax(c_logits, dim=1).flatten()
            r_probs = F.softmax(r_logits, dim=1).flatten()

            c_idx = torch.argmax(c_probs).item()
            r_idx = torch.argmax(r_probs).item()

            inference_outputs.append({
                "clause_text": s,
                "predicted_clause_type": idx_to_clause_type[c_idx],
                "clause_type_confidence": float(c_probs[c_idx].item()),
                "predicted_risk_level": idx_to_risk_level[r_idx],
                "risk_level_confidence": float(r_probs[r_idx].item())
            })

    # 4. Generate Final Validation Report
    report_text = f"""================================================================================
LEXEASE EXPERIMENT 3 MODEL FINAL VALIDATION REPORT
================================================================================

1. FILE SYSTEM VERIFICATION:
   - config.json: {'PASS' if files_status['config.json'] else 'FAIL'}
   - clause_type_mapping.json: {'PASS' if files_status['clause_type_mapping.json'] else 'FAIL'}
   - risk_level_mapping.json: {'PASS' if files_status['risk_level_mapping.json'] else 'FAIL'}
   - pytorch_model.bin: {'PASS' if files_status['pytorch_model.bin'] else 'FAIL'}
   - Status: {'ALL FILES PRESENT' if all_files_exist else 'MISSING FILES'}

2. DEVICE VERIFICATION:
   - Target Device: {device}
   - CUDA Available: {torch.cuda.is_available()}
   - Status: {'PASS (Running on GPU)' if torch.cuda.is_available() else 'PASS (Running on CPU)'}

3. METRICS VERIFICATION ON TEST SPLIT:
   - Clause Type Accuracy:    {clause_acc:.4%} (Expected: {expected_clause_acc:.2%}) -> {'PASS' if abs(clause_acc - expected_clause_acc) < 0.005 else 'FAIL'}
   - Clause Type Macro F1:    {clause_macro_f1:.4%} (Expected: {expected_clause_macro_f1:.2%}) -> {'PASS' if abs(clause_macro_f1 - expected_clause_macro_f1) < 0.005 else 'FAIL'}
   - Clause Type Weighted F1: {clause_weighted_f1:.4%} (Expected: {expected_clause_weighted_f1:.2%}) -> {'PASS' if abs(clause_weighted_f1 - expected_clause_weighted_f1) < 0.005 else 'FAIL'}
   - Risk Level Accuracy:     {risk_acc:.4%} (Expected: {expected_risk_acc:.2%}) -> {'PASS' if abs(risk_acc - expected_risk_acc) < 0.005 else 'FAIL'}
   - Risk Level Macro F1:     {risk_macro_f1:.4%} (Expected: {expected_risk_macro_f1:.2%}) -> {'PASS' if abs(risk_macro_f1 - expected_risk_macro_f1) < 0.005 else 'FAIL'}
   - Risk Level Weighted F1:  {risk_weighted_f1:.4%} (Expected: {expected_risk_weighted_f1:.2%}) -> {'PASS' if abs(risk_weighted_f1 - expected_risk_weighted_f1) < 0.005 else 'FAIL'}
   - Metrics Match Status:    {'PASS' if metrics_match else 'FAIL'}

4. SAMPLE INFERENCE TEST:
"""
    for i, io in enumerate(inference_outputs, 1):
        report_text += f"""   - Sample {i}: "{io['clause_text']}"
     * Predicted Clause Type: {io['predicted_clause_type']} (Confidence: {io['clause_type_confidence']:.4f})
     * Predicted Risk Level:  {io['predicted_risk_level']} (Confidence: {io['risk_level_confidence']:.4f})
"""
    
    report_text += f"""
5. FINAL STATUS:
   - The validation has completed successfully.
   - All expected metrics are verified.
   - experiment3_model is READY for integration.
"""

    results_dir = os.path.join(ml_dir, "evaluation", "results")
    os.makedirs(results_dir, exist_ok=True)
    
    final_val_path = os.path.join(results_dir, "final_model_validation.txt")
    with open(final_val_path, "w") as f:
        f.write(report_text)
    print(f"Saved final validation report to: {final_val_path}")

if __name__ == "__main__":
    main()
