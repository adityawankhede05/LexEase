import os
import sys
import json
import logging
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

# Ensure root of project is in Python path for module importing
script_dir = os.path.dirname(os.path.abspath(__file__))
ml_dir = os.path.dirname(script_dir)
project_root = os.path.dirname(ml_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import model definition and dataset class
from ml.training.train import MultiTaskTransformer, ClauseDataset

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def evaluate_model() -> None:
    # Resolve model directory and config
    model_dir = os.path.join(ml_dir, "models", "best_model")
    config_path = os.path.join(model_dir, "config.json")
    clause_mapping_path = os.path.join(model_dir, "clause_type_mapping.json")
    risk_mapping_path = os.path.join(model_dir, "risk_level_mapping.json")

    logger.info("Checking model output folders...")
    if not all(os.path.exists(p) for p in [config_path, clause_mapping_path, risk_mapping_path]):
        logger.error("Model artifacts (config, mappings, weights) not found at models/best_model. Train the model first.")
        sys.exit(1)

    # Load parameters and mappings
    with open(config_path, "r") as f:
        config = json.load(f)
    with open(clause_mapping_path, "r") as f:
        clause_type_to_idx = json.load(f)
    with open(risk_mapping_path, "r") as f:
        risk_level_to_idx = json.load(f)

    idx_to_clause_type = {v: k for k, v in clause_type_to_idx.items()}
    idx_to_risk_level = {v: k for k, v in risk_level_to_idx.items()}

    # Load test split
    test_path = os.path.join(ml_dir, "data", "splits", "test.csv")
    logger.info(f"Loading test split from {test_path}...")
    if not os.path.exists(test_path):
        logger.error(f"Test split file not found at {test_path}.")
        sys.exit(1)

    test_df = pd.read_csv(test_path)
    
    # Validate columns
    required_cols = {"clause_text", "clause_type", "risk_level"}
    if not required_cols.issubset(test_df.columns):
        logger.error(f"Test dataset is missing required columns. Expected: {required_cols}")
        sys.exit(1)

    # Apply mappings
    test_df["clause_type_target"] = test_df["clause_type"].map(clause_type_to_idx)
    test_df["risk_level_target"] = test_df["risk_level"].map(risk_level_to_idx)

    # Setup tokenizer, model, and data loader
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
    logger.info(f"Evaluating on device: {device}")

    # Reconstruct multi-task transformer
    model = MultiTaskTransformer(
        model_name=config["model_name"],
        num_clause_types=len(clause_type_to_idx),
        num_risk_levels=len(risk_level_to_idx)
    )
    model.load_state_dict(torch.load(os.path.join(model_dir, "pytorch_model.bin"), map_location=device))
    model = model.to(device)
    model.eval()

    test_clause_preds, test_clause_targets = [], []
    test_risk_preds, test_risk_targets = [], []

    logger.info("Running evaluation predictions...")
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            clause_targets = batch["clause_type"].to(device)
            risk_targets = batch["risk_level"].to(device)

            clause_logits, risk_logits = model(input_ids, attention_mask)

            test_clause_preds.extend(torch.argmax(clause_logits, dim=1).cpu().numpy())
            test_clause_targets.extend(clause_targets.cpu().numpy())
            test_risk_preds.extend(torch.argmax(risk_logits, dim=1).cpu().numpy())
            test_risk_targets.extend(risk_targets.cpu().numpy())

    # Calculate metrics
    clause_acc = accuracy_score(test_clause_targets, test_clause_preds)
    clause_macro_f1 = f1_score(test_clause_targets, test_clause_preds, average="macro", zero_division=0)
    clause_weighted_f1 = f1_score(test_clause_targets, test_clause_preds, average="weighted", zero_division=0)

    risk_acc = accuracy_score(test_risk_targets, test_risk_preds)
    risk_macro_f1 = f1_score(test_risk_targets, test_risk_preds, average="macro", zero_division=0)
    risk_weighted_f1 = f1_score(test_risk_targets, test_risk_preds, average="weighted", zero_division=0)

    logger.info("\n" + "="*40 + "\nEVALUATION METRICS REPORT\n" + "="*40)
    logger.info(f"Clause Type Accuracy:    {clause_acc:.4f}")
    logger.info(f"Clause Type Macro F1:    {clause_macro_f1:.4f}")
    logger.info(f"Clause Type Weighted F1: {clause_weighted_f1:.4f}")
    logger.info("-" * 40)
    logger.info(f"Risk Level Accuracy:     {risk_acc:.4f}")
    logger.info(f"Risk Level Macro F1:     {risk_macro_f1:.4f}")
    logger.info(f"Risk Level Weighted F1:  {risk_weighted_f1:.4f}")
    logger.info("-" * 40)

    # Risk level confusion matrix
    risk_labels = [idx_to_risk_level[i] for i in sorted(risk_level_to_idx.values())]
    matrix = confusion_matrix(test_risk_targets, test_risk_preds, labels=sorted(risk_level_to_idx.values()))
    
    logger.info("Confusion Matrix (Risk Level):")
    logger.info(f" Labels order: {risk_labels}")
    for row in matrix:
        logger.info(f" {row.tolist()}")
    logger.info("-" * 40)

    # Classification report
    clause_report = classification_report(
        test_clause_targets, 
        test_clause_preds, 
        target_names=[idx_to_clause_type[i] for i in sorted(clause_type_to_idx.values())], 
        zero_division=0
    )
    risk_report = classification_report(
        test_risk_targets, 
        test_risk_preds, 
        target_names=risk_labels, 
        zero_division=0
    )

    logger.info(f"Risk Level Classification Report:\n{risk_report}")
    logger.info(f"Clause Type Classification Report:\n{clause_report}")

    # Save the evaluation results under ml/evaluation/results/
    results_dir = os.path.join(ml_dir, "evaluation", "results")
    os.makedirs(results_dir, exist_ok=True)
    
    report_text_path = os.path.join(results_dir, "evaluation_report.txt")
    with open(report_text_path, "w") as f:
        f.write("="*60 + "\nLEXEASE MULTI-TASK TRANSFORMER EVALUATION REPORT\n" + "="*60 + "\n\n")
        f.write("--- CLAUSE TYPE CLASSIFICATION METRICS (41 Classes) ---\n")
        f.write(f"Accuracy:    {clause_acc:.4f}\n")
        f.write(f"Macro F1:    {clause_macro_f1:.4f}\n")
        f.write(f"Weighted F1: {clause_weighted_f1:.4f}\n\n")
        f.write(f"Classification Report:\n{clause_report}\n")
        f.write("="*60 + "\n\n")
        f.write("--- RISK LEVEL CLASSIFICATION METRICS (3 Classes) ---\n")
        f.write(f"Accuracy:    {risk_acc:.4f}\n")
        f.write(f"Macro F1:    {risk_macro_f1:.4f}\n")
        f.write(f"Weighted F1: {risk_weighted_f1:.4f}\n\n")
        f.write("Confusion Matrix:\n")
        f.write(f"Labels: {risk_labels}\n")
        for row in matrix:
            f.write(f"  {row.tolist()}\n")
        f.write(f"\nClassification Report:\n{risk_report}\n")
        
    logger.info(f"Saved evaluation report text to: {report_text_path}")

    # Also save structured JSON metrics
    metrics_json_path = os.path.join(results_dir, "metrics.json")
    with open(metrics_json_path, "w") as f:
        json.dump({
            "clause_type": {
                "accuracy": float(clause_acc),
                "macro_f1": float(clause_macro_f1),
                "weighted_f1": float(clause_weighted_f1)
            },
            "risk_level": {
                "accuracy": float(risk_acc),
                "macro_f1": float(risk_macro_f1),
                "weighted_f1": float(risk_weighted_f1),
                "confusion_matrix": matrix.tolist(),
                "labels": risk_labels
            }
        }, f, indent=2)
    logger.info(f"Saved evaluation report JSON to: {metrics_json_path}")

if __name__ == "__main__":
    evaluate_model()
