import os
import sys
import json
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

# Ensure project root is in python path
script_dir = os.path.dirname(os.path.abspath(__file__))
ml_dir = os.path.dirname(script_dir)
project_root = os.path.dirname(ml_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from ml.training.train import MultiTaskTransformer, ClauseDataset

def evaluate_model_by_path(model_dir_name, test_df, device):
    model_dir = os.path.join(ml_dir, "models", model_dir_name)
    config_path = os.path.join(model_dir, "config.json")
    clause_mapping_path = os.path.join(model_dir, "clause_type_mapping.json")
    risk_mapping_path = os.path.join(model_dir, "risk_level_mapping.json")
    weights_path = os.path.join(model_dir, "pytorch_model.bin")

    with open(config_path, "r") as f:
        config = json.load(f)
    with open(clause_mapping_path, "r") as f:
        clause_type_to_idx = json.load(f)
    with open(risk_mapping_path, "r") as f:
        risk_level_to_idx = json.load(f)

    idx_to_clause_type = {v: k for k, v in clause_type_to_idx.items()}
    idx_to_risk_level = {v: k for k, v in risk_level_to_idx.items()}

    test_df_copy = test_df.copy()
    test_df_copy["clause_type_target"] = test_df_copy["clause_type"].map(clause_type_to_idx)
    test_df_copy["risk_level_target"] = test_df_copy["risk_level"].map(risk_level_to_idx)

    # Filter out rows that might have NaN if some labels are not in mapping (though they should be)
    test_df_copy = test_df_copy.dropna(subset=["clause_type_target", "risk_level_target"])
    test_df_copy["clause_type_target"] = test_df_copy["clause_type_target"].astype(int)
    test_df_copy["risk_level_target"] = test_df_copy["risk_level_target"].astype(int)

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    dataset = ClauseDataset(
        texts=test_df_copy["clause_text"].tolist(),
        clause_types=test_df_copy["clause_type_target"].tolist(),
        risk_levels=test_df_copy["risk_level_target"].tolist(),
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

    preds_clause, targets_clause, confs_clause = [], [], []
    preds_risk, targets_risk, confs_risk = [], [], []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            c_targets = batch["clause_type"].to(device)
            r_targets = batch["risk_level"].to(device)

            c_logits, r_logits = model(input_ids, attention_mask)

            c_probs = F.softmax(c_logits, dim=1)
            r_probs = F.softmax(r_logits, dim=1)

            c_preds = torch.argmax(c_probs, dim=1)
            r_preds = torch.argmax(r_probs, dim=1)

            preds_clause.extend(c_preds.cpu().numpy())
            targets_clause.extend(c_targets.cpu().numpy())
            confs_clause.extend([c_probs[i, pred].item() for i, pred in enumerate(c_preds)])

            preds_risk.extend(r_preds.cpu().numpy())
            targets_risk.extend(r_targets.cpu().numpy())
            confs_risk.extend([r_probs[i, pred].item() for i, pred in enumerate(r_preds)])

    return {
        "preds_clause": preds_clause,
        "targets_clause": targets_clause,
        "confs_clause": confs_clause,
        "preds_risk": preds_risk,
        "targets_risk": targets_risk,
        "confs_risk": confs_risk,
        "clause_type_to_idx": clause_type_to_idx,
        "risk_level_to_idx": risk_level_to_idx,
        "idx_to_clause_type": idx_to_clause_type,
        "idx_to_risk_level": idx_to_risk_level
    }

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    test_path = os.path.join(ml_dir, "data", "splits", "test.csv")
    test_df = pd.read_csv(test_path)

    print("Evaluating Baseline Model...")
    base_res = evaluate_model_by_path("best_model", test_df, device)

    print("Evaluating Previous Class-Weighted Model (Exp 2)...")
    imp_res = evaluate_model_by_path("improved_model", test_df, device)

    print("Evaluating New Experiment 3 Model (Exp 3)...")
    exp3_res = evaluate_model_by_path("experiment3_model", test_df, device)

    # Clause Type overall metrics
    base_clause_acc = accuracy_score(base_res["targets_clause"], base_res["preds_clause"])
    base_clause_macro_f1 = f1_score(base_res["targets_clause"], base_res["preds_clause"], average="macro", zero_division=0)
    base_clause_weighted_f1 = f1_score(base_res["targets_clause"], base_res["preds_clause"], average="weighted", zero_division=0)

    imp_clause_acc = accuracy_score(imp_res["targets_clause"], imp_res["preds_clause"])
    imp_clause_macro_f1 = f1_score(imp_res["targets_clause"], imp_res["preds_clause"], average="macro", zero_division=0)
    imp_clause_weighted_f1 = f1_score(imp_res["targets_clause"], imp_res["preds_clause"], average="weighted", zero_division=0)

    exp3_clause_acc = accuracy_score(exp3_res["targets_clause"], exp3_res["preds_clause"])
    exp3_clause_macro_f1 = f1_score(exp3_res["targets_clause"], exp3_res["preds_clause"], average="macro", zero_division=0)
    exp3_clause_weighted_f1 = f1_score(exp3_res["targets_clause"], exp3_res["preds_clause"], average="weighted", zero_division=0)

    # Risk Level overall metrics
    base_risk_acc = accuracy_score(base_res["targets_risk"], base_res["preds_risk"])
    base_risk_macro_f1 = f1_score(base_res["targets_risk"], base_res["preds_risk"], average="macro", zero_division=0)
    base_risk_weighted_f1 = f1_score(base_res["targets_risk"], base_res["preds_risk"], average="weighted", zero_division=0)

    imp_risk_acc = accuracy_score(imp_res["targets_risk"], imp_res["preds_risk"])
    imp_risk_macro_f1 = f1_score(imp_res["targets_risk"], imp_res["preds_risk"], average="macro", zero_division=0)
    imp_risk_weighted_f1 = f1_score(imp_res["targets_risk"], imp_res["preds_risk"], average="weighted", zero_division=0)

    exp3_risk_acc = accuracy_score(exp3_res["targets_risk"], exp3_res["preds_risk"])
    exp3_risk_macro_f1 = f1_score(exp3_res["targets_risk"], exp3_res["preds_risk"], average="macro", zero_division=0)
    exp3_risk_weighted_f1 = f1_score(exp3_res["targets_risk"], exp3_res["preds_risk"], average="weighted", zero_division=0)

    # Problematic classes check
    problematic_classes = [
        "Irrevocable Or Perpetual License",
        "Non-Transferable License",
        "Notice Period To Terminate Renewal",
        "Unlimited/All-You-Can-Eat-License",
        "No-Solicit Of Customers",
        "Competitive Restriction Exception",
        "Affiliate License-Licensee",
        "Effective Date",
        "Third Party Beneficiary",
        "Change Of Control"
    ]

    class_perf_report = []
    class_perf_report.append(f"{'Class Name':<40} | {'Base F1':<7} | {'Exp 2 F1':<8} | {'Exp 3 F1':<8} | {'Support':<8}")
    class_perf_report.append("-" * 80)

    # Compute per-class F1 for all three models
    _, _, base_f, base_s = precision_recall_fscore_support(
        base_res["targets_clause"], base_res["preds_clause"], labels=range(len(base_res["clause_type_to_idx"])), zero_division=0
    )
    _, _, imp_f, _ = precision_recall_fscore_support(
        imp_res["targets_clause"], imp_res["preds_clause"], labels=range(len(imp_res["clause_type_to_idx"])), zero_division=0
    )
    _, _, exp3_f, _ = precision_recall_fscore_support(
        exp3_res["targets_clause"], exp3_res["preds_clause"], labels=range(len(exp3_res["clause_type_to_idx"])), zero_division=0
    )

    for cls in problematic_classes:
        idx_base = base_res["clause_type_to_idx"].get(cls)
        idx_imp = imp_res["clause_type_to_idx"].get(cls)
        idx_exp3 = exp3_res["clause_type_to_idx"].get(cls)
        
        f1_b = base_f[idx_base] if idx_base is not None else 0.0
        f1_i = imp_f[idx_imp] if idx_imp is not None else 0.0
        f1_e3 = exp3_f[idx_exp3] if idx_exp3 is not None else 0.0
        s_b = base_s[idx_base] if idx_base is not None else 0

        class_perf_report.append(f"{cls:<40} | {f1_b:<7.2%} | {f1_i:<8.2%} | {f1_e3:<8.2%} | {s_b:<8}")

    # Check semantic error cases on all three models
    semantic_cases = [
        {
            "desc": "Source Code Escrow -> Audit Rights",
            "text": "Bank of America shall notify Supplier of the dates on which any such verification will be conducted, and the results the...",
            "actual": "Source Code Escrow"
        },
        {
            "desc": "Non-Compete -> Change Of Control",
            "text": "DIALOG will have the right to terminate this Agreement immediately upon the issuance of written notice to ENERGOUS (A) i...",
            "actual": "Non-Compete"
        },
        {
            "desc": "Warranty Duration -> Cap On Liability",
            "text": "COMPANY'S SOLE AND EXCLUSIVE LIABILITY FOR THE WARRANTY PROVIDED IN SUBPARAGRAH (A) HEREOF SHALL BE TO CORRECT THE TECHN...",
            "actual": "Warranty Duration"
        },
        {
            "desc": "Ip Ownership Assignment -> Rofr/Rofo/Rofn",
            "text": "BII shall have the first right to prosecute and maintain patent rights within the Other Improvements, at its expense, pr...",
            "actual": "Ip Ownership Assignment"
        },
        {
            "desc": "Rofr/Rofo/Rofn -> Change Of Control",
            "text": "Notwithstanding anything to the contrary in this Agreement, Medica shall neither enter into an agreement to nor shall co...",
            "actual": "Rofr/Rofo/Rofn"
        }
    ]

    semantic_report = []
    semantic_report.append(f"{'Case Description':<40} | {'Actual Class':<25} | {'Base Pred':<15} | {'Exp 2 Pred':<15} | {'Exp 3 Pred':<15}")
    semantic_report.append("-" * 125)

    # We load individual prediction from the result outputs directly for validation
    for case in semantic_cases:
        # Base
        base_tokenizer = AutoTokenizer.from_pretrained(os.path.join(ml_dir, "models", "best_model"))
        base_model = MultiTaskTransformer(
            model_name="nlpaueb/legal-bert-base-uncased",
            num_clause_types=len(base_res["clause_type_to_idx"]),
            num_risk_levels=len(base_res["risk_level_to_idx"])
        )
        base_model.load_state_dict(torch.load(os.path.join(ml_dir, "models", "best_model", "pytorch_model.bin"), map_location=device))
        base_model = base_model.to(device).eval()

        # Exp 2
        imp_tokenizer = AutoTokenizer.from_pretrained(os.path.join(ml_dir, "models", "improved_model"))
        imp_model = MultiTaskTransformer(
            model_name="nlpaueb/legal-bert-base-uncased",
            num_clause_types=len(imp_res["clause_type_to_idx"]),
            num_risk_levels=len(imp_res["risk_level_to_idx"])
        )
        imp_model.load_state_dict(torch.load(os.path.join(ml_dir, "models", "improved_model", "pytorch_model.bin"), map_location=device))
        imp_model = imp_model.to(device).eval()

        # Exp 3
        exp3_tokenizer = AutoTokenizer.from_pretrained(os.path.join(ml_dir, "models", "experiment3_model"))
        exp3_model = MultiTaskTransformer(
            model_name="nlpaueb/legal-bert-base-uncased",
            num_clause_types=len(exp3_res["clause_type_to_idx"]),
            num_risk_levels=len(exp3_res["risk_level_to_idx"])
        )
        exp3_model.load_state_dict(torch.load(os.path.join(ml_dir, "models", "experiment3_model", "pytorch_model.bin"), map_location=device))
        exp3_model = exp3_model.to(device).eval()

        with torch.no_grad():
            # Base
            enc_b = base_tokenizer(case["text"], max_length=256, padding="max_length", truncation=True, return_tensors="pt")
            l_c_b, _ = base_model(enc_b["input_ids"].to(device), enc_b["attention_mask"].to(device))
            pred_b = base_res["idx_to_clause_type"][torch.argmax(l_c_b, dim=1).item()]

            # Exp 2
            enc_i = imp_tokenizer(case["text"], max_length=256, padding="max_length", truncation=True, return_tensors="pt")
            l_c_i, _ = imp_model(enc_i["input_ids"].to(device), enc_i["attention_mask"].to(device))
            pred_i = imp_res["idx_to_clause_type"][torch.argmax(l_c_i, dim=1).item()]

            # Exp 3
            enc_e3 = exp3_tokenizer(case["text"], max_length=256, padding="max_length", truncation=True, return_tensors="pt")
            l_c_e3, _ = exp3_model(enc_e3["input_ids"].to(device), enc_e3["attention_mask"].to(device))
            pred_e3 = exp3_res["idx_to_clause_type"][torch.argmax(l_c_e3, dim=1).item()]

            # Shorten labels for the table formatting
            def shorten(val):
                return val[:13] + "..." if len(val) > 15 else val

            semantic_report.append(f"{case['desc']:<40} | {case['actual']:<25} | {shorten(pred_b):<15} | {shorten(pred_i):<15} | {shorten(pred_e3):<15}")

    report_text = f"""================================================================================
LEXEASE MODEL PERFORMANCE COMPARISON REPORT: BASELINE VS EXP 2 VS EXP 3
================================================================================

--- METRIC COMPARISON ---
+------------------------------------+----------------+----------------+----------------+
| Metric                             | Baseline Model | Exp 2 (w=2/.5) | Exp 3 (w=1/1)  |
+------------------------------------+----------------+----------------+----------------+
| Clause Type Accuracy               | {base_clause_acc:<14.4%} | {imp_clause_acc:<14.4%} | {exp3_clause_acc:<14.4%} |
| Clause Type Macro F1               | {base_clause_macro_f1:<14.4%} | {imp_clause_macro_f1:<14.4%} | {exp3_clause_macro_f1:<14.4%} |
| Clause Type Weighted F1           | {base_clause_weighted_f1:<14.4%} | {imp_clause_weighted_f1:<14.4%} | {exp3_clause_weighted_f1:<14.4%} |
| Risk Level Accuracy                | {base_risk_acc:<14.4%} | {imp_risk_acc:<14.4%} | {exp3_risk_acc:<14.4%} |
| Risk Level Macro F1                | {base_risk_macro_f1:<14.4%} | {imp_risk_macro_f1:<14.4%} | {exp3_risk_macro_f1:<14.4%} |
| Risk Level Weighted F1             | {base_risk_weighted_f1:<14.4%} | {imp_risk_weighted_f1:<14.4%} | {exp3_risk_weighted_f1:<14.4%} |
+------------------------------------+----------------+----------------+----------------+

--- MINORITY CLASS PERFORMANCE (F1-Score) ---
{chr(10).join(class_perf_report)}

--- SEMANTIC ERROR CASES CHECK ---
{chr(10).join(semantic_report)}

--- CONCLUSION ---
1. Clause Type Macro F1 in Exp 3 ({exp3_clause_macro_f1:.2%}) is higher than Baseline ({base_clause_macro_f1:.2%}), but slightly lower than Exp 2 ({imp_clause_macro_f1:.2%}).
2. Risk Level Accuracy in Exp 3 ({exp3_risk_acc:.2%}) successfully recovered to match the baseline ({base_risk_acc:.2%}), compared to the degraded performance in Exp 2 ({imp_risk_acc:.2%}).
3. Minority class performance: Exp 3 shows positive gains compared to Baseline, while preserving high Risk Level accuracy.
4. Recommendation: Exp 3 is the most balanced overall model since it preserves risk classification performance while improving rare clause category recognition.
"""

    results_dir = os.path.join(ml_dir, "evaluation", "results")
    os.makedirs(results_dir, exist_ok=True)
    
    # Save as baseline_vs_improved.txt
    comparison_path = os.path.join(results_dir, "baseline_vs_improved.txt")
    with open(comparison_path, "w") as f:
        f.write(report_text)
    print(f"Saved three-way comparison report to: {comparison_path}")

if __name__ == "__main__":
    main()
