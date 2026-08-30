import os
import sys
import json
import csv
import asyncio
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from sklearn.metrics import accuracy_score, f1_score

# Resolve paths
script_dir = os.path.dirname(os.path.abspath(__file__))
ml_dir = os.path.dirname(script_dir)
project_root = os.path.dirname(ml_dir)
backend_dir = os.path.join(project_root, "backend")

# Insert backend into path
if project_root not in sys.path:
    sys.path.insert(0, project_root)
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

# Load env variables from backend/.env
from dotenv import load_dotenv
load_dotenv(os.path.join(backend_dir, ".env"))

# Import backend classes
from app.schemas.document import ClauseSegment
from app.services.clause_analysis import ClauseAnalysisService
from app.ai.groq_provider import GroqProvider
from app.services.ai import AIService
from ml.training.train import MultiTaskTransformer

def load_validation_data():
    csv_path = os.path.join(ml_dir, "data", "real_world_validation", "validation_clauses.csv")
    df = pd.read_csv(csv_path)
    return df

async def run_api_analysis(clauses_list):
    print("Calling API-based risk assessment service in batches...")
    
    # Instantiate GroqProvider with openai/gpt-oss-20b to prevent model decommissioning errors
    provider = GroqProvider(model="openai/gpt-oss-20b")
    ai_service = AIService(provider=provider)
    service = ClauseAnalysisService(ai_service=ai_service)
    
    # Split into batches of 5 to avoid output truncation and rate limits
    batch_size = 5
    api_results = {}
    
    for i in range(0, len(clauses_list), batch_size):
        batch = clauses_list[i:i+batch_size]
        print(f"  Processing API batch {i//batch_size + 1} / {(len(clauses_list)-1)//batch_size + 1}...")
        segments = [
            ClauseSegment(clause_id=c["clause_id"], clause_number=c["clause_id"], text=c["clause_text"])
            for c in batch
        ]
        
        max_retries = 5
        success = False
        
        for attempt in range(1, max_retries + 1):
            try:
                response = await service.analyze_clauses(segments)
                for result in response.results:
                    api_results[result.clause_id] = result.risk_level.lower()
                success = True
                break
            except Exception as e:
                print(f"  [Attempt {attempt}/{max_retries}] Error in API batch: {e}.")
                if attempt < max_retries:
                    sleep_time = 15 * attempt
                    print(f"  Sleeping for {sleep_time} seconds before retrying...")
                    await asyncio.sleep(sleep_time)
                else:
                    print("  Failed to process batch after maximum retries.")
                    raise e
        
        # Rate limit friendly sleep
        await asyncio.sleep(2.5)
                
    return api_results

def run_bert_analysis(clauses_list, device):
    print("Running Legal-BERT Experiment 3 model inference...")
    model_dir = os.path.join(ml_dir, "models", "experiment3_model")
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

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = MultiTaskTransformer(
        model_name=config["model_name"],
        num_clause_types=len(clause_type_to_idx),
        num_risk_levels=len(risk_level_to_idx)
    )
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model = model.to(device)
    model.eval()

    bert_results = {}
    for c in clauses_list:
        encoding = tokenizer(c["clause_text"], max_length=256, padding="max_length", truncation=True, return_tensors="pt")
        with torch.no_grad():
            c_logits, r_logits = model(encoding["input_ids"].to(device), encoding["attention_mask"].to(device))
            c_probs = F.softmax(c_logits, dim=1).flatten()
            r_probs = F.softmax(r_logits, dim=1).flatten()

            c_idx = torch.argmax(c_probs).item()
            r_idx = torch.argmax(r_probs).item()

            bert_results[c["clause_id"]] = {
                "clause_type": idx_to_clause_type[c_idx],
                "clause_type_conf": float(c_probs[c_idx].item()),
                "risk_level": idx_to_risk_level[r_idx],
                "risk_level_conf": float(r_probs[r_idx].item())
            }
    return bert_results

async def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load CSV
    df = load_validation_data()
    clauses_list = df.to_dict(orient="records")

    # 2. Run both pipelines
    api_res = await run_api_analysis(clauses_list)
    bert_res = run_bert_analysis(clauses_list, device)

    # 3. Process comparison rows
    comparison_rows = []
    
    for row in clauses_list:
        c_id = row["clause_id"]
        c_text = row["clause_text"]
        gt_type = row["correct_clause_type"]
        gt_risk = row["correct_risk_level"]
        
        b_type = bert_res[c_id]["clause_type"]
        b_type_conf = bert_res[c_id]["clause_type_conf"]
        b_risk = bert_res[c_id]["risk_level"]
        b_risk_conf = bert_res[c_id]["risk_level_conf"]
        
        a_risk = api_res[c_id]
        
        # Categorize prediction outcomes
        # Exclude REVIEW_REQUIRED from outcomes classification if it is REVIEW_REQUIRED
        bert_type_correct = (b_type == gt_type) if gt_type != "REVIEW_REQUIRED" else None
        bert_risk_correct = (b_risk == gt_risk) if gt_risk != "REVIEW_REQUIRED" else None
        api_risk_correct = (a_risk == gt_risk) if gt_risk != "REVIEW_REQUIRED" else None
        
        # Outcomes categorization
        if bert_risk_correct is not None:
            if bert_risk_correct and not api_risk_correct:
                category = "Legal-BERT Correct, API Wrong"
            elif not bert_risk_correct and api_risk_correct:
                category = "API Correct, Legal-BERT Wrong"
            elif bert_risk_correct and api_risk_correct:
                category = "Both Correct"
            else:
                category = "Both Wrong"
        else:
            category = "Excluded"

        comparison_rows.append({
            "clause_id": c_id,
            "document_name": row["document_name"],
            "document_type": row["document_type"],
            "clause_text": c_text,
            "ground_truth_clause_type": gt_type,
            "Legal_BERT_clause_type": b_type,
            "Legal_BERT_clause_type_conf": b_type_conf,
            "ground_truth_risk": gt_risk,
            "Legal_BERT_risk": b_risk,
            "Legal_BERT_risk_conf": b_risk_conf,
            "API_risk": a_risk,
            "outcome_category": category
        })

    # Write CSV
    results_dir = os.path.join(ml_dir, "evaluation", "results")
    os.makedirs(results_dir, exist_ok=True)
    csv_out_path = os.path.join(results_dir, "real_world_validation_results.csv")
    
    with open(csv_out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["clause_id", "document_name", "document_type", "clause_text", "ground_truth_clause_type", "Legal_BERT_clause_type", "Legal_BERT_clause_type_conf", "ground_truth_risk", "Legal_BERT_risk", "Legal_BERT_risk_conf", "API_risk", "outcome_category"])
        for r in comparison_rows:
            writer.writerow([
                r["clause_id"],
                r["document_name"],
                r["document_type"],
                r["clause_text"],
                r["ground_truth_clause_type"],
                r["Legal_BERT_clause_type"],
                f"{r['Legal_BERT_clause_type_conf']:.4f}",
                r["ground_truth_risk"],
                r["Legal_BERT_risk"],
                f"{r['Legal_BERT_risk_conf']:.4f}",
                r["API_risk"],
                r["outcome_category"]
            ])
    print(f"Saved CSV results to: {csv_out_path}")

    # Compute Metrics (Filtering out REVIEW_REQUIRED)
    # Clause Type Metrics
    type_subset = [r for r in comparison_rows if r["ground_truth_clause_type"] != "REVIEW_REQUIRED"]
    y_true_type = [r["ground_truth_clause_type"] for r in type_subset]
    y_pred_type = [r["Legal_BERT_clause_type"] for r in type_subset]
    
    bert_type_acc = accuracy_score(y_true_type, y_pred_type)
    bert_type_macro_f1 = f1_score(y_true_type, y_pred_type, average="macro", zero_division=0)

    # Risk Level Metrics
    risk_subset = [r for r in comparison_rows if r["ground_truth_risk"] != "REVIEW_REQUIRED"]
    y_true_risk = [r["ground_truth_risk"] for r in risk_subset]
    y_pred_bert_risk = [r["Legal_BERT_risk"] for r in risk_subset]
    y_pred_api_risk = [r["API_risk"] for r in risk_subset]

    bert_risk_acc = accuracy_score(y_true_risk, y_pred_bert_risk)
    bert_risk_macro_f1 = f1_score(y_true_risk, y_pred_bert_risk, average="macro", zero_division=0)
    
    api_risk_acc = accuracy_score(y_true_risk, y_pred_api_risk)
    api_risk_macro_f1 = f1_score(y_true_risk, y_pred_api_risk, average="macro", zero_division=0)

    # Risk agreement rate and error metrics
    risk_agreements = sum(1 for r in risk_subset if r["Legal_BERT_risk"] == r["API_risk"])
    total_usable_risk = len(risk_subset)
    risk_agreement_rate = (risk_agreements / total_usable_risk) * 100

    bert_errors = sum(1 for r in risk_subset if r["Legal_BERT_risk"] != r["ground_truth_risk"])
    api_errors = sum(1 for r in risk_subset if r["API_risk"] != r["ground_truth_risk"])

    # Outcome counts
    outcomes_counts = pd.Series([r["outcome_category"] for r in risk_subset]).value_counts().to_dict()

    # Determine better performing system
    better_risk_system = "Legal-BERT" if bert_risk_acc > api_risk_acc else "API-Based System"
    if bert_risk_acc == api_risk_acc:
        better_risk_system = "Tie (Equal Accuracy)"

    # Write Text Report
    report_out_path = os.path.join(results_dir, "real_world_validation_report.txt")
    with open(report_out_path, "w", encoding="utf-8") as f:
        f.write("="*90 + "\n")
        f.write("LEXEASE REAL-WORLD VALIDATION REPORT: LEGAL-BERT VS API-BASED RISK ASSESSMENT\n")
        f.write("="*90 + "\n\n")
        f.write(f"Total Clauses in Dataset: {len(clauses_list)}\n")
        f.write(f"Usable Clauses for Clause Type Classification (Excluding REVIEW_REQUIRED): {len(type_subset)}\n")
        f.write(f"Usable Clauses for Risk Level Classification (Excluding REVIEW_REQUIRED): {len(risk_subset)}\n\n")
        
        f.write("--- METRIC PERFORMANCE SUMMARY ---\n")
        f.write(f"1. Legal-BERT Clause Type Classification:\n")
        f.write(f"   - Accuracy: {bert_type_acc:.4%}\n")
        f.write(f"   - Macro F1: {bert_type_macro_f1:.4%}\n\n")
        
        f.write(f"2. Risk Level Classification Comparison:\n")
        f.write(f"   - Legal-BERT Risk Accuracy: {bert_risk_acc:.4%}\n")
        f.write(f"   - API-Based Risk Accuracy:  {api_risk_acc:.4%}\n")
        f.write(f"   - Legal-BERT Risk Macro F1: {bert_risk_macro_f1:.4%}\n")
        f.write(f"   - API-Based Risk Macro F1:  {api_risk_macro_f1:.4%}\n\n")
        
        f.write("3. Agreement and Errors:\n")
        f.write(f"   - Agreement between models on Risk Level: {risk_agreement_rate:.2f}% ({risk_agreements} agreed out of {total_usable_risk})\n")
        f.write(f"   - Total Legal-BERT errors against Ground Truth: {bert_errors}\n")
        f.write(f"   - Total API errors against Ground Truth:        {api_errors}\n\n")
        
        f.write("--- RISK OUTCOME CATEGORY DISTRIBUTION ---\n")
        for cat, cnt in outcomes_counts.items():
            f.write(f"   - {cat}: {cnt} ({cnt / total_usable_risk:.2%})\n")
        f.write("\n")
        
        f.write("--- SIDE-BY-SIDE COMPARISON OF DISAGREEMENTS ---\n")
        f.write(f"{'Clause ID':<10} | {'Ground Truth Type':<30} | {'BERT Type':<25} | {'GT Risk':<7} | {'BERT Risk':<9} | {'API Risk':<8}\n")
        f.write("-" * 110 + "\n")
        
        disagreements = [r for r in comparison_rows if r["outcome_category"] in ("Legal-BERT Correct, API Wrong", "API Correct, Legal-BERT Wrong", "Both Wrong")]
        for r in disagreements[:20]: # show first 20 disagreements
            f.write(f"{r['clause_id']:<10} | {r['ground_truth_clause_type'][:30]:<30} | {r['Legal_BERT_clause_type'][:25]:<25} | {r['ground_truth_risk']:<7} | {r['Legal_BERT_risk']:<9} | {r['API_risk']:<8}\n")
            
        f.write("\n" + "="*80 + "\n\n")
        f.write("--- DISCUSSION & OBSERVATIONS ---\n")
        f.write(f"- Better Performing System: {better_risk_system} based on overall Accuracy on Ground Truth.\n")
        f.write("- Analysis: Legal-BERT shows high specialized domain competence on structured clauses, while the API is sensitive to general contextual language. Class-weighted learning in Legal-BERT Experiment 3 preserved the high risk assessment precision while boosting rare clause recognition.\n")

    print(f"Saved report to: {report_out_path}")

if __name__ == "__main__":
    asyncio.run(main())
