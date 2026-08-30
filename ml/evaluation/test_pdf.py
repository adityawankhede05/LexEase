import os
import sys
import json
import csv
import asyncio
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

# Resolve paths
script_dir = os.path.dirname(os.path.abspath(__file__))
ml_dir = os.path.dirname(script_dir)
project_root = os.path.dirname(ml_dir)
backend_dir = os.path.join(project_root, "backend")

# Insert project root and backend into path
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
from ml.training.train import MultiTaskTransformer

# The 6 clauses from the PDF
clauses = [
    {
        "id": "c1",
        "number": "Clause 1",
        "text": "The Tenant shall pay monthly rent of INR 25,000 to the Landlord on or before the 5th day of each calendar month."
    },
    {
        "id": "c2",
        "number": "Clause 2",
        "text": "The Tenant shall pay a refundable security deposit of INR 50,000 before taking possession of the premises. The deposit shall be returned within 30 days after the tenancy ends, subject to lawful deductions."
    },
    {
        "id": "c3",
        "number": "Clause 3",
        "text": "Either party may terminate this Agreement by giving the other party at least 30 days written notice."
    },
    {
        "id": "c4",
        "number": "Clause 4",
        "text": "The Tenant shall keep the premises clean and in good condition. The Landlord shall be responsible for major structural repairs unless the damage was caused by the Tenant's negligence or misuse."
    },
    {
        "id": "c5",
        "number": "Clause 5",
        "text": "The Tenant shall pay electricity and water charges based on actual consumption. Property taxes and structural maintenance charges shall be paid by the Landlord."
    },
    {
        "id": "c6",
        "number": "Clause 6",
        "text": "The premises shall be used only for residential purposes. The Tenant shall not use the premises for any unlawful activity or commercial operation without the Landlord's written consent."
    }
]

async def run_api_analysis():
    print("Calling API-based risk assessment service...")
    service = ClauseAnalysisService()
    segments = [
        ClauseSegment(clause_id=c["id"], clause_number=c["number"], text=c["text"])
        for c in clauses
    ]
    response = await service.analyze_clauses(segments)
    
    api_results = {}
    for result in response.results:
        api_results[result.clause_id] = {
            "risk_level": result.risk_level.lower(),
            "explanation": result.explanation,
            "recommendation": result.recommendation
        }
    return api_results

def run_bert_analysis(device):
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
    for c in clauses:
        encoding = tokenizer(c["text"], max_length=256, padding="max_length", truncation=True, return_tensors="pt")
        with torch.no_grad():
            c_logits, r_logits = model(encoding["input_ids"].to(device), encoding["attention_mask"].to(device))
            c_probs = F.softmax(c_logits, dim=1).flatten()
            r_probs = F.softmax(r_logits, dim=1).flatten()

            c_idx = torch.argmax(c_probs).item()
            r_idx = torch.argmax(r_probs).item()

            bert_results[c["id"]] = {
                "clause_type": idx_to_clause_type[c_idx],
                "clause_type_conf": float(c_probs[c_idx].item()),
                "risk_level": idx_to_risk_level[r_idx],
                "risk_level_conf": float(r_probs[r_idx].item())
            }
    return bert_results

async def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Run both pipelines
    api_res = await run_api_analysis()
    bert_res = run_bert_analysis(device)

    # Compare results
    comparison_data = []
    agreements_count = 0
    disagreements_count = 0

    for c in clauses:
        c_id = c["id"]
        c_num = c["number"]
        text = c["text"]
        
        b_type = bert_res[c_id]["clause_type"]
        b_type_conf = bert_res[c_id]["clause_type_conf"]
        b_risk = bert_res[c_id]["risk_level"]
        b_risk_conf = bert_res[c_id]["risk_level_conf"]
        
        a_risk = api_res[c_id]["risk"] if "risk" in api_res[c_id] else api_res[c_id]["risk_level"]
        a_expl = api_res[c_id]["explanation"]
        
        agrees = (b_risk == a_risk)
        if agrees:
            agreements_count += 1
        else:
            disagreements_count += 1

        comparison_data.append({
            "id": c_id,
            "number": c_num,
            "text": text,
            "bert_type": b_type,
            "bert_type_conf": b_type_conf,
            "bert_risk": b_risk,
            "bert_risk_conf": b_risk_conf,
            "api_risk": a_risk,
            "api_explanation": a_expl,
            "agreement": "Agree" if agrees else "Disagree"
        })

    percent_agreement = (agreements_count / len(clauses)) * 100

    # Write CSV
    results_dir = os.path.join(ml_dir, "evaluation", "results")
    os.makedirs(results_dir, exist_ok=True)
    
    csv_path = os.path.join(results_dir, "new_pdf_comparison.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Clause ID", "Clause Number", "Text", "Legal-BERT Clause Type", "Legal-BERT Type Confidence", "Legal-BERT Risk Level", "Legal-BERT Risk Confidence", "API Risk Level", "Agreement"])
        for row in comparison_data:
            writer.writerow([
                row["id"],
                row["number"],
                row["text"],
                row["bert_type"],
                f"{row['bert_type_conf']:.4f}",
                row["bert_risk"],
                f"{row['bert_risk_conf']:.4f}",
                row["api_risk"],
                row["agreement"]
            ])
    print(f"Saved CSV results to: {csv_path}")

    # Write full report TXT
    report_path = os.path.join(results_dir, "new_pdf_comparison.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("="*90 + "\n")
        f.write("REAL-WORLD COMPARISON: LEGAL-BERT vs API-BASED RISK ASSESSMENT\n")
        f.write("="*90 + "\n\n")
        f.write(f"Document Name: Residential Rental Agreement.pdf\n")
        f.write(f"Number of Clauses Tested: {len(clauses)}\n")
        f.write(f"Inference Device: {device}\n")
        f.write(f"Agreement Rate: {percent_agreement:.2f}% (Agreed: {agreements_count}, Disagreed: {disagreements_count})\n\n")
        
        f.write("--- SIDE-BY-SIDE COMPARISON TABLE ---\n")
        f.write(f"{'Clause':<10} | {'BERT Clause Type':<30} | {'BERT Risk':<10} | {'API Risk':<10} | {'Agreement':<10}\n")
        f.write("-" * 80 + "\n")
        for row in comparison_data:
            f.write(f"{row['number']:<10} | {row['bert_type'][:30]:<30} | {row['bert_risk']:<10} | {row['api_risk']:<10} | {row['agreement']:<10}\n")
        f.write("\n" + "="*80 + "\n\n")
        
        f.write("--- DETAILED CLAUSE COMPARISONS ---\n\n")
        for row in comparison_data:
            f.write(f"[{row['number']}] Text: \"{row['text']}\"\n")
            f.write(f"  * Legal-BERT Classification: {row['bert_type']} (Confidence: {row['bert_type_conf']:.4f})\n")
            f.write(f"  * Legal-BERT Risk Level:    {row['bert_risk'].upper()} (Confidence: {row['bert_risk_conf']:.4f})\n")
            f.write(f"  * API-Based Risk Level:     {row['api_risk'].upper()}\n")
            f.write(f"  * API-Based Explanation:    {row['api_explanation']}\n")
            f.write(f"  * Agreement Status:         {row['agreement']}\n\n")
            
        f.write("--- DISCUSSION & OBSERVATIONS ---\n")
        f.write("1. Analysis of Disagreements:\n")
        f.write("   - Compare the granular differences between Legal-BERT and API predictions. Large Language Models (LLMs) used in APIs usually assess risk based on broad commonsense reasoning and instructional context, whereas the specialized Legal-BERT model classifies based on domain-specific training corpus statistics.\n")
        f.write("   - For instance, if Legal-BERT predicts a high risk while the API predicts low risk, it might be due to Legal-BERT capturing minor, subtle phrasing signals or dataset-specific biases, whereas the API analyzes context globally.\n")
        f.write("2. No Ground Truth Disclaimer:\n")
        f.write("   - Note that because we do not have annotated ground-truth labels for this new document, we cannot evaluate the absolute accuracy of either system. Instead, we analyze their agreement rate to understand model variance in wild documents.\n")
    print(f"Saved full report to: {report_path}")

    # Write summary TXT
    summary_path = os.path.join(results_dir, "new_pdf_comparison_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"Document Name: Residential Rental Agreement.pdf\n")
        f.write(f"Number of Clauses Tested: {len(clauses)}\n")
        f.write(f"Agreement Rate: {percent_agreement:.2f}%\n")
        f.write(f"Agreed: {agreements_count}\n")
        f.write(f"Disagreed: {disagreements_count}\n")
        f.write(f"Inference Device: {device}\n")
    print(f"Saved summary to: {summary_path}")

if __name__ == "__main__":
    asyncio.run(main())
