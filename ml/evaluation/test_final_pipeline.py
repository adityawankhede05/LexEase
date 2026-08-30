"""
LexEase Final Pipeline Integration Test
========================================
Uses the existing Experiment 8 clause classifier and the final risk model.
NO training. NO data generation. Pure integration/verification run.
"""

import os
import sys
import json
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import torch.nn as nn

# Resolve paths
script_dir  = os.path.dirname(os.path.abspath(__file__))
ml_dir      = os.path.dirname(script_dir)
project_root= os.path.dirname(ml_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ────────────────────────────────────────────────────────────────────────────
# 1.  Model definitions  (mirrors train_experiment8 & evaluate_pretrained)
# ────────────────────────────────────────────────────────────────────────────

class AdaptedClauseClassifier(nn.Module):
    def __init__(self, model_name="nlpaueb/legal-bert-base-uncased", num_classes=22, dropout_rate=0.25):
        super().__init__()
        self.bert       = AutoModel.from_pretrained(model_name)
        self.dropout    = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(768, num_classes)

    def forward(self, input_ids, attention_mask):
        out  = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = out.last_hidden_state[:, 0, :]
        return self.classifier(self.dropout(pooled))


class MultiTaskRiskModel(nn.Module):
    """AnkushRaheja / final_risk_model architecture.
    Exact architecture as saved in final_risk_model/pytorch_model.bin:
      classifier: Sequential(Linear(768→384), ReLU, Dropout, Linear(384→num_labels))
      regressor:  Sequential(Linear(768→384), ReLU, Dropout, Linear(384→1), Sigmoid)
    """
    def __init__(self, model_name="nlpaueb/legal-bert-base-uncased", num_labels=100, dropout_rate=0.25):
        super().__init__()
        self.bert       = AutoModel.from_pretrained(model_name)
        self.dropout    = nn.Dropout(dropout_rate)
        self.classifier = nn.Sequential(
            nn.Linear(768, 384), nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(384, num_labels)
        )
        self.regressor  = nn.Sequential(
            nn.Linear(768, 384), nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(384, 1), nn.Sigmoid()
        )

    def forward(self, input_ids, attention_mask):
        out    = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = out.last_hidden_state[:, 0, :]
        dropped= self.dropout(pooled)
        cls_logits   = self.classifier(dropped)
        risk_score   = self.regressor(dropped).squeeze(-1)
        return cls_logits, risk_score


# ────────────────────────────────────────────────────────────────────────────
# 2.  Load Experiment 8 clause classifier
# ────────────────────────────────────────────────────────────────────────────

def load_clause_classifier():
    model_dir = os.path.join(ml_dir, "models", "experiment8_indian_classifier")
    with open(os.path.join(model_dir, "clause_type_mapping.json")) as f:
        mapping = json.load(f)
    idx_to_label = {v: k for k, v in mapping.items()}
    num_classes  = len(mapping)

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model     = AdaptedClauseClassifier(num_classes=num_classes)
    model.load_state_dict(
        torch.load(os.path.join(model_dir, "pytorch_model.bin"), map_location=DEVICE)
    )
    model.to(DEVICE).eval()
    print(f"[OK] Experiment 8 clause classifier loaded — {num_classes} classes.")
    return tokenizer, model, idx_to_label


def classify_clause(text, tokenizer, model, idx_to_label):
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=256, padding=True)
    with torch.no_grad():
        logits = model(enc["input_ids"].to(DEVICE), enc["attention_mask"].to(DEVICE))
        probs  = F.softmax(logits, dim=-1)
        conf, idx = torch.max(probs, dim=-1)
    clause_type = idx_to_label[idx.item()]
    confidence  = conf.item()
    if confidence < 0.25:
        return "UNKNOWN / OTHER", confidence
    return clause_type, confidence


# ────────────────────────────────────────────────────────────────────────────
# 3.  Load final risk model  (ml/models/final_risk_model/)
#     This is the AnkushRaheja-base model fine-tuned on our training data.
#     Risk methodology:
#       raw_score = regressor(BERT pooled) * 100   → 0-100
#       LOW    if raw_score < 41
#       MEDIUM if 41 ≤ raw_score < 60
#       HIGH   if raw_score ≥ 60
#     Thresholds were calibrated on validation.csv via grid search, then frozen.
# ────────────────────────────────────────────────────────────────────────────

# Frozen thresholds calibrated on validation.csv
TH_LOW_MED  = 41.0
TH_MED_HIGH = 60.0

def load_risk_model():
    model_dir = os.path.join(ml_dir, "models", "final_risk_model")
    with open(os.path.join(model_dir, "metadata.json")) as f:
        meta = json.load(f)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model     = MultiTaskRiskModel(num_labels=meta["num_labels"])
    model.load_state_dict(
        torch.load(os.path.join(model_dir, "pytorch_model.bin"), map_location=DEVICE)
    )
    model.to(DEVICE).eval()
    print(f"[OK] Final risk model loaded (base: {meta['model_name']}).")
    return tokenizer, model


def predict_risk(text, tokenizer, model):
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=256, padding=True)
    with torch.no_grad():
        _, risk_raw = model(enc["input_ids"].to(DEVICE), enc["attention_mask"].to(DEVICE))
        score = float(risk_raw.item()) * 100
    if score < TH_LOW_MED:
        level = "LOW"
    elif score < TH_MED_HIGH:
        level = "MEDIUM"
    else:
        level = "HIGH"
    return round(score, 2), level


# ────────────────────────────────────────────────────────────────────────────
# 4.  Test clauses  — one per target domain
# ────────────────────────────────────────────────────────────────────────────

TEST_CLAUSES = [
    # ── RENTAL / LEASE ──────────────────────────────────────────────────────
    {
        "domain": "Rental — Rent (LOW)",
        "text": "The Tenant shall pay a monthly rent of INR 20,000 on or before the 5th day of each calendar month."
    },
    {
        "domain": "Rental — Security Deposit (HIGH)",
        "text": ("The Tenant shall pay a non-refundable security deposit of INR 1,00,000. "
                 "The deposit shall be forfeited in full if the Tenant terminates the agreement "
                 "before the 24-month lock-in period, with no exceptions.")
    },
    {
        "domain": "Rental — Security Deposit (LOW)",
        "text": ("The Tenant shall pay a refundable security deposit of INR 50,000 before occupying the premises. "
                 "The deposit shall be returned within 30 days of vacating, subject to deductions for actual physical damage only.")
    },
    {
        "domain": "Rental — Termination (MEDIUM)",
        "text": ("The Landlord may terminate this agreement on 30 days' notice. "
                 "The Tenant must give 90 days' notice and pay an early-exit penalty of 2 months' rent.")
    },
    # ── EMPLOYMENT ──────────────────────────────────────────────────────────
    {
        "domain": "Employment — Non-Compete (HIGH)",
        "text": ("The Employee shall not, directly or indirectly, engage in any business competitive with the Employer "
                 "anywhere in India for a period of 3 years after termination of employment. "
                 "Breach shall attract liquidated damages of INR 10 Lakhs.")
    },
    {
        "domain": "Employment — Non-Compete (LOW)",
        "text": ("During the term of employment, the Employee shall not work for a direct competitor "
                 "within a 5 km radius of the office premises.")
    },
    {
        "domain": "Employment — Salary (LOW)",
        "text": ("The Employee shall be paid a monthly salary of INR 80,000, "
                 "payable on the last working day of each month, with annual increments as per performance review.")
    },
    # ── NDA / CONFIDENTIALITY ───────────────────────────────────────────────
    {
        "domain": "NDA — Confidentiality (HIGH)",
        "text": ("The Receiving Party shall maintain the confidential information in strict secrecy in perpetuity. "
                 "Any breach shall give rise to automatic liquidated damages of INR 50 Lakhs payable immediately, "
                 "with no limitation of liability for consequential damages.")
    },
    {
        "domain": "NDA — Confidentiality (LOW)",
        "text": ("Both parties agree to keep each other's proprietary information confidential "
                 "for a period of 2 years from the date of disclosure.")
    },
    # ── PROPERTY / SALE ─────────────────────────────────────────────────────
    {
        "domain": "Property — Indemnity (HIGH)",
        "text": ("The Vendor shall indemnify the Purchaser against all losses, damages, and liabilities "
                 "arising out of any defect in title, without any cap or limitation whatsoever.")
    },
    {
        "domain": "Property — Stamp Duty (LOW)",
        "text": ("All stamp duty and registration charges payable on this agreement shall be borne equally "
                 "by both parties in accordance with the applicable law.")
    },
    # ── LOAN / FINANCE ──────────────────────────────────────────────────────
    {
        "domain": "Loan — Penalty (HIGH)",
        "text": ("Any default in repayment shall attract a compounded penalty interest of 24% per annum from the date of default. "
                 "The Lender shall have the right to seize all movable and immovable assets of the Borrower.")
    },
    {
        "domain": "Loan — Repayment (LOW)",
        "text": ("The Borrower shall repay the loan in 36 equal monthly instalments at an interest rate of 10% per annum. "
                 "A grace period of 7 days shall apply for late payments.")
    },
    # ── CORPORATE / COMMERCIAL ──────────────────────────────────────────────
    {
        "domain": "Corporate — Governing Law (LOW)",
        "text": ("This Agreement shall be governed by and construed in accordance with the laws of India. "
                 "Any disputes shall be subject to the exclusive jurisdiction of the courts in Mumbai.")
    },
    {
        "domain": "Corporate — Limitation of Liability (HIGH)",
        "text": ("Under no circumstances shall either party's liability be limited. "
                 "Both parties shall be jointly and severally liable for all direct, indirect, consequential, "
                 "and punitive damages without any cap or restriction.")
    },
]


# ────────────────────────────────────────────────────────────────────────────
# 5.  Run the integration test
# ────────────────────────────────────────────────────────────────────────────

def run_test():
    print(f"\nUsing device: {DEVICE}\n")

    c_tok, c_model, idx_to_label = load_clause_classifier()
    r_tok, r_model               = load_risk_model()

    print("\n" + "="*95)
    print(f"{'Domain / Expected':<42} {'Clause Type':<30} {'Risk Score':>10}  {'Level'}")
    print("="*95)

    rows = []
    for item in TEST_CLAUSES:
        c_type, c_conf = classify_clause(item["text"], c_tok, c_model, idx_to_label)
        r_score, r_level = predict_risk(item["text"], r_tok, r_model)
        print(f"{item['domain']:<42} {c_type:<30} {r_score:>10.2f}   {r_level}")
        rows.append({"domain": item["domain"], "clause_type": c_type,
                     "confidence": round(c_conf, 3), "risk_score": r_score, "risk_level": r_level})

    print("="*95)
    return rows


if __name__ == "__main__":
    run_test()
    print("\n[DONE] Integration test complete.")
