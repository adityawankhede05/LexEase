import os
import re
import pandas as pd
import numpy as np
from datasets import load_dataset

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r'\s+', ' ', text)
    text = text.lower().strip()
    return text

def is_duplicate(text, text_list, threshold=0.85):
    """
    Checks if text is a near-duplicate of any text in text_list using Jaccard similarity.
    """
    c_text = clean_text(text)
    words1 = set(c_text.split())
    if not words1:
        return False
    for other in text_list:
        c_other = clean_text(other)
        words2 = set(c_other.split())
        if not words2:
            continue
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        sim = len(intersection) / len(union)
        if sim >= threshold:
            return True
    return False

def extract_template_clauses():
    """
    Extracts paragraphs from d-riti template PDFs and labels them as LOW risk.
    """
    ds = load_dataset('d-riti/Dataset-For-Indian-legal-knowledge-base')
    records = []
    
    # Let's define the indices of standard templates
    # Row 18: NondisclosureAgreement.pdf (NDA)
    # Row 24: Vendor Aggrement_Update-d359f7b0.pdf (Vendor)
    # Row 25: Vendor-Agreement-Sample.pdf (Vendor)
    # Row 31: sample-employment-agreement-template.pdf (Employment)
    
    templates = [
        {"idx": 18, "domain": "NDA", "clause_type": "Confidentiality", "source": "NondisclosureAgreement.pdf"},
        {"idx": 24, "domain": "Vendor/Supply", "clause_type": "Payment", "source": "Vendor Aggrement_Update.pdf"},
        {"idx": 25, "domain": "Vendor/Supply", "clause_type": "Payment", "source": "Vendor-Agreement-Sample.pdf"},
        {"idx": 31, "domain": "Employment", "clause_type": "Termination", "source": "sample-employment-agreement.pdf"}
    ]
    
    for t in templates:
        pdf = ds['train'][t['idx']]['pdf']
        full_text = ""
        for page in pdf.pages:
            p_text = page.extract_text()
            if p_text:
                full_text += p_text + "\n"
        
        # Split text into paragraphs
        paragraphs = full_text.split('\n\n')
        for para in paragraphs:
            para = para.strip()
            # Keep paragraphs that look like actual clauses (between 15 and 150 words)
            word_count = len(para.split())
            if 15 <= word_count <= 150:
                # Basic cleaning: remove section numbers, bracket placeholders like [Vendor's Name]
                cleaned = re.sub(r'^(\d+\.|\d+\.\d+|\-\s*)', '', para).strip()
                # Determine clause type based on content search
                c_type = t['clause_type']
                if "confidential" in cleaned.lower() or "disclosure" in cleaned.lower():
                    c_type = "Confidentiality"
                elif "terminate" in cleaned.lower() or "notice" in cleaned.lower():
                    c_type = "Termination"
                elif "payment" in cleaned.lower() or "invoice" in cleaned.lower() or "fee" in cleaned.lower():
                    c_type = "Payment"
                elif "indemn" in cleaned.lower():
                    c_type = "Indemnity"
                elif "liab" in cleaned.lower() or "limit" in cleaned.lower():
                    c_type = "Limitation of Liability"
                elif "arbitrat" in cleaned.lower() or "dispute" in cleaned.lower():
                    c_type = "Arbitration"
                elif "govern" in cleaned.lower() or "jurisdiction" in cleaned.lower():
                    c_type = "Governing Law"
                elif "intellectual" in cleaned.lower() or "ip" in cleaned.lower() or "patent" in cleaned.lower():
                    c_type = "Intellectual Property"
                
                records.append({
                    "clause_text": cleaned,
                    "clause_type": c_type,
                    "domain": t['domain'],
                    "risk_level": "LOW", # Standard template clauses are generally low/standard risk
                    "source": f"d-riti/{t['source']}",
                    "is_synthetic": False
                })
                
    df = pd.DataFrame(records)
    print(f"Extracted {len(df)} clauses from d-riti template agreements.")
    return df

def main():
    print("Preparing final multidomain risk training data...")
    
    # 1. Load real-world validation set (HELD OUT — DO NOT USE FOR TRAINING)
    vc = pd.read_csv('ml/data/real_world_validation/validation_clauses.csv')
    val_texts = vc['clause_text'].tolist()
    print(f"Loaded {len(val_texts)} held-out validation clauses to prevent leakage.")
    
    # 2. Parse Risky Patterns
    risky_pat_df = pd.read_csv('driti_parsed_risky_patterns.csv')
    print(f"Loaded {len(risky_pat_df)} clauses parsed from Risky Patterns PDF.")
    
    # 3. Extract template clauses
    templates_df = extract_template_clauses()
    
    # Combine new Indian data
    new_indian_df = pd.concat([risky_pat_df, templates_df], ignore_index=True)
    print(f"Total new Indian-domain pool: {len(new_indian_df)}")
    
    # Filter out any duplicates or near-duplicates of the held-out validation clauses
    clean_new_indian = []
    dup_count = 0
    for _, row in new_indian_df.iterrows():
        if is_duplicate(row['clause_text'], val_texts, threshold=0.85):
            dup_count += 1
            continue
        clean_new_indian.append(row)
    
    clean_new_indian_df = pd.DataFrame(clean_new_indian)
    print(f"Filtered out {dup_count} validation leakage clauses. Clean new Indian pool: {len(clean_new_indian_df)}")
    
    # 4. Load existing splits/train.csv (corporate)
    train_corp = pd.read_csv('ml/data/splits/train.csv')
    
    # Sample from splits/train.csv to keep corporate vs Indian balanced
    # splits/train.csv has 7,557 rows. Let's sample 1,000 corporate rows (balanced across risk levels)
    corp_low = train_corp[train_corp['risk_level'] == 'low'].sample(n=350, random_state=42)
    corp_med = train_corp[train_corp['risk_level'] == 'medium'].sample(n=350, random_state=42)
    corp_high = train_corp[train_corp['risk_level'] == 'high'].sample(n=300, random_state=42)
    corp_sampled = pd.concat([corp_low, corp_med, corp_high], ignore_index=True)
    
    # Standardize columns
    corp_sampled = corp_sampled[['clause_text', 'clause_type', 'risk_level']].copy()
    corp_sampled['domain'] = "Original Corporate"
    corp_sampled['source'] = "splits/train.csv"
    corp_sampled['is_synthetic'] = False
    
    # 5. Load experiment5_contrastive_train.csv
    exp5 = pd.read_csv('ml/data/experiment5_contrastive_train.csv')
    exp5 = exp5[['clause_text', 'clause_type', 'risk_level']].copy()
    exp5['domain'] = "Original Corporate"
    exp5['source'] = "splits/experiment5_contrastive_train.csv"
    exp5['is_synthetic'] = True
    
    # Combine all training pool
    train_pool = pd.concat([clean_new_indian_df, corp_sampled, exp5], ignore_index=True)
    print(f"Combined training pool size: {len(train_pool)}")
    
    # De-duplicate the entire training pool internally (prevent duplicate leakage between splits)
    unique_pool = []
    seen_texts = []
    pool_dup_count = 0
    for _, row in train_pool.iterrows():
        if is_duplicate(row['clause_text'], seen_texts, threshold=0.90):
            pool_dup_count += 1
            continue
        unique_pool.append(row)
        seen_texts.append(row['clause_text'])
    
    unique_pool_df = pd.DataFrame(unique_pool)
    print(f"Internally de-duplicated pool. Removed {pool_dup_count} duplicate rows. Final unique pool size: {len(unique_pool_df)}")
    
    # Map risk levels to uppercase
    unique_pool_df['risk_level'] = unique_pool_df['risk_level'].str.upper()
    print("Final risk distribution:\n", unique_pool_df['risk_level'].value_counts())
    
    # Save the final combined dataset
    unique_pool_df.to_csv('ml/data/final_multidomain_risk_dataset.csv', index=False)
    print("Saved final multidomain risk dataset to ml/data/final_multidomain_risk_dataset.csv")

if __name__ == "__main__":
    main()
