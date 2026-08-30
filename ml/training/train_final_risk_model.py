import os
import sys
import random
import json
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from torch.optim import AdamW
from huggingface_hub import hf_hub_download
from sklearn.model_selection import train_test_split

# Resolve paths
script_dir = os.path.dirname(os.path.abspath(__file__))
ml_dir = os.path.dirname(script_dir)
project_root = os.path.dirname(ml_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from ml.evaluation.evaluate_pretrained_risk_models import MultiTaskLegalModel
from ml.training.train_experiment6 import RiskRegressionDataset

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load final multidomain training data
    dataset_path = os.path.join(ml_dir, "data", "final_multidomain_risk_dataset.csv")
    df = pd.read_csv(dataset_path)
    print(f"Loaded final multidomain dataset of size: {len(df)}")
    
    # Stratified split to keep training and validation risk distributions balanced
    train_df, val_df = train_test_split(
        df,
        test_size=0.20,
        random_state=42,
        stratify=df['risk_level']
    )
    
    print(f"Train size: {len(train_df)} | Val size: {len(val_df)}")
    
    # Target mapping: map low/medium/high to continuous targets [0, 1]
    target_mapping = {"LOW": 0.25, "MEDIUM": 0.50, "HIGH": 0.75}
    
    # Risk category weights to balance training (optional sample weights)
    # Counts in dataset are: LOW: 408, MEDIUM: 394, HIGH: 374 (very balanced already)
    weight_mapping = {"LOW": 1.0, "MEDIUM": 1.0, "HIGH": 1.0}
    
    train_targets = train_df["risk_level"].map(target_mapping).tolist()
    train_weights = train_df["risk_level"].map(weight_mapping).tolist()
    
    val_targets = val_df["risk_level"].map(target_mapping).tolist()
    val_weights = val_df["risk_level"].map(weight_mapping).tolist()
    
    # Load pretrained tokenizer and model configuration from AnkushRaheja
    repo_id = "AnkushRaheja/Cls_Class_Risk_Scr"
    metadata_path = hf_hub_download(repo_id=repo_id, filename="metadata.json")
    model_weights_path = hf_hub_download(repo_id=repo_id, filename="full_model.pt")
    
    with open(metadata_path, "r") as f:
        meta = json.load(f)
        
    tokenizer = AutoTokenizer.from_pretrained(repo_id)
    model = MultiTaskLegalModel("nlpaueb/legal-bert-base-uncased", meta["num_labels"])
    model.load_state_dict(torch.load(model_weights_path, map_location=device))
    model = model.to(device)
    
    # Freeze Legal-BERT encoder
    for param in model.bert.parameters():
        param.requires_grad = False
        
    # Datasets and loaders
    train_dataset = RiskRegressionDataset(
        texts=train_df["clause_text"].tolist(),
        risk_targets=train_targets,
        sample_weights=train_weights,
        tokenizer=tokenizer,
        max_len=256
    )
    
    val_dataset = RiskRegressionDataset(
        texts=val_df["clause_text"].tolist(),
        risk_targets=val_targets,
        sample_weights=val_weights,
        tokenizer=tokenizer,
        max_len=256
    )
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    
    # Optimizer for regressor head only
    optimizer = AdamW(model.regressor.parameters(), lr=1e-3, weight_decay=0.01)
    
    epochs = 5
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )
    
    huber_loss = nn.HuberLoss(delta=0.1, reduction="none")
    best_val_loss = float("inf")
    
    model_save_dir = os.path.join(ml_dir, "models", "final_risk_model")
    os.makedirs(model_save_dir, exist_ok=True)
    
    # Save metadata.json
    with open(os.path.join(model_save_dir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=4)
        
    print("Starting final risk model training (regression head only)...")
    
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0.0
        for step, batch in enumerate(train_loader):
            optimizer.zero_grad()
            
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            targets = batch["risk_target"].to(device)
            weights = batch["sample_weight"].to(device)
            
            _, risk_outputs = model(input_ids, attention_mask)
            raw_loss = huber_loss(risk_outputs, targets)
            loss = (raw_loss * weights).mean()
            
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            
            total_train_loss += loss.item()
            
        # Validation
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                targets = batch["risk_target"].to(device)
                weights = batch["sample_weight"].to(device)
                
                _, risk_outputs = model(input_ids, attention_mask)
                raw_loss = huber_loss(risk_outputs, targets)
                loss = (raw_loss * weights).mean()
                total_val_loss += loss.item()
                
        avg_train_loss = total_train_loss / len(train_loader)
        avg_val_loss = total_val_loss / len(val_loader)
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            print("Saving best weights...")
            torch.save(model.state_dict(), os.path.join(model_save_dir, "pytorch_model.bin"))
            tokenizer.save_pretrained(model_save_dir)
            
    print("Training of Final Risk Model complete.")

if __name__ == "__main__":
    main()
