import os
import sys
import random
import logging
import json
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from torch.optim import AdamW

# Resolve paths
script_dir = os.path.dirname(os.path.abspath(__file__))
ml_dir = os.path.dirname(script_dir)
project_root = os.path.dirname(ml_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from ml.evaluation.evaluate_pretrained_risk_models import MultiTaskLegalModel

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class RiskRegressionDataset(Dataset):
    def __init__(self, texts, risk_targets, sample_weights, tokenizer, max_len):
        self.texts = texts
        self.risk_targets = risk_targets
        self.sample_weights = sample_weights
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        target = self.risk_targets[idx]
        weight = self.sample_weights[idx]

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "risk_target": torch.tensor(target, dtype=torch.float),
            "sample_weight": torch.tensor(weight, dtype=torch.float)
        }

def train_variant(variant_name, train_df, val_df, unfreeze_encoder_layers, lr_head, lr_encoder, device):
    logger.info(f"=== Starting training for variant {variant_name} ===")
    
    set_seed(42)
    
    repo_id = "AnkushRaheja/Cls_Class_Risk_Scr"
    from huggingface_hub import hf_hub_download
    metadata_path = hf_hub_download(repo_id=repo_id, filename="metadata.json")
    model_weights_path = hf_hub_download(repo_id=repo_id, filename="full_model.pt")
    
    with open(metadata_path, "r") as f:
        meta = json.load(f)
        
    tokenizer = AutoTokenizer.from_pretrained(repo_id)
    model = MultiTaskLegalModel("nlpaueb/legal-bert-base-uncased", meta["num_labels"])
    model.load_state_dict(torch.load(model_weights_path, map_location=device))
    model = model.to(device)
    
    # Freeze/Unfreeze logic
    # First freeze all BERT layers
    for param in model.bert.parameters():
        param.requires_grad = False
        
    if unfreeze_encoder_layers:
        logger.info("Unfreezing final BERT encoder layer (layer 11)...")
        for param in model.bert.encoder.layer[11].parameters():
            param.requires_grad = True
            
    # Set up optimizer parameters with different learning rates
    params = []
    # Regressor head parameters
    head_params = list(model.regressor.parameters())
    params.append({"params": head_params, "lr": lr_head})
    
    if unfreeze_encoder_layers:
        encoder_params = list(model.bert.encoder.layer[11].parameters())
        params.append({"params": encoder_params, "lr": lr_encoder})
        
    optimizer = AdamW(params)
    
    # Dataloaders
    max_len = 256
    batch_size = 16
    epochs = 5
    
    # Map risk levels to continuous targets (LOW -> 0.25, MEDIUM -> 0.50, HIGH -> 0.75)
    target_mapping = {"low": 0.25, "medium": 0.50, "high": 0.75}
    # Class weights based on train.csv risk level counts: LOW=1.69, MEDIUM=0.94, HIGH=0.74
    weight_mapping = {"low": 1.69, "medium": 0.94, "high": 0.74}
    
    train_targets = train_df["risk_level"].map(target_mapping).tolist()
    train_weights = train_df["risk_level"].map(weight_mapping).tolist()
    
    val_targets = val_df["risk_level"].map(target_mapping).tolist()
    val_weights = val_df["risk_level"].map(weight_mapping).tolist()
    
    train_dataset = RiskRegressionDataset(
        texts=train_df["clause_text"].tolist(),
        risk_targets=train_targets,
        sample_weights=train_weights,
        tokenizer=tokenizer,
        max_len=max_len
    )
    
    val_dataset = RiskRegressionDataset(
        texts=val_df["clause_text"].tolist(),
        risk_targets=val_targets,
        sample_weights=val_weights,
        tokenizer=tokenizer,
        max_len=max_len
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )
    
    # Huber Loss is robust to outliers
    huber_loss = nn.HuberLoss(delta=0.1, reduction="none")
    
    best_val_loss = float("inf")
    output_model_dir = os.path.join(ml_dir, "models", "experiment6_model", variant_name)
    os.makedirs(output_model_dir, exist_ok=True)
    
    # Save meta & mappings
    with open(os.path.join(output_model_dir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=4)
        
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
            
            # Weighted Huber Loss
            raw_loss = huber_loss(risk_outputs, targets)
            loss = (raw_loss * weights).mean()
            
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            
            total_train_loss += loss.item()
            
        avg_train_loss = total_train_loss / len(train_loader)
        
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
                
        avg_val_loss = total_val_loss / len(val_loader)
        logger.info(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            logger.info(f"Saving best weights to {output_model_dir}...")
            torch.save(model.state_dict(), os.path.join(output_model_dir, "pytorch_model.bin"))
            tokenizer.save_pretrained(output_model_dir)
            
    logger.info(f"Variant {variant_name} complete.")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load data
    train_path = os.path.join(ml_dir, "data", "splits", "train.csv")
    val_path = os.path.join(ml_dir, "data", "splits", "validation.csv")
    contrastive_train_path = os.path.join(ml_dir, "data", "experiment5_contrastive_train.csv")
    
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    contrastive_df = pd.read_csv(contrastive_train_path)
    
    # 6A: Original training only, frozen encoder
    train_variant(
        variant_name="6a",
        train_df=train_df,
        val_df=val_df,
        unfreeze_encoder_layers=False,
        lr_head=2e-4,
        lr_encoder=0.0,
        device=device
    )
    
    # 6B: Combined training (original + contrastive), frozen encoder
    combined_train_df = pd.concat([train_df, contrastive_df], ignore_index=True)
    train_variant(
        variant_name="6b",
        train_df=combined_train_df,
        val_df=val_df,
        unfreeze_encoder_layers=False,
        lr_head=2e-4,
        lr_encoder=0.0,
        device=device
    )
    
    # 6C: Combined training, unfreeze last layer (layer 11) with smaller LR
    train_variant(
        variant_name="6c",
        train_df=combined_train_df,
        val_df=val_df,
        unfreeze_encoder_layers=True,
        lr_head=2e-4,
        lr_encoder=2e-6,
        device=device
    )

if __name__ == "__main__":
    main()
