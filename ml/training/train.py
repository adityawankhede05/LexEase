import os
import sys
import yaml
import json
import random
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, f1_score

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def set_seed(seed: int) -> None:
    """Sets random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class MultiTaskTransformer(nn.Module):
    """
    Multi-Task Transformer Model for Legal Clause Classification.
    Shared encoder with two independent classification heads.
    """
    def __init__(self, model_name: str, num_clause_types: int, num_risk_levels: int):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        
        # Classification heads
        self.clause_type_head = nn.Linear(hidden_size, num_clause_types)
        self.risk_level_head = nn.Linear(hidden_size, num_risk_levels)
        
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # Use first token [CLS] representation
        cls_output = outputs.last_hidden_state[:, 0, :]
        
        clause_type_logits = self.clause_type_head(cls_output)
        risk_level_logits = self.risk_level_head(cls_output)
        
        return clause_type_logits, risk_level_logits

class ClauseDataset(Dataset):
    """PyTorch Dataset for processing legal clause texts and labels."""
    def __init__(self, texts: list, clause_types: list, risk_levels: list, tokenizer, max_len: int):
        self.texts = texts
        self.clause_types = clause_types
        self.risk_levels = risk_levels
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __len__(self):
        return len(self.texts)
        
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "clause_type": torch.tensor(self.clause_types[idx], dtype=torch.long),
            "risk_level": torch.tensor(self.risk_levels[idx], dtype=torch.long)
        }

def train_model() -> None:
    # 1. Load Configurations
    script_dir = os.path.dirname(os.path.abspath(__file__))
    ml_dir = os.path.dirname(script_dir)
    config_path = os.path.join(ml_dir, "config", "config.yaml")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    set_seed(config["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Resolve dataset paths
    train_path = os.path.join(ml_dir, "data", "splits", "train.csv")
    val_path = os.path.join(ml_dir, "data", "splits", "validation.csv")

    logger.info(f"Loading train/validation splits:\n - Train: {train_path}\n - Validation: {val_path}")
    if not os.path.exists(train_path) or not os.path.exists(val_path):
        logger.error("Dataset split files do not exist at splits/. Cannot train model.")
        sys.exit(1)

    # 2. Load Datasets
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)

    # Validate columns
    required_cols = {"clause_text", "clause_type", "risk_level"}
    if not required_cols.issubset(train_df.columns) or not required_cols.issubset(val_df.columns):
        logger.error(f"Missing required columns in dataset splits. Required: {required_cols}")
        sys.exit(1)

    # 3. Create Label Mappings from existing target columns
    clause_type_to_idx = {
        str(k): int(v)
        for k, v in train_df.groupby("clause_type")["clause_type_target"].first().items()
    }
    risk_level_to_idx = {
        str(k): int(v)
        for k, v in train_df.groupby("risk_level")["risk_level_target"].first().items()
    }

    logger.info(f"Loaded mappings for {len(clause_type_to_idx)} unique clause types and {len(risk_level_to_idx)} risk levels.")

    # Convert target columns to standard integers
    train_df["clause_type_target"] = train_df["clause_type_target"].astype(int)
    train_df["risk_level_target"] = train_df["risk_level_target"].astype(int)
    val_df["clause_type_target"] = val_df["clause_type_target"].astype(int)
    val_df["risk_level_target"] = val_df["risk_level_target"].astype(int)

    # 4. Tokenizer & Dataloaders
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    
    train_dataset = ClauseDataset(
        texts=train_df["clause_text"].tolist(),
        clause_types=train_df["clause_type_target"].tolist(),
        risk_levels=train_df["risk_level_target"].tolist(),
        tokenizer=tokenizer,
        max_len=config["max_len"]
    )
    
    val_dataset = ClauseDataset(
        texts=val_df["clause_text"].tolist(),
        clause_types=val_df["clause_type_target"].tolist(),
        risk_levels=val_df["risk_level_target"].tolist(),
        tokenizer=tokenizer,
        max_len=config["max_len"]
    )

    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)

    # 5. Initialize Model
    model = MultiTaskTransformer(
        model_name=config["model_name"],
        num_clause_types=len(clause_type_to_idx),
        num_risk_levels=len(risk_level_to_idx)
    )
    model = model.to(device)

    # Loss components
    criterion_clause = nn.CrossEntropyLoss()
    criterion_risk = nn.CrossEntropyLoss()

    optimizer = AdamW(model.parameters(), lr=float(config["learning_rate"]))
    
    total_steps = len(train_loader) * config["epochs"]
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )

    best_val_loss = float("inf")
    output_model_dir = os.path.join(ml_dir, config["output_dir"])
    os.makedirs(output_model_dir, exist_ok=True)

    # Save mappings
    with open(os.path.join(output_model_dir, "clause_type_mapping.json"), "w") as f:
        json.dump(clause_type_to_idx, f, indent=2)
    with open(os.path.join(output_model_dir, "risk_level_mapping.json"), "w") as f:
        json.dump(risk_level_to_idx, f, indent=2)

    logger.info("Starting training loop...")
    w_clause = config["loss_weights"]["clause_type"]
    w_risk = config["loss_weights"]["risk_level"]

    for epoch in range(config["epochs"]):
        model.train()
        total_train_loss = 0.0

        for step, batch in enumerate(train_loader):
            optimizer.zero_grad()
            
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            clause_targets = batch["clause_type"].to(device)
            risk_targets = batch["risk_level"].to(device)

            clause_logits, risk_logits = model(input_ids, attention_mask)

            loss_c = criterion_clause(clause_logits, clause_targets)
            loss_r = criterion_risk(risk_logits, risk_targets)
            loss = w_clause * loss_c + w_risk * loss_r

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            total_train_loss += loss.item()

            if (step + 1) % 20 == 0 or (step + 1) == len(train_loader):
                logger.info(f"Epoch {epoch+1} | Step {step+1}/{len(train_loader)} | Loss: {loss.item():.4f}")

        avg_train_loss = total_train_loss / len(train_loader)

        # Validation
        model.eval()
        total_val_loss = 0.0
        val_clause_preds, val_clause_targets = [], []
        val_risk_preds, val_risk_targets = [], []

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                clause_targets = batch["clause_type"].to(device)
                risk_targets = batch["risk_level"].to(device)

                clause_logits, risk_logits = model(input_ids, attention_mask)

                loss_c = criterion_clause(clause_logits, clause_targets)
                loss_r = criterion_risk(risk_logits, risk_targets)
                loss = w_clause * loss_c + w_risk * loss_r
                total_val_loss += loss.item()

                val_clause_preds.extend(torch.argmax(clause_logits, dim=1).cpu().numpy())
                val_clause_targets.extend(clause_targets.cpu().numpy())
                val_risk_preds.extend(torch.argmax(risk_logits, dim=1).cpu().numpy())
                val_risk_targets.extend(risk_targets.cpu().numpy())

        avg_val_loss = total_val_loss / len(val_loader)
        
        # Calculate metrics
        clause_acc = accuracy_score(val_clause_targets, val_clause_preds)
        clause_f1 = f1_score(val_clause_targets, val_clause_preds, average="macro", zero_division=0)
        
        risk_acc = accuracy_score(val_risk_targets, val_risk_preds)
        risk_f1 = f1_score(val_risk_targets, val_risk_preds, average="macro", zero_division=0)

        logger.info(
            f"Epoch {epoch+1}/{config['epochs']} | "
            f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}\n"
            f" - Clause Type Acc: {clause_acc:.4f} F1: {clause_f1:.4f}\n"
            f" - Risk Level Acc: {risk_acc:.4f} F1: {risk_f1:.4f}"
        )

        # Checkpoint Saving
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            logger.info(f"Validation loss improved. Saving model model weights to {output_model_dir}...")
            
            # Save Model
            torch.save(model.state_dict(), os.path.join(output_model_dir, "pytorch_model.bin"))
            # Save Tokenizer
            tokenizer.save_pretrained(output_model_dir)
            # Save config copy
            with open(os.path.join(output_model_dir, "config.json"), "w") as f:
                json.dump(config, f, indent=2)

    logger.info("Training complete.")

if __name__ == "__main__":
    train_model()
