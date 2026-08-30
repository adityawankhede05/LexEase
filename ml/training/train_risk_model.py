import os
import sys
import random
import logging
import yaml
import json
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup, BertModel
from torch.optim import AdamW

# Resolve paths
script_dir = os.path.dirname(os.path.abspath(__file__))
ml_dir = os.path.dirname(script_dir)
project_root = os.path.dirname(ml_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class RiskDataset(Dataset):
    def __init__(self, texts, risk_levels, tokenizer, max_len):
        self.texts = texts
        self.risk_levels = risk_levels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        risk = self.risk_levels[idx]

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
            "risk_level": torch.tensor(risk, dtype=torch.long)
        }

class RiskTransformer(nn.Module):
    def __init__(self, model_name, num_risk_levels):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.drop = nn.Dropout(p=0.3)
        self.risk_classifier = nn.Linear(self.bert.config.hidden_size, num_risk_levels)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]
        pooled_output = self.drop(pooled_output)
        risk_logits = self.risk_classifier(pooled_output)
        return risk_logits

def train_model() -> None:
    # 1. Load Configurations
    config_path = os.path.join(ml_dir, "config", "config_exp4.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    set_seed(config.get("seed", 42))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    train_path = os.path.join(ml_dir, "data", "splits", "train.csv")
    val_path = os.path.join(ml_dir, "data", "splits", "validation.csv")

    if not os.path.exists(train_path) or not os.path.exists(val_path):
        logger.error("Dataset split files do not exist. Cannot train.")
        sys.exit(1)

    # 2. Load Datasets
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)

    # Convert target columns to standard integers
    train_df["risk_level_target"] = train_df["risk_level_target"].astype(int)
    val_df["risk_level_target"] = val_df["risk_level_target"].astype(int)

    # 3. Create Label Mappings
    risk_level_to_idx = {
        str(k): int(v)
        for k, v in train_df.groupby("risk_level")["risk_level_target"].first().items()
    }
    logger.info(f"Loaded mappings: {risk_level_to_idx}")

    # 4. Calculate Class Weights for risk_level ONLY from train_df
    num_risk_classes = len(risk_level_to_idx)
    risk_counts = np.zeros(num_risk_classes)
    for val in train_df["risk_level_target"].tolist():
        risk_counts[val] += 1
    risk_counts = np.maximum(risk_counts, 1.0) # Prevent division by zero
    risk_class_weights = len(train_df) / (num_risk_classes * risk_counts)
    
    # Normalize weights so they average to 1.0
    risk_class_weights = risk_class_weights / np.mean(risk_class_weights)
    risk_class_weights_tensor = torch.FloatTensor(risk_class_weights).to(device)

    logger.info(f"Calculated risk level class weights: {risk_class_weights}")

    # 5. Tokenizer & Dataloaders
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    
    train_dataset = RiskDataset(
        texts=train_df["clause_text"].tolist(),
        risk_levels=train_df["risk_level_target"].tolist(),
        tokenizer=tokenizer,
        max_len=config["max_len"]
    )
    
    val_dataset = RiskDataset(
        texts=val_df["clause_text"].tolist(),
        risk_levels=val_df["risk_level_target"].tolist(),
        tokenizer=tokenizer,
        max_len=config["max_len"]
    )

    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)

    # 6. Initialize Model
    model = RiskTransformer(
        model_name=config["model_name"],
        num_risk_levels=len(risk_level_to_idx)
    )
    model = model.to(device)

    # Apply class weights to risk classification
    criterion_risk = nn.CrossEntropyLoss(weight=risk_class_weights_tensor)

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
    with open(os.path.join(output_model_dir, "risk_level_mapping.json"), "w") as f:
        json.dump(risk_level_to_idx, f, indent=4)

    # Save config
    with open(os.path.join(output_model_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=4)

    logger.info("Starting training loop...")
    for epoch in range(config["epochs"]):
        model.train()
        total_train_loss = 0.0

        for step, batch in enumerate(train_loader):
            optimizer.zero_grad()
            
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            risk_targets = batch["risk_level"].to(device)

            risk_logits = model(input_ids, attention_mask)
            loss = criterion_risk(risk_logits, risk_targets)

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
        val_risk_preds, val_risk_targets = [], []

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                risk_targets = batch["risk_level"].to(device)

                risk_logits = model(input_ids, attention_mask)
                loss = criterion_risk(risk_logits, risk_targets)

                total_val_loss += loss.item()

                val_risk_preds.extend(torch.argmax(risk_logits, dim=1).cpu().numpy())
                val_risk_targets.extend(risk_targets.cpu().numpy())

        avg_val_loss = total_val_loss / len(val_loader)
        
        # Calculate Epoch metrics
        val_risk_acc = np.mean(np.array(val_risk_preds) == np.array(val_risk_targets))
        
        from sklearn.metrics import f1_score
        val_risk_f1 = f1_score(val_risk_targets, val_risk_preds, average="macro", zero_division=0)

        logger.info(
            f"Epoch {epoch+1}/{config['epochs']} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f}"
        )
        logger.info(f" - Risk Level Acc: {val_risk_acc:.4f} F1: {val_risk_f1:.4f}")

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            logger.info(f"Validation loss improved. Saving model weights to {output_model_dir}...")
            torch.save(model.state_dict(), os.path.join(output_model_dir, "pytorch_model.bin"))
            tokenizer.save_pretrained(output_model_dir)

    logger.info("Training complete.")

if __name__ == "__main__":
    train_model()
