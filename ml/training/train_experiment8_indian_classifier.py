import os
import sys
import json
import random
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.model_selection import train_test_split

# Resolve paths
script_dir = os.path.dirname(os.path.abspath(__file__))
ml_dir = os.path.dirname(script_dir)
project_root = os.path.dirname(ml_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Set seeds
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class UnifiedClauseDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
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
            "label": torch.tensor(label, dtype=torch.long)
        }

class AdaptedClauseClassifier(nn.Module):
    def __init__(self, model_name="nlpaueb/legal-bert-base-uncased", num_classes=22, dropout_rate=0.25):
        super(AdaptedClauseClassifier, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(768, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]
        x = self.dropout(pooled_output)
        return self.classifier(x)

def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Load Adaptation Dataset
    dataset_path = os.path.join(ml_dir, "data", "experiment8_indian_clause_dataset.csv")
    df = pd.read_csv(dataset_path)
    
    # Create class mapping
    unique_classes = sorted(df["clause_type"].unique().tolist())
    class_mapping = {c: idx for idx, c in enumerate(unique_classes)}
    
    df["label"] = df["clause_type"].map(class_mapping)
    
    print(f"Dataset Size: {len(df)}")
    print(f"Number of Classes: {len(unique_classes)}")
    
    # Train/Test Split
    train_df, test_df = train_test_split(
        df, test_size=0.20, random_state=42, stratify=df["label"]
    )
    
    # Initialize tokenizer and dataset
    model_name = "nlpaueb/legal-bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    train_dataset = UnifiedClauseDataset(
        texts=train_df["clause_text"].tolist(),
        labels=train_df["label"].tolist(),
        tokenizer=tokenizer
    )
    
    test_dataset = UnifiedClauseDataset(
        texts=test_df["clause_text"].tolist(),
        labels=test_df["label"].tolist(),
        tokenizer=tokenizer
    )
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    
    # Estimate training time (BERT frozen, extremely fast)
    # Forward/backward on 65 batches of batch_size 16 is ~2 seconds on GPU, ~12 seconds on CPU.
    # Total time for 3 epochs is < 1 minute.
    print(f"Estimated training time on {device}: Under 1 minute (BERT frozen).")
    
    # Instantiate Model
    model = AdaptedClauseClassifier(model_name=model_name, num_classes=len(unique_classes))
    model = model.to(device)
    
    # Freeze encoder
    for param in model.bert.parameters():
        param.requires_grad = False
        
    optimizer = AdamW(model.classifier.parameters(), lr=5e-3, weight_decay=0.01)
    
    epochs = 15
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )
    
    loss_fn = nn.CrossEntropyLoss()
    
    # Training Loop
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for step, batch in enumerate(train_loader):
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            
            outputs = model(input_ids, attention_mask)
            loss = loss_fn(outputs, labels)
            
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{epochs} | Avg Loss: {total_loss / len(train_loader):.4f}")
        
    # Save Model Weights, Mapping & Tokenizer
    model_save_dir = os.path.join(ml_dir, "models", "experiment8_indian_classifier")
    os.makedirs(model_save_dir, exist_ok=True)
    
    torch.save(model.state_dict(), os.path.join(model_save_dir, "pytorch_model.bin"))
    tokenizer.save_pretrained(model_save_dir)
    
    with open(os.path.join(model_save_dir, "clause_type_mapping.json"), "w") as f:
        json.dump(class_mapping, f, indent=4)
        
    # Save test split for evaluation script
    test_df.to_csv(os.path.join(ml_dir, "data", "experiment8_test_split.csv"), index=False)
    print(f"Model saved successfully to {model_save_dir}.")

if __name__ == "__main__":
    main()
