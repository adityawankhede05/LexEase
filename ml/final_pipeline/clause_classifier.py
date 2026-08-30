import os
import sys
import json
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

# Resolve paths
script_dir = os.path.dirname(os.path.abspath(__file__))
ml_dir = os.path.dirname(script_dir)
project_root = os.path.dirname(ml_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from ml.training.train import MultiTaskTransformer
from ml.training.train_experiment8_indian_classifier import AdaptedClauseClassifier

class HierarchicalClauseClassifier:
    def __init__(self, device):
        self.device = device
        
        # Load Experiment 3 (Corporate Classifier)
        self.model_dir_exp3 = os.path.join(ml_dir, "models", "experiment3_model")
        with open(os.path.join(self.model_dir_exp3, "clause_type_mapping.json"), "r") as f:
            self.mapping_exp3 = json.load(f)
        self.idx_to_type_exp3 = {v: k for k, v in self.mapping_exp3.items()}
        
        self.tokenizer_exp3 = AutoTokenizer.from_pretrained("nlpaueb/legal-bert-base-uncased")
        self.model_exp3 = MultiTaskTransformer("nlpaueb/legal-bert-base-uncased", len(self.mapping_exp3), 3)
        self.model_exp3.load_state_dict(torch.load(os.path.join(self.model_dir_exp3, "pytorch_model.bin"), map_location=device))
        self.model_exp3 = self.model_exp3.to(device)
        self.model_exp3.eval()
        
        # Load Experiment 8 (Indian Legal Classifier)
        self.model_dir_exp8 = os.path.join(ml_dir, "models", "experiment8_indian_classifier")
        with open(os.path.join(self.model_dir_exp8, "clause_type_mapping.json"), "r") as f:
            self.mapping_exp8 = json.load(f)
        self.idx_to_type_exp8 = {v: k for k, v in self.mapping_exp8.items()}
        
        self.tokenizer_exp8 = AutoTokenizer.from_pretrained(self.model_dir_exp8)
        self.model_exp8 = AdaptedClauseClassifier(num_classes=len(self.mapping_exp8))
        self.model_exp8.load_state_dict(torch.load(os.path.join(self.model_dir_exp8, "pytorch_model.bin"), map_location=device))
        self.model_exp8 = self.model_exp8.to(device)
        self.model_exp8.eval()

    def classify_clause(self, text: str, domain: str):
        # Determine which model to route to
        if domain in ["Corporate/Commercial", "General/Other"]:
            # Route to Experiment 3
            inputs = self.tokenizer_exp3(text, return_tensors="pt", truncation=True, max_length=256, padding=True)
            input_ids = inputs["input_ids"].to(self.device)
            attention_mask = inputs["attention_mask"].to(self.device)
            
            with torch.no_grad():
                clause_logits, _ = self.model_exp3(input_ids, attention_mask)
                probs = F.softmax(clause_logits, dim=-1)
                conf, pred_idx = torch.max(probs, dim=-1)
                
            pred_idx = pred_idx.item()
            conf = conf.item()
            
            if conf < 0.25:
                return "UNKNOWN/OTHER", conf
            return self.idx_to_type_exp3[pred_idx], conf
        else:
            # Route to Experiment 8 (Indian Legal Adaptation)
            inputs = self.tokenizer_exp8(text, return_tensors="pt", truncation=True, max_length=256, padding=True)
            input_ids = inputs["input_ids"].to(self.device)
            attention_mask = inputs["attention_mask"].to(self.device)
            
            with torch.no_grad():
                outputs = self.model_exp8(input_ids, attention_mask)
                probs = F.softmax(outputs, dim=-1)
                conf, pred_idx = torch.max(probs, dim=-1)
                
            pred_idx = pred_idx.item()
            conf = conf.item()
            
            if conf < 0.25:
                return "UNKNOWN/OTHER", conf
            return self.idx_to_type_exp8[pred_idx], conf
