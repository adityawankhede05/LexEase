import os
import sys
import json
import logging
from typing import Dict, Any, Optional
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

# Resolve project paths for module importing
script_dir = os.path.dirname(os.path.abspath(__file__))
ml_dir = os.path.dirname(script_dir)
project_root = os.path.dirname(ml_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import model definition
from ml.training.train import MultiTaskTransformer

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Cache variables
_MODEL: Optional[MultiTaskTransformer] = None
_TOKENIZER: Optional[AutoTokenizer] = None
_CLAUSE_TYPE_MAPPING: Optional[Dict[str, int]] = None
_RISK_LEVEL_MAPPING: Optional[Dict[str, int]] = None
_IDX_TO_CLAUSE_TYPE: Optional[Dict[int, str]] = None
_IDX_TO_RISK_LEVEL: Optional[Dict[int, str]] = None
_CONFIG: Optional[Dict[str, Any]] = None

def load_resources() -> bool:
    """Loads and caches the model, tokenizer, config, and mappings."""
    global _MODEL, _TOKENIZER, _CLAUSE_TYPE_MAPPING, _RISK_LEVEL_MAPPING, _IDX_TO_CLAUSE_TYPE, _IDX_TO_RISK_LEVEL, _CONFIG
    
    if _MODEL is not None:
        return True

    model_dir = os.path.join(ml_dir, "models", "best_model")
    config_path = os.path.join(model_dir, "config.json")
    clause_mapping_path = os.path.join(model_dir, "clause_type_mapping.json")
    risk_mapping_path = os.path.join(model_dir, "risk_level_mapping.json")
    weights_path = os.path.join(model_dir, "pytorch_model.bin")

    if not all(os.path.exists(p) for p in [config_path, clause_mapping_path, risk_mapping_path, weights_path]):
        logger.warning(f"Model artifacts not found in {model_dir}. Runner will use fallback mode.")
        return False

    try:
        # Load config and mappings
        with open(config_path, "r") as f:
            _CONFIG = json.load(f)
        with open(clause_mapping_path, "r") as f:
            _CLAUSE_TYPE_MAPPING = json.load(f)
        with open(risk_mapping_path, "r") as f:
            _RISK_LEVEL_MAPPING = json.load(f)

        _IDX_TO_CLAUSE_TYPE = {v: k for k, v in _CLAUSE_TYPE_MAPPING.items()}
        _IDX_TO_RISK_LEVEL = {v: k for k, v in _RISK_LEVEL_MAPPING.items()}

        # Load tokenizer and model
        _TOKENIZER = AutoTokenizer.from_pretrained(model_dir)
        _MODEL = MultiTaskTransformer(
            model_name=_CONFIG["model_name"],
            num_clause_types=len(_CLAUSE_TYPE_MAPPING),
            num_risk_levels=len(_RISK_LEVEL_MAPPING)
        )
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _MODEL.load_state_dict(torch.load(weights_path, map_location=device))
        _MODEL = _MODEL.to(device)
        _MODEL.eval()
        
        logger.info("Multi-task transformer resources loaded successfully.")
        return True
    except Exception as e:
        logger.error(f"Error loading model resources: {e}")
        return False

def predict_clause(text: str) -> Dict[str, Any]:
    """
    Accepts raw legal clause text and uses the multi-task transformer
    to predict clause type and risk level, returning softmax confidence scores.

    Args:
        text: Legal clause text string.

    Returns:
        Dictionary mapping prediction outputs.
    """
    if not load_resources():
        # Fallback dictionary if model weights do not exist yet
        return {
            "clause_text": text,
            "predicted_clause_type": "Unknown (Fallback)",
            "predicted_risk_level": "low",
            "confidence": 0.5,
            "clause_type_confidence": 0.5,
            "risk_level_confidence": 1.0,
            "is_fallback": True
        }

    try:
        device = next(_MODEL.parameters()).device
        encoding = _TOKENIZER(
            text,
            add_special_tokens=True,
            max_length=_CONFIG["max_len"],
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt"
        )

        input_ids = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)

        with torch.no_grad():
            clause_logits, risk_logits = _MODEL(input_ids, attention_mask)

            # Softmax probabilities
            clause_probs = F.softmax(clause_logits, dim=1).flatten()
            risk_probs = F.softmax(risk_logits, dim=1).flatten()

        # Class predictions
        pred_clause_idx = torch.argmax(clause_probs).item()
        pred_risk_idx = torch.argmax(risk_probs).item()

        # Probabilities
        clause_type_conf = float(clause_probs[pred_clause_idx].item())
        risk_level_conf = float(risk_probs[pred_risk_idx].item())

        predicted_clause_type = _IDX_TO_CLAUSE_TYPE.get(pred_clause_idx, "Unknown")
        predicted_risk_level = _IDX_TO_RISK_LEVEL.get(pred_risk_idx, "low")

        # Joint confidence score (product of both heads)
        joint_conf = clause_type_conf * risk_level_conf

        return {
            "clause_text": text,
            "predicted_clause_type": predicted_clause_type,
            "predicted_risk_level": predicted_risk_level,
            "confidence": round(joint_conf, 4),
            "clause_type_confidence": round(clause_type_conf, 4),
            "risk_level_confidence": round(risk_level_conf, 4),
            "is_fallback": False
        }
    except Exception as e:
        logger.error(f"Error during transformer prediction: {e}")
        return {
            "clause_text": text,
            "predicted_clause_type": "Unknown (Error Fallback)",
            "predicted_risk_level": "low",
            "confidence": 0.5,
            "clause_type_confidence": 0.5,
            "risk_level_confidence": 1.0,
            "is_fallback": True
        }

if __name__ == "__main__":
    sample = "The Licensee shall pay a monthly rent of INR 10,000 to the Licensor."
    res = predict_clause(sample)
    print(json.dumps(res, indent=2))
