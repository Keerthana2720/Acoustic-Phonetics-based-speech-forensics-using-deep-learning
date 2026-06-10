#!/usr/bin/env python
"""Test script to validate model loading and prediction"""
import os
import sys
import torch
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.model import MultiStreamForensics
import yaml

# Load config
config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
with open(config_path, 'r') as f:
    cfg = yaml.safe_load(f)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# Load model
model = MultiStreamForensics(cfg).to(device)
model.eval()

# Try to load checkpoint
checkpoint_path = os.path.join(os.path.dirname(__file__), 'checkpoints', 'audio_forensics.pt')
print(f"Checkpoint path: {checkpoint_path}")
print(f"Checkpoint exists: {os.path.exists(checkpoint_path)}")

if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    print(f"Checkpoint keys: {checkpoint.keys() if isinstance(checkpoint, dict) else 'Not a dict'}")
    
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        print(f"Val Accuracy: {checkpoint.get('val_accuracy', 'N/A')}")
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        print("✅ Model loaded from model_state_dict")
    elif isinstance(checkpoint, dict):
        model.load_state_dict(checkpoint, strict=False)
        print("✅ Model loaded from dict")
    else:
        print("❌ Checkpoint format not recognized")
    
    # Test prediction with dummy data
    print("\n--- Testing Prediction ---")
    lfcc_feat = np.random.randn(10, 20).astype(np.float32)
    pfe_feat = np.random.randn(10, 5).astype(np.float32)
    ppg_feat = np.random.randn(10, 32).astype(np.float32)
    
    with torch.no_grad():
        pred_result = model.predict(lfcc_feat, pfe_feat, ppg_feat, return_probs=True, device=device)
    
    print(f"Prediction result: {pred_result}")
    print(f"Prediction: {pred_result.get('prediction')} (0=Real, 1=Fake)")
    print(f"Real Prob: {pred_result.get('real_prob'):.2f}%")
    print(f"Fake Prob: {pred_result.get('fake_prob'):.2f}%")
    print(f"Confidence: {pred_result.get('confidence'):.2f}%")
