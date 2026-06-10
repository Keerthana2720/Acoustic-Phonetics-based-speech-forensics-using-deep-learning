#!/usr/bin/env python
"""Debug dataset loading to find NaN issues"""
import os
import sys
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from src.dataset import ForensicsDataset

config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
with open(config_path, 'r') as f:
    cfg = yaml.safe_load(f)

speech_forensics_dir = os.path.dirname(os.path.abspath(__file__))
root = os.path.join(speech_forensics_dir, cfg["paths"]["processed"])

print(f"Loading from: {root}")
print(f"Root exists: {os.path.exists(root)}")

try:
    train_ds = ForensicsDataset(root, split="train", config=cfg)
    print(f"Train samples: {len(train_ds)}")
    
    if len(train_ds) > 0:
        # Get first sample
        streams, label, attack = train_ds[0]
        print(f"\nFirst sample:")
        print(f"  Label: {label}")
        print(f"  Attack: {attack}")
        print(f"  LFCC shape: {streams['lfcc'].shape}, dtype: {streams['lfcc'].dtype}")
        print(f"  PFE shape: {streams['pfe'].shape}, dtype: {streams['pfe'].dtype}")
        print(f"  PPG shape: {streams['ppg'].shape}, dtype: {streams['ppg'].dtype}")
        
        # Check for NaN/Inf
        import numpy as np
        for key, feat in streams.items():
            has_nan = np.any(np.isnan(feat))
            has_inf = np.any(np.isinf(feat))
            print(f"  {key}: NaN={has_nan}, Inf={has_inf}")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
