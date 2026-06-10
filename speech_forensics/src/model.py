import torch
import torch.nn as nn
import numpy as np


class MultiStreamForensics(nn.Module):   # ✅ CORRECT BASE CLASS
    def __init__(self, config):
        super().__init__()

        # -----------------------------
        # SAFE CONFIG VALUES
        # -----------------------------
        hidden_size = int(config["model"]["hidden_size"])
        lstm_layers = int(config["model"]["lstm_layers"])
        dropout = float(config["model"]["dropout"])
        self.use_ppg = config.get("features", {}).get("use_ppg", True)

        # -----------------------------
        # FIXED FEATURE DIMENSIONS
        # -----------------------------
        LFCC_DIM = 20
        PFE_DIM = 5
        PPG_DIM = 32

        # -----------------------------
        # LSTM STREAMS
        # -----------------------------
        self.lfcc_lstm = nn.LSTM(
            input_size=LFCC_DIM,
            hidden_size=hidden_size,
            num_layers=lstm_layers,
            dropout=dropout,
            batch_first=True
        )

        self.pfe_lstm = nn.LSTM(
            input_size=PFE_DIM,
            hidden_size=hidden_size,
            num_layers=lstm_layers,
            dropout=dropout,
            batch_first=True
        )

        if self.use_ppg:
            self.ppg_lstm = nn.LSTM(
                input_size=PPG_DIM,
                hidden_size=hidden_size,
                num_layers=lstm_layers,
                dropout=dropout,
                batch_first=True
            )
        else:
            self.ppg_lstm = None

        # -----------------------------
        # CLASSIFIER (Binary: Real/Fake)
        # -----------------------------
        fusion_dim = hidden_size * (3 if self.use_ppg else 2)

        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )

        # -----------------------------
        # ATTACK TYPE CLASSIFIER
        # -----------------------------
        num_attack_types = int(config["model"]["num_attack_types"])
        self.attack_classifier = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_attack_types)
        )

    def forward(self, lfcc, pfe, ppg):
        lfcc_out, _ = self.lfcc_lstm(lfcc)
        pfe_out, _ = self.pfe_lstm(pfe)
        
        lfcc_last = lfcc_out[:, -1, :]
        pfe_last = pfe_out[:, -1, :]
        
        if self.use_ppg and self.ppg_lstm is not None:
            ppg_out, _ = self.ppg_lstm(ppg)
            ppg_last = ppg_out[:, -1, :]
            fused = torch.cat([lfcc_last, pfe_last, ppg_last], dim=1)
        else:
            fused = torch.cat([lfcc_last, pfe_last], dim=1)
        
        # Binary classification (Real/Fake)
        auth_logits = self.classifier(fused)
        
        # Attack type classification
        attack_logits = self.attack_classifier(fused)
        
        return auth_logits, attack_logits, fused

    def predict(self, lfcc, pfe, ppg, return_probs=False, device=None):
        """Predict Real/Fake with confidence scores and attack type"""
        self.eval()
        
        # Get device from model parameters if not provided
        if device is None:
            device = next(self.parameters()).device

        lfcc = torch.tensor(lfcc, dtype=torch.float32).unsqueeze(0).to(device)
        pfe = torch.tensor(pfe, dtype=torch.float32).unsqueeze(0).to(device)
        ppg = torch.tensor(ppg, dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            auth_logits, attack_logits, fused_features = self.forward(lfcc, pfe, ppg)
            
            # Handle NaN in logits
            if torch.isnan(auth_logits).any():
                # If logits are NaN, return default prediction
                return {
                    'prediction': 0,
                    'confidence': 50.0,
                    'real_prob': 50.0,
                    'fake_prob': 50.0,
                    'attack_type': 0,
                    'attack_confidence': 20.0,
                    'attack_probs': np.array([0.2] * 5, dtype=np.float32),
                    'features': np.zeros(128, dtype=np.float32)
                } if return_probs else 0
            
            # Replace any remaining NaN with small value
            auth_logits = torch.nan_to_num(auth_logits, nan=0.0)
            attack_logits = torch.nan_to_num(attack_logits, nan=0.0)
            
            # Get probabilities
            auth_probs = torch.softmax(auth_logits, dim=1)[0]
            attack_probs = torch.softmax(attack_logits, dim=1)[0]
            
            # Get predictions
            auth_pred = torch.argmax(auth_probs).item()
            attack_pred = torch.argmax(attack_probs).item()
            
            if return_probs:
                return {
                    'prediction': auth_pred,
                    'confidence': float(auth_probs[auth_pred].item() * 100),
                    'real_prob': float(auth_probs[0].item() * 100),
                    'fake_prob': float(auth_probs[1].item() * 100),
                    'attack_type': attack_pred,
                    'attack_confidence': float(attack_probs[attack_pred].item() * 100),
                    'attack_probs': attack_probs.cpu().numpy(),
                    'features': fused_features[0].cpu().numpy()
                }
            else:
                return auth_pred
    
    def predict_temporal(self, lfcc, pfe, ppg, segment_duration=1.0, hop_duration=0.5, device=None):
        """Predict tampered regions with temporal localization"""
        self.eval()
        
        # Get device from model parameters if not provided
        if device is None:
            device = next(self.parameters()).device
        
        # Assuming features are already segmented or we need to segment them
        # For now, return frame-level predictions
        results = []
        
        # Process in sliding windows
        n_frames = len(lfcc)
        hop_frames = int(hop_duration * 100)  # Approximate frames per hop
        seg_frames = int(segment_duration * 100)
        
        for start in range(0, n_frames, hop_frames):
            end = min(start + seg_frames, n_frames)
            
            lfcc_seg = lfcc[start:end]
            pfe_seg = pfe[start:end]
            ppg_seg = ppg[start:end]
            
            if len(lfcc_seg) < seg_frames * 0.5:  # Skip too short segments
                continue
            
            pred_result = self.predict(lfcc_seg, pfe_seg, ppg_seg, return_probs=True, device=device)
            time_start = start * 0.01  # Assuming 10ms per frame
            time_end = end * 0.01
            
            results.append({
                'start_time': time_start,
                'end_time': time_end,
                'prediction': pred_result['prediction'],
                'confidence': pred_result['confidence']
            })
        
        return results


# ----------------------------------------------------
# UNIVERSAL MODEL LOADER (for backward compatibility)
# ----------------------------------------------------
def load_model(cfg=None):
    """
    Generic model loader for dashboard compatibility
    """
    import yaml
    import os
    
    # Try to load config
    if cfg is None:
        config_path = os.path.join(os.path.dirname(__file__), "../../config.yaml")
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                cfg = yaml.safe_load(f)
        else:
            # Default config
            cfg = {
                "model": {
                    "hidden_size": 256,
                    "lstm_layers": 2,
                    "dropout": 0.2,
                    "num_attack_types": 5
                }
            }
    
    model = MultiStreamForensics(cfg)
    return model
