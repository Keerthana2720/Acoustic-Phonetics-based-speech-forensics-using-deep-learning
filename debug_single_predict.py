import os
import sys
import yaml
import torch
# Ensure speech_forensics package is importable
proj_root = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(proj_root, 'speech_forensics'))
from src import features
from src.model import load_model

config_path = os.path.join(os.path.dirname(__file__), 'speech_forensics', 'config.yaml')
with open(config_path, 'r') as f:
    cfg = yaml.safe_load(f)

model = load_model(cfg)
model = model.to('cpu')

ckpt_path = os.path.join(os.path.dirname(__file__), 'speech_forensics', 'checkpoints', 'audio_forensics.pt')
ckpt = torch.load(ckpt_path, map_location='cpu')
if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
    missing, unexpected = model.load_state_dict(ckpt['model_state_dict'], strict=False)
    print('Loaded model_state_dict, missing:', missing, 'unexpected:', unexpected)
else:
    missing, unexpected = model.load_state_dict(ckpt, strict=False)
    print('Loaded raw ckpt, missing:', missing, 'unexpected:', unexpected)

# Load audio and features
wav = os.path.join(os.path.dirname(__file__), 'speech_forensics', 'data', 'processed', 'fake', 'fake_000002_seg000.wav')
import librosa
y, sr = librosa.load(wav, sr=cfg.get('sample_rate', 16000))
lfcc_feat = features.lfcc(y, sr)
pfe_feat = features.pitch_formants_energy(y, sr)
ppg_extractor = features.PPGExtractor()
ppg_feat = ppg_extractor.extract(y, sr)

pred_result = model.predict(lfcc_feat, pfe_feat, ppg_feat, return_probs=True, device='cpu')
print('Predict result:', pred_result)

# Also try swapping classes to see
orig = pred_result.copy()
swapped = dict(orig)
swapped['prediction'] = 0 if orig['prediction'] == 1 else 1
swapped['fake_prob'], swapped['real_prob'] = orig['real_prob'], orig['fake_prob']
print('Swapped:', swapped)
