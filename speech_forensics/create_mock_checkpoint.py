import os
import sys
import yaml
import torch

BASE = os.path.abspath(os.path.dirname(__file__))
# ensure project src is importable as package
sys.path.insert(0, BASE)

from src import model

cfg_path = os.path.join(BASE, 'config.yaml')
with open(cfg_path, 'r') as f:
    cfg = yaml.safe_load(f)

net = model.MultiStreamForensics(cfg)
net.eval()

os.makedirs(os.path.join(BASE, 'checkpoints'), exist_ok=True)
ckpt_path = os.path.join(BASE, 'checkpoints', 'audio_forensics.pt')

torch.save(net.state_dict(), ckpt_path)
print('Saved mock checkpoint to', ckpt_path)
