import os
import argparse
import math
import yaml
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
import random
import sys
import io

# Fix Unicode encoding on Windows
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

from .dataset import ForensicsDataset
from .model import MultiStreamForensics


# ======================================================
# Reproducibility
# ======================================================
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # Deterministic CuDNN for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ======================================================
# Collate Function
# ======================================================
def collate_fn(batch):
    lfccs, pfes, ppgs, labels, attacks = [], [], [], [], []

    for streams, label, attack in batch:
        # convert incoming feature arrays/tensors into float tensors
        def to_tensor(x):
            if isinstance(x, np.ndarray):
                return torch.from_numpy(x).float()
            if torch.is_tensor(x):
                return x.float()
            return torch.tensor(x, dtype=torch.float32)

        lfccs.append(to_tensor(streams["lfcc"]))
        pfes.append(to_tensor(streams["pfe"]))
        ppgs.append(to_tensor(streams["ppg"]))
        labels.append(label)
        attacks.append(attack)

    def pad(seq):
        max_len = max(x.shape[0] for x in seq)
        return torch.stack([
            F.pad(x, (0, 0, 0, max_len - x.shape[0])) for x in seq
        ])

    return (
        pad(lfccs),
        pad(pfes),
        pad(ppgs),
        torch.tensor(labels),
        torch.tensor(attacks)
    )


def evaluate(model, loader, device):
    """Run evaluation and return validation accuracy (percentage)."""
    model.eval()
    preds, gts = [], []
    with torch.no_grad():
        for lfcc, pfe, ppg, labels, _ in loader:
            lfcc, pfe, ppg = lfcc.to(device), pfe.to(device), ppg.to(device)
            labels = labels.to(device)

            auth_logits, _, _ = model(lfcc, pfe, ppg)
            pred = torch.argmax(auth_logits, dim=1)

            preds.extend(pred.cpu().numpy())
            gts.extend(labels.cpu().numpy())

    val_acc = accuracy_score(gts, preds) * 100
    return val_acc


# ======================================================
# Training
# ======================================================
def main():
    # ---------------------------
    # Load config
    # ---------------------------
    # Get the speech_forensics directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "..", "config.yaml")
    
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg.get("seed", 42))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n[INFO] Using device: {device}")
    sys.stdout.flush()

    # ---------------------------
    # Dataset
    # ---------------------------
    # Use absolute path for processed data
    speech_forensics_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    root = os.path.join(speech_forensics_dir, cfg["paths"]["processed"])

    train_ds = ForensicsDataset(root, split="train", config=cfg)
    val_ds = ForensicsDataset(root, split="val", config=cfg)

    # Dataset sizes and class distribution
    print(f"[DATA] Train samples: {len(train_ds)}")
    print(f"[DATA] Val samples: {len(val_ds)}")
    sys.stdout.flush()

    # Show class balance
    train_labels = [train_ds.labels[i] for i in train_ds.indices]
    val_labels = [val_ds.labels[i] for i in val_ds.indices]
    def _counts(lst):
        unique, counts = np.unique(lst, return_counts=True)
        return dict(zip(unique.tolist(), counts.tolist()))
    print(f"[DISTRIBUTION] Train label counts: {_counts(train_labels)} (0=real,1=fake)")
    print(f"[DISTRIBUTION] Val label counts: {_counts(val_labels)} (0=real,1=fake)")
    sys.stdout.flush()

    # DataLoader options (use num_workers and pin_memory where appropriate)
    num_workers = int(cfg.get('training', {}).get('num_workers', 4))
    dataloader_kwargs = dict(
        batch_size=cfg["training"]["batch_size"],
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=(device == 'cuda'),
        persistent_workers=(num_workers > 0)
    )

    # Optionally use WeightedRandomSampler to oversample minority classes
    if cfg.get('training', {}).get('use_weight_sampler', False):
        from torch.utils.data import WeightedRandomSampler

        train_labels = [train_ds.labels[i] for i in train_ds.indices]
        class_counts = np.bincount(train_labels, minlength=2).astype(np.float32)
        class_weights = 1.0 / (class_counts + 1e-6)
        samples_weight = np.array([class_weights[l] for l in train_labels], dtype=np.float64)
        samples_weight = torch.DoubleTensor(samples_weight)
        sampler = WeightedRandomSampler(samples_weight, num_samples=len(samples_weight), replacement=True)

        train_loader = DataLoader(
            train_ds,
            sampler=sampler,
            **dataloader_kwargs
        )
    else:
        train_loader = DataLoader(
            train_ds,
            shuffle=True,
            **dataloader_kwargs
        )

    # Validation loader
    val_loader = DataLoader(
        val_ds,
        shuffle=False,
        **dataloader_kwargs
    )

    # Print augmentation config if enabled
    if cfg.get('training', {}).get('augment', False):
        aug_cfg = cfg.get('training', {}).get('augment_config', {})
        print(f"[AUGMENT] Augmentation enabled for training. Config: {aug_cfg}")
        sys.stdout.flush()

    # ---------------------------
    # Model (IMPORTANT FIX)
    # ---------------------------
    # CLI options: allow loading a checkpoint, resuming optimizer/scheduler,
    # or running evaluation only.
    parser = argparse.ArgumentParser(description="Train/evaluate model")
    parser.add_argument('--checkpoint', '-c', default=None, help='Path to pretrained checkpoint to load')
    parser.add_argument('--resume', action='store_true', help='Resume optimizer/scheduler state from checkpoint if available')
    parser.add_argument('--eval-only', action='store_true', help='Only run evaluation and exit')
    parser.add_argument('--fine-tune', action='store_true', help='Run a short fine-tune with augmentations and oversampling (use with --checkpoint to initialize)')
    parser.add_argument('--fine-epochs', type=int, default=None, help='Number of epochs for fine-tune (overrides config training.epochs when --fine-tune is used)')
    args = parser.parse_args()

    # If user requested a short fine-tune, enable augmentation and oversampling
    if args.fine_tune:
        print("[FINE-TUNE] Fine-tune mode enabled: applying augmentation & oversampling")
        sys.stdout.flush()
        cfg.setdefault('training', {})
        cfg['training']['augment'] = True
        cfg['training']['use_weight_sampler'] = True
        if args.fine_epochs is not None:
            cfg['training']['epochs'] = int(args.fine_epochs)

    model = MultiStreamForensics(cfg).to(device)
    # Optionally compile model with torch.compile() (requires PyTorch >= 2.0)
    if cfg.get('training', {}).get('use_torch_compile', False):
        try:
            model = torch.compile(model)
            print("[INFO] Model compiled with torch.compile()")
        except Exception as e:
            print(f"[WARNING] torch.compile() failed: {e}")
        sys.stdout.flush()

    # ---------------------------
    # Class weighting to handle imbalance (optional)
    # ---------------------------
    use_class_weight = cfg.get('training', {}).get('use_class_weight', True)
    if use_class_weight:
        # Compute class frequencies from training split
        train_labels = [train_ds.labels[i] for i in train_ds.indices]
        counts = np.bincount(train_labels, minlength=2).astype(np.float32)
        # Inverse frequency weighting
        class_weights = (1.0 / (counts + 1e-6))
        class_weights = class_weights / class_weights.sum() * len(class_weights)
        class_weights = torch.tensor(class_weights).float().to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        print(f"[WEIGHTS] Using class weights: {class_weights.cpu().numpy()}")
        sys.stdout.flush()
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = AdamW(model.parameters(), lr=cfg["training"]["lr"])
    scheduler = ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

    # If a checkpoint path was provided, try to load weights and optionally resume
    if args.checkpoint:
        if os.path.exists(args.checkpoint):
            checkpoint = torch.load(args.checkpoint, map_location=device)
            if 'model_state_dict' in checkpoint:
                missing, unexpected = model.load_state_dict(checkpoint['model_state_dict'], strict=False)
                print(f"[CHECKPOINT] Loaded model weights from {args.checkpoint}")
                if missing:
                    print(f"[WARNING] Missing keys when loading checkpoint: {missing}")
                if unexpected:
                    print(f"[WARNING] Unexpected keys when loading checkpoint: {unexpected}")
                if 'val_accuracy' in checkpoint:
                    print(f"[INFO] Checkpoint reported val_accuracy: {checkpoint['val_accuracy']:.2f}%")
                sys.stdout.flush()

                # If eval-only, run an evaluation and compare
                if args.eval_only:
                    val_acc = evaluate(model, val_loader, device)
                    print(f"[EVAL] Current model val_acc: {val_acc:.2f}%")
                    loaded_val = checkpoint.get('val_accuracy', None)
                    if loaded_val is not None and not math.isclose(val_acc, loaded_val, rel_tol=1e-3):
                        print("[WARNING] Current val_acc differs from checkpoint val_accuracy")
                    print("Exiting due to --eval-only flag")
                    return
            else:
                print("[WARNING] Checkpoint does not contain 'model_state_dict'")
            sys.stdout.flush()

            if args.resume:
                if 'optimizer_state_dict' in checkpoint:
                    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    print("[RESTORE] Restored optimizer state from checkpoint")
                if 'scheduler_state_dict' in checkpoint:
                    try:
                        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                        print("[RESTORE] Restored scheduler state from checkpoint")
                    except Exception as e:
                        print(f"[WARNING] Could not restore scheduler state: {e}")
            sys.stdout.flush()
        else:
            print(f"[WARNING] Checkpoint path {args.checkpoint} not found")
            sys.stdout.flush()

    # ---------------------------
    # Training Loop
    # ---------------------------
    best_val_acc = 0.0
    os.makedirs("checkpoints", exist_ok=True)

    for epoch in range(cfg["training"]["epochs"]):
        model.train()
        total_loss = 0
        batch_count = 0

        for lfcc, pfe, ppg, labels, _ in train_loader:
            lfcc, pfe, ppg = lfcc.to(device), pfe.to(device), ppg.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            auth_logits, _, _ = model(lfcc, pfe, ppg)
            
            # Ensure logits are valid (no NaN/Inf)
            if torch.isnan(auth_logits).any() or torch.isinf(auth_logits).any():
                print(f"[WARNING] Invalid logits detected, skipping batch")
                continue
            
            loss = criterion(auth_logits, labels)
            
            # Skip if loss is NaN
            if torch.isnan(loss):
                print(f"[WARNING] NaN loss detected, skipping batch")
                continue
            sys.stdout.flush()
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            batch_count += 1

        # ---------------------------
        # Validation
        # ---------------------------
        model.eval()
        preds, gts = [], []

        with torch.no_grad():
            for lfcc, pfe, ppg, labels, _ in val_loader:
                lfcc, pfe, ppg = lfcc.to(device), pfe.to(device), ppg.to(device)
                labels = labels.to(device)

                auth_logits, _, _ = model(lfcc, pfe, ppg)
                pred = torch.argmax(auth_logits, dim=1)

                preds.extend(pred.cpu().numpy())
                gts.extend(labels.cpu().numpy())

        val_acc = accuracy_score(gts, preds) * 100
        scheduler.step(1 - val_acc)

        avg_loss = total_loss / batch_count if batch_count > 0 else 0.0
        print(
            f"Epoch [{epoch+1}/{cfg['training']['epochs']}], "
            f"Loss: {avg_loss:.4f}, Val Acc: {val_acc:.2f}%"
        )
        sys.stdout.flush()

        # ---------------------------
        # SAVE MODEL (CRITICAL FIX)
        # ---------------------------
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            # Save with both state_dict and config for better compatibility
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'config': cfg,
                'val_accuracy': val_acc,
                'epoch': epoch
            }, "checkpoints/audio_forensics.pt")
            print("[SUCCESS] Saved best model")
            sys.stdout.flush()

    print("\n[COMPLETE] Training complete")
    print(f"[RESULT] Best Validation Accuracy: {best_val_acc:.2f}%")


if __name__ == "__main__":
    main()
