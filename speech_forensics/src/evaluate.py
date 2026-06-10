"""
Performance Metrics Evaluation
Calculates accuracy, precision, recall, F1-score, EER, and confusion matrix
"""
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc
)
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Dict, List


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                     y_probs: np.ndarray = None) -> Dict[str, float]:
    """
    Calculate all performance metrics
    
    Args:
        y_true: True labels (0 or 1)
        y_pred: Predicted labels (0 or 1)
        y_probs: Predicted probabilities for class 1 (fake)
    
    Returns:
        Dictionary with all metrics
    """
    metrics = {}
    
    # Basic classification metrics
    metrics['accuracy'] = accuracy_score(y_true, y_pred) * 100
    metrics['precision'] = precision_score(y_true, y_pred, zero_division=0) * 100
    metrics['recall'] = recall_score(y_true, y_pred, zero_division=0) * 100
    metrics['f1_score'] = f1_score(y_true, y_pred, zero_division=0) * 100
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    metrics['confusion_matrix'] = cm
    
    # Calculate EER if probabilities are provided
    if y_probs is not None:
        metrics['eer'] = calculate_eer(y_true, y_probs)
        metrics['auc'] = calculate_auc(y_true, y_probs)
    else:
        metrics['eer'] = None
        metrics['auc'] = None
    
    return metrics


def calculate_eer(y_true: np.ndarray, y_probs: np.ndarray) -> float:
    """
    Calculate Equal Error Rate (EER)
    
    Args:
        y_true: True labels
        y_probs: Predicted probabilities for class 1
    
    Returns:
        EER percentage
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_probs)
    fnr = 1 - tpr
    
    # Find threshold where FPR = FNR
    eer_threshold = thresholds[np.nanargmin(np.absolute((fnr - fpr)))]
    eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
    
    return eer * 100


def calculate_auc(y_true: np.ndarray, y_probs: np.ndarray) -> float:
    """
    Calculate Area Under ROC Curve (AUC)
    
    Args:
        y_true: True labels
        y_probs: Predicted probabilities for class 1
    
    Returns:
        AUC score
    """
    from sklearn.metrics import roc_auc_score
    try:
        auc_score = roc_auc_score(y_true, y_probs)
        return auc_score * 100
    except ValueError:
        return 0.0


def plot_confusion_matrix(cm: np.ndarray, save_path: str = None) -> plt.Figure:
    """
    Plot confusion matrix
    
    Args:
        cm: Confusion matrix (2x2)
        save_path: Optional path to save figure
    
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Real', 'Fake'],
                yticklabels=['Real', 'Fake'])
    
    ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
    ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_roc_curve(y_true: np.ndarray, y_probs: np.ndarray, 
                   save_path: str = None) -> plt.Figure:
    """
    Plot ROC curve
    
    Args:
        y_true: True labels
        y_probs: Predicted probabilities for class 1
        save_path: Optional path to save figure
    
    Returns:
        Matplotlib figure
    """
    from sklearn.metrics import roc_curve, auc
    
    fpr, tpr, thresholds = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.plot(fpr, tpr, color='darkorange', lw=2, 
            label=f'ROC curve (AUC = {roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    ax.set_title('Receiver Operating Characteristic (ROC) Curve', 
                 fontsize=14, fontweight='bold')
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def print_metrics_table(metrics: Dict[str, float]):
    """
    Print metrics in a formatted table
    
    Args:
        metrics: Dictionary of metrics
    """
    print("\n" + "=" * 50)
    print("PERFORMANCE METRICS")
    print("=" * 50)
    print(f"{'Metric':<20} {'Value':<15}")
    print("-" * 50)
    print(f"{'Accuracy':<20} {metrics['accuracy']:.2f}%")
    print(f"{'Precision':<20} {metrics['precision']:.2f}%")
    print(f"{'Recall':<20} {metrics['recall']:.2f}%")
    print(f"{'F1 Score':<20} {metrics['f1_score']:.2f}%")
    
    if metrics['eer'] is not None:
        print(f"{'EER':<20} {metrics['eer']:.2f}%")
    if metrics['auc'] is not None:
        print(f"{'AUC':<20} {metrics['auc']:.2f}%")
    
    print("=" * 50)
    print("\nConfusion Matrix:")
    print(metrics['confusion_matrix'])
    print()


def evaluate_model(model, dataloader, device='cpu'):
    """
    Evaluate model on a dataloader
    
    Args:
        model: Trained model
        dataloader: DataLoader with test data
        device: Device to run evaluation on
    
    Returns:
        Tuple (metrics_dict, preds, probs, labels)
    """
    model.eval()
    
    def pad_and_stack(seq):
        """Pad a list of tensors (frames x feat) along time dimension and stack into (B, T, F)"""
        max_len = max(x.shape[0] for x in seq)
        return torch.stack([
            torch.from_numpy(x) if isinstance(x, np.ndarray) else x for x in [
                np.pad(x, ((0, max_len - x.shape[0]), (0, 0)), mode='constant') if isinstance(x, np.ndarray) else F.pad(x, (0, 0, 0, max_len - x.shape[0]))
                for x in seq
            ]
        ]).float()

    all_preds = []
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            # Support two batch styles:
            # 1) Collated batch: (lfcc, pfe, ppg, labels, attacks)
            # 2) Per-sample batch: list of tuples (streams_dict, label, attack)
            if isinstance(batch, (list, tuple)) and len(batch) == 5 and torch.is_tensor(batch[0]):
                lfcc, pfe, ppg, labels, _ = batch
            else:
                # Handle list of samples
                lfcc_list, pfe_list, ppg_list, label_list = [], [], [], []
                for item in batch:
                    if isinstance(item, (list, tuple)) and len(item) >= 2:
                        streams, lab = item[0], item[1]
                        lfcc_list.append(streams['lfcc'])
                        pfe_list.append(streams['pfe'])
                        ppg_list.append(streams['ppg'])
                        label_list.append(lab.numpy() if isinstance(lab, torch.Tensor) else lab)
                    else:
                        raise ValueError("Unexpected batch item format in dataloader")

                lfcc = pad_and_stack(lfcc_list).to(device)
                pfe = pad_and_stack(pfe_list).to(device)
                ppg = pad_and_stack(ppg_list).to(device)
                labels = torch.tensor(label_list)

            lfcc = lfcc.to(device)
            pfe = pfe.to(device)
            ppg = ppg.to(device)
            labels = labels.to(device)

            auth_logits, _, _ = model(lfcc, pfe, ppg)
            probs = torch.softmax(auth_logits, dim=1)

            preds = torch.argmax(probs, dim=1).cpu().numpy()
            fake_probs = probs[:, 1].cpu().numpy()
            labels_np = labels.cpu().numpy() if isinstance(labels, torch.Tensor) else np.array(labels)

            all_preds.extend(preds)
            all_probs.extend(fake_probs)
            all_labels.extend(labels_np)
    
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    
    metrics = calculate_metrics(all_labels, all_preds, all_probs)
    
    return metrics, all_preds, all_probs, all_labels


if __name__ == "__main__":
    # Quick CLI to evaluate validation dataset with a checkpoint
    import os, yaml
    from torch.utils.data import DataLoader
    from .dataset import ForensicsDataset
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(script_dir, "..", "config.yaml")
        with open(config_path, 'r') as f:
            cfg = yaml.safe_load(f)
    except Exception:
        cfg = None

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    processed_rel = (cfg or {}).get('paths', {}).get('processed', 'data/processed')
    processed_root = os.path.join(base_dir, processed_rel)

    val_ds = ForensicsDataset(root=processed_root, split='val', config=cfg)
    from .train import collate_fn
    batch_size = (cfg or {}).get('training', {}).get('batch_size', 32)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    # Try to load checkpoint from project root
    ckpt_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'checkpoints', 'audio_forensics.pt')
    model = None
    if os.path.exists(ckpt_path):
        from .model import load_model
        model = load_model(cfg)
        model = model.to(device)
        ckpt = torch.load(ckpt_path, map_location=device)
        if isinstance(ckpt, dict):
            if 'model_state_dict' in ckpt:
                model.load_state_dict(ckpt['model_state_dict'], strict=False)
            else:
                model.load_state_dict(ckpt, strict=False)
        print(f"Loaded checkpoint from {ckpt_path}. Checkpoint val_accuracy: {ckpt.get('val_accuracy', 'N/A') if isinstance(ckpt, dict) else 'N/A'}")

    if model is None:
        print("No checkpoint found, exiting")
    else:
        metrics, preds, probs, labels = evaluate_model(model, val_loader, device=device)
        print_metrics_table(metrics)

        # Show some misclassified examples for debugging
        false_negatives = []  # label=1 but predicted 0
        false_positives = []  # label=0 but predicted 1
        for i in range(len(val_ds)):
            streams, lab, atk = val_ds[i]
            fname = val_ds.files[val_ds.indices[i]]
            pred_result = model.predict(streams['lfcc'], streams['pfe'], streams['ppg'], return_probs=True, device=device)
            pred = pred_result['prediction']
            if lab.item() == 1 and pred == 0:
                false_negatives.append((fname, pred_result))
            if lab.item() == 0 and pred == 1:
                false_positives.append((fname, pred_result))

        print(f"Found {len(false_negatives)} false negatives and {len(false_positives)} false positives in val set")
        if len(false_negatives) > 0:
            print("Sample false negatives (label=1 but predicted=0):")
            for f, r in false_negatives[:10]:
                print(f" - {os.path.basename(f)}  fake_prob={r['fake_prob']:.2f}% real_prob={r['real_prob']:.2f}%")
        if len(false_positives) > 0:
            print("Sample false positives (label=0 but predicted=1):")
            for f, r in false_positives[:10]:
                print(f" - {os.path.basename(f)}  fake_prob={r['fake_prob']:.2f}% real_prob={r['real_prob']:.2f}%")


