import os
import librosa
import torch
import numpy as np
from torch.utils.data import Dataset
from . import features


# ============================================================
# 1. Indian Speech Dataset (Language-wise – optional module)
# ============================================================
class IndianSpeechDataset(Dataset):
    """
    Loads Indian language speech data from:
    data/processed_real/<language>/*.wav
    """

    def __init__(self, root="data/processed_real", sr=16000):
        self.sr = sr
        self.files = []
        self.languages = []

        if not os.path.exists(root):
            raise FileNotFoundError(f"Dataset path not found: {root}")

        for lang in os.listdir(root):
            lang_path = os.path.join(root, lang)
            if not os.path.isdir(lang_path):
                continue

            for f in os.listdir(lang_path):
                if f.endswith(".wav"):
                    self.files.append(os.path.join(lang_path, f))
                    self.languages.append(lang)

        if len(self.files) == 0:
            raise RuntimeError("No WAV files found in processed dataset")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        audio, _ = librosa.load(self.files[idx], sr=self.sr, mono=True)
        return torch.tensor(audio, dtype=torch.float32), self.languages[idx]


# ============================================================
# 2. MAIN FORENSICS DATASET (REAL vs FAKE)
# ============================================================
class ForensicsDataset(Dataset):
    """
    Dataset for forensic speech detection
    Folder structure required:

    data/processed_real/
        ├── real/
        └── fake/
    """

    def __init__(
        self,
        root="data/processed_real",
        sr=16000,
        split="train",
        train_ratio=0.8,
        seed=42,
        config=None
    ):
        # store root for precomputed feature lookup
        self.root = root
        self.sr = sr
        self.files = []
        self.labels = []        # 0 = real, 1 = fake
        self.attack_types = []  # -1 for real, 0–4 for fake
        self.split = split
        self.config = config or {}
        self.use_ppg = self.config.get("features", {}).get("use_ppg", True)

        real_dir = os.path.join(root, "real")
        fake_dir = os.path.join(root, "fake")

        if not os.path.exists(real_dir):
            raise FileNotFoundError("Missing folder: data/processed_real/real")

        if not os.path.exists(fake_dir):
            raise FileNotFoundError("Missing folder: data/processed_real/fake")

        # ---------------------------
        # Load REAL samples
        # ---------------------------
        for f in os.listdir(real_dir):
            if f.endswith((".wav", ".mp3", ".flac")):
                self.files.append(os.path.join(real_dir, f))
                self.labels.append(0)
                self.attack_types.append(-1)

        # ---------------------------
        # Load FAKE samples
        # ---------------------------
        for f in os.listdir(fake_dir):
            if f.endswith((".wav", ".mp3", ".flac")):
                path = os.path.join(fake_dir, f)
                self.files.append(path)
                self.labels.append(1)

                name = f.lower()
                if "tts" in name:
                    atk = 0
                elif "vc" in name:
                    atk = 1
                elif "replay" in name:
                    atk = 2
                elif "gan" in name or "deepfake" in name:
                    atk = 3
                elif "splice" in name or "edit" in name:
                    atk = 4
                else:
                    atk = 0

                self.attack_types.append(atk)

        if len(self.files) == 0:
            raise RuntimeError("No audio files found in processed_real")

        # ---------------------------
        # Train / Validation split
        # ---------------------------
        np.random.seed(seed)
        indices = np.random.permutation(len(self.files))
        split_idx = int(train_ratio * len(indices))

        self.indices = (
            indices[:split_idx]
            if split == "train"
            else indices[split_idx:]
        )

        # Feature extractor
        if self.use_ppg:
            self.ppg_extractor = features.PPGExtractor()
        else:
            self.ppg_extractor = None

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i = self.indices[idx]
        filepath = self.files[i]
        label = self.labels[i]
        attack = self.attack_types[i]

        # Load audio
        audio, sr = librosa.load(filepath, sr=self.sr, mono=True)

        # Optional augmentation for training split (applied on waveform before feature extraction)
        if self.split == 'train' and self.config.get('training', {}).get('augment', False):
            aug_cfg = self.config.get('training', {}).get('augment_config', {})
            try:
                audio = features.apply_augmentations(audio, sr, aug_cfg)
            except Exception:
                # If augmentation fails, fallback to original audio
                pass

        # Try to load precomputed features first
        category = os.path.basename(os.path.dirname(filepath))
        base_name = os.path.splitext(os.path.basename(filepath))[0]
        feat_path = os.path.join(self.root, "features", category, base_name + ".npz")

        if os.path.exists(feat_path):
            try:
                npz = np.load(feat_path)
                lfcc = npz["lfcc"]
                pfe = npz["pfe"]
                if "ppg" in npz:
                    ppg = npz["ppg"]
                else:
                    ppg = np.zeros((lfcc.shape[0], 32))
            except Exception:
                # fallback to on-the-fly extraction if npz load fails
                lfcc = features.lfcc(audio, sr)
                pfe = features.pitch_formants_energy(audio, sr)
                if self.use_ppg and self.ppg_extractor is not None:
                    ppg = self.ppg_extractor.extract(audio, sr)
                else:
                    ppg = np.zeros((lfcc.shape[0], 32))
        else:
            # Feature extraction (on-the-fly)
            lfcc = features.lfcc(audio, sr)
            pfe = features.pitch_formants_energy(audio, sr)
            if self.use_ppg and self.ppg_extractor is not None:
                ppg = self.ppg_extractor.extract(audio, sr)
                # Normalize PPG dimension → 32
                if ppg.shape[-1] > 32:
                    ppg = ppg[:, :32]
                elif ppg.shape[-1] < 32:
                    pad = np.zeros((ppg.shape[0], 32 - ppg.shape[-1]))
                    ppg = np.concatenate([ppg, pad], axis=-1)
            else:
                ppg = np.zeros((lfcc.shape[0], 32))

        features_dict = {
            "lfcc": np.array(lfcc, dtype=np.float32),
            "pfe": np.array(pfe, dtype=np.float32),
            "ppg": np.array(ppg, dtype=np.float32),
        }

        return features_dict, torch.tensor(label), torch.tensor(attack)
