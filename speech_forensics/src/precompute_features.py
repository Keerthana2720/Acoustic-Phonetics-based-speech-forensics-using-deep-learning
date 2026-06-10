"""
Precompute LFCC and PFE features for processed dataset.
Saves per-audio .npz files under data/processed/features/<real|fake>/*.npz
"""
import os
import sys
import numpy as np
from pathlib import Path

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src import features  # local features module

DATA_PROCESSED = os.path.join(PROJECT_ROOT, "speech_forensics", "data", "processed")
FEATURES_DIR = os.path.join(DATA_PROCESSED, "features")

ensure = lambda p: Path(p).mkdir(parents=True, exist_ok=True)


def process_folder(category):
    folder = os.path.join(DATA_PROCESSED, category)
    out_dir = os.path.join(FEATURES_DIR, category)
    ensure(out_dir)

    files = [f for f in os.listdir(folder) if f.lower().endswith('.wav')]
    files.sort()

    for i, fname in enumerate(files, 1):
        path = os.path.join(folder, fname)
        base = os.path.splitext(fname)[0]
        out_path = os.path.join(out_dir, base + '.npz')

        if os.path.exists(out_path):
            continue

        try:
            # compute features
            audio, sr = features.load_audio(path)
            lfcc = features.lfcc(audio, sr)
            pfe = features.pitch_formants_energy(audio, sr)

            # ppg skipped by default; include if extractor exists
            try:
                ppg = features.PPGExtractor().extract(audio, sr)
                if ppg is not None:
                    np.savez(out_path, lfcc=lfcc.astype(np.float32), pfe=pfe.astype(np.float32), ppg=ppg.astype(np.float32))
                else:
                    np.savez(out_path, lfcc=lfcc.astype(np.float32), pfe=pfe.astype(np.float32))
            except Exception:
                np.savez(out_path, lfcc=lfcc.astype(np.float32), pfe=pfe.astype(np.float32))

            if i % 100 == 0:
                print(f"Processed {i}/{len(files)} in {category}")
        except Exception as e:
            print(f"Failed: {path} -> {e}")


if __name__ == '__main__':
    for cat in ['real', 'fake']:
        if os.path.exists(os.path.join(DATA_PROCESSED, cat)):
            print(f"Processing category: {cat}")
            process_folder(cat)
        else:
            print(f"Skipping, missing: {cat}")

    print("Done precomputing features")
