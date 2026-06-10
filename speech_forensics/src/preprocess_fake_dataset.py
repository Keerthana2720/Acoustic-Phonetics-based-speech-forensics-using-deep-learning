"""
Preprocess Fake Audio Dataset
Converts fake WAV files to processed segments
Organizes them in data/processed/fake/ for training
"""
import os
import sys
import shutil
import librosa
import soundfile as sf
from pathlib import Path
import numpy as np

# Get project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Configuration
FAKE_INPUT_DIR = os.path.join(PROJECT_ROOT, "fake")
DATA_PROCESSED_FAKE = os.path.join(PROJECT_ROOT, "speech_forensics", "data", "processed", "fake")
DATA_RAW = os.path.join(PROJECT_ROOT, "speech_forensics", "data", "raw")

SEGMENT_DURATION = 3  # 3-second segments
TARGET_SR = 16000  # 16kHz sampling rate
MIN_SEGMENT_LENGTH = 0.6  # Skip segments shorter than 60% of target length

def ensure_dir(path):
    """Create directory if it doesn't exist"""
    Path(path).mkdir(parents=True, exist_ok=True)

def load_audio(path):
    """Load audio using librosa"""
    y, sr = librosa.load(path, sr=TARGET_SR, mono=True)
    return y, sr

def normalize_audio(y):
    """Normalize audio to [-1, 1] range"""
    max_val = np.max(np.abs(y))
    if max_val > 0:
        return y / max_val
    return y

def remove_silence(y, sr):
    """Remove leading/trailing silence"""
    try:
        y_trimmed, _ = librosa.effects.trim(y, top_db=30)
        return y_trimmed
    except:
        return y  # Return original if trimming fails

def segment_audio(y, sr, segment_duration, output_base_path, file_counter):
    """Split audio into n-second chunks and save them"""
    seg_len = int(segment_duration * sr)
    segments_created = []
    count = 0

    for start in range(0, len(y), seg_len):
        end = min(start + seg_len, len(y))
        chunk = y[start:end]

        # Skip very short clips
        if len(chunk) < seg_len * MIN_SEGMENT_LENGTH:
            continue

        # Pad if necessary to reach exact segment length
        if len(chunk) < seg_len:
            padding = np.zeros(seg_len - len(chunk))
            chunk = np.concatenate([chunk, padding])

        # Create output filename
        out_file = os.path.join(output_base_path, f"fake_{file_counter:06d}_seg{count:03d}.wav")
        sf.write(out_file, chunk, sr)
        segments_created.append(out_file)
        count += 1

    return segments_created

def preprocess_fake_dataset(max_files=None):
    """
    Preprocess fake audio dataset
    
    Args:
        max_files: Maximum files to process (None = all)
    """
    print("=" * 70)
    print("FAKE AUDIO DATASET PREPROCESSING")
    print("=" * 70)
    print(f"\n📁 Input directory: {FAKE_INPUT_DIR}")
    print(f"📁 Output directory: {DATA_PROCESSED_FAKE}")
    print(f"⚙️  Target sample rate: {TARGET_SR} Hz")
    print(f"⚙️  Segment duration: {SEGMENT_DURATION} seconds")
    print()

    # Check if input directory exists
    if not os.path.exists(FAKE_INPUT_DIR):
        print(f"ERROR: Input directory not found: {FAKE_INPUT_DIR}")
        print("Please make sure 'fake' folder exists in the project root.")
        return

    # Create output directories
    ensure_dir(DATA_PROCESSED_FAKE)
    print(f"Created output directory: {DATA_PROCESSED_FAKE}\n")

    # Get all audio files
    audio_files = []
    for file in os.listdir(FAKE_INPUT_DIR):
        if file.lower().endswith((".wav", ".mp3", ".flac", ".m4a")):
            audio_files.append(os.path.join(FAKE_INPUT_DIR, file))

    if not audio_files:
        print(f"ERROR: No audio files found in {FAKE_INPUT_DIR}")
        return

    audio_files.sort()

    # Limit files if specified
    if max_files:
        audio_files = audio_files[:max_files]

    print(f"Found {len(audio_files)} audio file(s)\n")
    print("-" * 70)

    total_files_processed = 0
    total_segments_created = 0
    file_counter = 0
    errors = 0

    for file_idx, in_file in enumerate(audio_files, 1):
        try:
            filename = os.path.basename(in_file)
            print(f"\n[{file_idx}/{len(audio_files)}] Processing: {filename}")

            # Load and preprocess audio
            y, sr = load_audio(in_file)
            
            # Skip very short files
            duration = len(y) / sr
            if duration < 0.5:
                print(f"   SKIP: Audio too short ({duration:.2f}s)")
                continue

            # Normalize and remove silence
            y = normalize_audio(y)
            y = remove_silence(y, sr)

            # Skip if still too short after preprocessing
            if len(y) < sr * 0.5:
                print(f"   SKIP: Audio too short after preprocessing")
                continue

            # Segment and save
            segments = segment_audio(y, sr, SEGMENT_DURATION, DATA_PROCESSED_FAKE, file_counter)
            
            if segments:
                print(f"   ✓ Created {len(segments)} segment(s)")
                total_segments_created += len(segments)
                total_files_processed += 1
                file_counter += 1
            else:
                print(f"   SKIP: Could not create segments")

        except Exception as e:
            errors += 1
            print(f"   ✗ ERROR: {e}")

    print("\n" + "=" * 70)
    print("PREPROCESSING COMPLETE!")
    print("=" * 70)
    print(f"Total files processed: {total_files_processed}")
    print(f"Total segments created: {total_segments_created}")
    print(f"Output location: {DATA_PROCESSED_FAKE}")
    if errors > 0:
        print(f"Errors: {errors} file(s) failed")
    print("=" * 70)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Preprocess Fake Audio Dataset")
    parser.add_argument("--max", type=int, default=None,
                        help="Maximum files to process (default: all)")
    
    args = parser.parse_args()
    
    preprocess_fake_dataset(max_files=args.max)
