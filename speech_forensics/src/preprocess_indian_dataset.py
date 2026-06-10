"""
Preprocess Indian Language Audio Dataset
Converts MP3 files from Indian_Languages_Audio_Dataset to processed WAV files
Organizes them in data/processed/real/ for training
"""
import os
import sys
import shutil
import librosa
import soundfile as sf
from pydub import AudioSegment
import numpy as np
from pathlib import Path

# Get project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Configuration
INPUT_ROOT = os.path.join(PROJECT_ROOT, "Indian_Languages_Audio_Dataset")
DATA_RAW = os.path.join(PROJECT_ROOT, "speech_forensics", "data", "raw")
DATA_PROCESSED_REAL = os.path.join(PROJECT_ROOT, "speech_forensics", "data", "processed", "real")

SEGMENT_DURATION = 3  # 3-second segments
TARGET_SR = 16000  # 16kHz sampling rate
MIN_SEGMENT_LENGTH = 0.6  # Skip segments shorter than 60% of target length

def ensure_dir(path):
    """Create directory if it doesn't exist"""
    Path(path).mkdir(parents=True, exist_ok=True)

def convert_to_wav(input_file, out_file):
    """Converts MP3/M4A/others → WAV mono 16k"""
    try:
        audio = AudioSegment.from_file(input_file)
        audio = audio.set_channels(1)  # Mono
        audio = audio.set_frame_rate(TARGET_SR)  # 16kHz
        audio.export(out_file, format="wav")
        return True
    except Exception as e:
        print(f"  ⚠️  Error converting {input_file}: {e}")
        return False

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
        out_file = os.path.join(output_base_path, f"real_{file_counter:06d}_seg{count:03d}.wav")
        sf.write(out_file, chunk, sr)
        segments_created.append(out_file)
        count += 1

    return segments_created

def preprocess_indian_dataset(max_files_per_language=None, max_total_files=None):
    """
    Preprocess Indian language audio datasets
    
    Args:
        max_files_per_language: Maximum files to process per language (None = all)
        max_total_files: Maximum total files to process across all languages (None = all)
    """
    print("=" * 70)
    print("INDIAN LANGUAGE DATASET PREPROCESSING")
    print("=" * 70)
    print(f"\n📁 Input directory: {INPUT_ROOT}")
    print(f"📁 Output directory: {DATA_PROCESSED_REAL}")
    print(f"⚙️  Target sample rate: {TARGET_SR} Hz")
    print(f"⚙️  Segment duration: {SEGMENT_DURATION} seconds")
    print()

    # Check if input directory exists
    if not os.path.exists(INPUT_ROOT):
        print(f"ERROR: Input directory not found: {INPUT_ROOT}")
        print("Please make sure Indian_Languages_Audio_Dataset folder exists in the project root.")
        return

    # Create output directories
    ensure_dir(DATA_PROCESSED_REAL)
    print(f"Created output directory: {DATA_PROCESSED_REAL}\n")

    # Get all language directories
    languages = [d for d in os.listdir(INPUT_ROOT) 
                 if os.path.isdir(os.path.join(INPUT_ROOT, d)) and not d.startswith('.')]
    languages.sort()

    print(f"Found {len(languages)} language directories: {', '.join(languages)}\n")
    print("-" * 70)

    total_files_processed = 0
    total_segments_created = 0
    file_counter = 0  # Global counter for unique file names

    for lang_idx, lang in enumerate(languages, 1):
        lang_path = os.path.join(INPUT_ROOT, lang)
        
        # Get all audio files in this language directory
        audio_files = []
        for root, dirs, files in os.walk(lang_path):
            for file in files:
                if file.lower().endswith((".mp3", ".wav", ".flac", ".m4a")):
                    audio_files.append(os.path.join(root, file))

        if not audio_files:
            print(f"\n[{lang_idx}/{len(languages)}] SKIP {lang}: No audio files found")
            continue

        # Limit files per language if specified
        if max_files_per_language:
            audio_files = audio_files[:max_files_per_language]

        print(f"\n[{lang_idx}/{len(languages)}] Processing: {lang}")
        print(f"   Found {len(audio_files)} audio file(s)")

        lang_segments = 0
        lang_processed = 0
        lang_errors = 0

        for file_idx, in_file in enumerate(audio_files, 1):
            # Check total file limit
            if max_total_files and total_files_processed >= max_total_files:
                print(f"\n   Reached maximum total files limit ({max_total_files})")
                break

            try:
                # Determine output filename
                file_ext = os.path.splitext(in_file)[1].lower()
                
                # Convert to WAV if needed (using temporary file)
                temp_wav = None
                if file_ext != '.wav':
                    # Use librosa directly for MP3/FLAC (doesn't require ffmpeg)
                    try:
                        # Use a temp directory for conversion
                        temp_dir = os.path.join(PROJECT_ROOT, "speech_forensics", "data", "temp")
                        ensure_dir(temp_dir)
                        temp_wav = os.path.join(temp_dir, f"temp_{file_counter}_{lang}.wav")
                        
                        # Use librosa to load and save (works without ffmpeg)
                        y_temp, sr_temp = librosa.load(in_file, sr=TARGET_SR, mono=True)
                        sf.write(temp_wav, y_temp, TARGET_SR)
                        wav_file = temp_wav
                    except Exception as e:
                        # Fallback to pydub if librosa fails
                        if not convert_to_wav(in_file, temp_wav):
                            lang_errors += 1
                            if lang_errors <= 3:
                                print(f"   WARNING: Error converting {os.path.basename(in_file)}: {e}")
                            continue
                        wav_file = temp_wav
                else:
                    wav_file = in_file

                # Load and preprocess audio
                y, sr = load_audio(wav_file)
                
                # Skip very short files
                duration = len(y) / sr
                if duration < 0.5:
                    if temp_wav and os.path.exists(temp_wav):
                        os.remove(temp_wav)
                    continue

                # Normalize and remove silence
                y = normalize_audio(y)
                y = remove_silence(y, sr)

                # Skip if still too short after preprocessing
                if len(y) < sr * 0.5:
                    if temp_wav and os.path.exists(temp_wav):
                        os.remove(temp_wav)
                    continue

                # Segment and save
                segments = segment_audio(y, sr, SEGMENT_DURATION, DATA_PROCESSED_REAL, file_counter)
                
                if segments:
                    lang_segments += len(segments)
                    lang_processed += 1
                    total_segments_created += len(segments)
                    total_files_processed += 1
                    file_counter += 1

                    if file_idx % 100 == 0:
                        print(f"   Progress: {file_idx}/{len(audio_files)} files processed...")

                # Cleanup temp file
                if temp_wav and os.path.exists(temp_wav):
                    os.remove(temp_wav)

            except Exception as e:
                lang_errors += 1
                if file_idx <= 5:  # Only show first few errors
                    print(f"   WARNING: Error processing {os.path.basename(in_file)}: {e}")

        print(f"   Completed: {lang_processed} files -> {lang_segments} segments")
        if lang_errors > 0:
            print(f"   Errors: {lang_errors} files failed")

    # Cleanup temp directory
    temp_dir = os.path.join(PROJECT_ROOT, "speech_forensics", "data", "temp")
    if os.path.exists(temp_dir):
        try:
            shutil.rmtree(temp_dir)
        except:
            pass

    print("\n" + "=" * 70)
    print("PREPROCESSING COMPLETE!")
    print("=" * 70)
    print(f"Total files processed: {total_files_processed}")
    print(f"Total segments created: {total_segments_created}")
    print(f"Output location: {DATA_PROCESSED_REAL}")
    print(f"Ready for training! Run: python -m src.train")
    print("=" * 70)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Preprocess Indian Language Audio Dataset")
    parser.add_argument("--max-per-lang", type=int, default=None,
                        help="Maximum files to process per language (default: all)")
    parser.add_argument("--max-total", type=int, default=None,
                        help="Maximum total files to process (default: all)")
    
    args = parser.parse_args()
    
    preprocess_indian_dataset(
        max_files_per_language=args.max_per_lang,
        max_total_files=args.max_total
    )

