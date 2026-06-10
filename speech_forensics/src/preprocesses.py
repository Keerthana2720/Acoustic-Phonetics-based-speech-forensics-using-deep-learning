import os
import librosa
import soundfile as sf
from pydub import AudioSegment
import numpy as np

# ----------------------------------------------------------
# CONFIGURATION
# ----------------------------------------------------------

INPUT_ROOT = r"C:\Users\mkart\OneDrive\Desktop\major project\Indian_Languages_Audio_Dataset"        # your raw folders
OUTPUT_ROOT = r"C:\Users\mkart\OneDrive\Desktop\major project\Indian_Languages_Audio_Dataset"           # where processed files will be saved
SEGMENT_DURATION = 3                     # 3-second segments
TARGET_SR = 16000                        # sampling rate

# ----------------------------------------------------------
# UTIL FUNCTIONS
# ----------------------------------------------------------

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def convert_to_wav(input_file, out_file):
    """Converts MP3/M4A/others → WAV mono 16k"""
    audio = AudioSegment.from_file(input_file)
    audio = audio.set_channels(1)
    audio = audio.set_frame_rate(TARGET_SR)
    audio.export(out_file, format="wav")
    return out_file

def load_audio(path):
    y, sr = librosa.load(path, sr=TARGET_SR, mono=True)
    return y, sr

def normalize_audio(y):
    return y / (np.max(np.abs(y)) + 1e-9)

def remove_silence(y, sr):
    y_trimmed, _ = librosa.effects.trim(y, top_db=30)
    return y_trimmed

def segment_audio(y, sr, segment_duration, output_base):
    """Split into n-second chunks"""
    seg_len = int(segment_duration * sr)
    count = 0
    segments = []

    for start in range(0, len(y), seg_len):
        end = start + seg_len
        chunk = y[start:end]

        if len(chunk) < seg_len * 0.6:   # skip very small clips
            continue

        out_file = f"{output_base}_seg{count}.wav"
        sf.write(out_file, chunk, sr)
        segments.append(out_file)
        count += 1

    return segments

# ----------------------------------------------------------
# MAIN PROCESSOR
# ----------------------------------------------------------

def preprocess():
    print("🔍 Starting preprocessing for all 10 Indian language datasets...\n")

    languages = os.listdir(INPUT_ROOT)

    for lang in languages:
        lang_path = os.path.join(INPUT_ROOT, lang)
        if not os.path.isdir(lang_path):
            continue

        print(f"📌 Processing language: {lang}")
        
        out_lang_path = os.path.join(OUTPUT_ROOT, lang)
        ensure_dir(out_lang_path)

        for root, dirs, files in os.walk(lang_path):
            for file in files:
                in_file = os.path.join(root, file)

                if not file.lower().endswith((".wav", ".mp3", ".m4a", ".flac")):
                    continue

                # Temporary WAV path
                temp_wav = os.path.join(out_lang_path, file.replace(".mp3", ".wav")
                                                           .replace(".m4a", ".wav")
                                                           .replace(".flac", ".wav"))  

                # Convert to WAV if needed
                if not file.lower().endswith(".wav"):
                    convert_to_wav(in_file, temp_wav)
                    wav_file = temp_wav
                else:
                    wav_file = in_file

                # Load & preprocess
                y, sr = load_audio(wav_file)
                y = normalize_audio(y)
                y = remove_silence(y)

                # Segment and save
                base_name = os.path.splitext(file)[0]
                output_base = os.path.join(out_lang_path, base_name)

                segment_audio(y, sr, SEGMENT_DURATION, output_base)

        print(f"✅ Completed: {lang}\n")

    print("🎉 All language datasets processed successfully!")
    print(f"📁 Saved to: {OUTPUT_ROOT}")

if __name__ == "__main__":
    preprocess()
