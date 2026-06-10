#!/usr/bin/env python
"""
Generate forensic reports and visuals for ALL fake audio files.
This creates the tampered timelines and images that the dashboard will display.
"""
import os
import glob
import subprocess
import sys

def main():
    speech_forensics_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(speech_forensics_dir)
    
    # Find all fake WAV files
    fake_files = sorted(glob.glob(os.path.join("data", "processed", "fake", "fake_*.wav")))
    
    print(f"Found {len(fake_files)} fake audio files.")
    print(f"Generating forensic reports and visuals for all...\n")
    
    for i, wav_file in enumerate(fake_files, start=1):
        filename = os.path.basename(wav_file)
        print(f"[{i}/{len(fake_files)}] Processing: {filename}")
        
        # Run inference with visualization and localization enabled
        cmd = [
            sys.executable, "infer.py",
            "--audio", wav_file,
            "--checkpoint", "checkpoints/audio_forensics.pt",
            "--visualize",
            "--localize"
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, timeout=120)
            if result.returncode == 0:
                print(f"  ✅ Success")
            else:
                print(f"  ⚠️  Warning: returncode {result.returncode}")
                if result.stderr:
                    print(f"     Error: {result.stderr.decode('utf-8', errors='ignore')[:100]}")
        except subprocess.TimeoutExpired:
            print(f"  ❌ Timeout after 120 seconds")
        except Exception as e:
            print(f"  ❌ Error: {str(e)}")
    
    print("\n" + "="*60)
    print("✅ Done! All fake files now have timelines and images.")
    print("="*60)
    print("\nNow you can:")
    print("1. Start the dashboard: python run_dashboard.py")
    print("2. Upload any fake file (e.g., fake_000000_seg000.wav)")
    print("3. The dashboard will show tampered timelines and annotated images")

if __name__ == "__main__":
    main()
