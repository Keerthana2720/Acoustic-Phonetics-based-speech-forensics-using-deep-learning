"""
Inference script for speech forensics analysis
Provides command-line interface with all outputs
"""
import argparse
import os
import sys
import yaml
import torch
import numpy as np
import librosa
import io

# Fix Unicode encoding on Windows
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.append(PROJECT_ROOT)

from src import features
from src import model
from src import report
from src import localization
from src import visualize

# Attack type labels
ATTACK_TYPES = {
    0: "TTS (Text-to-Speech)",
    1: "VC (Voice Conversion)",
    2: "Replay Attack",
    3: "GAN-based Deepfake",
    4: "Splicing / Editing"
}


def main():
    parser = argparse.ArgumentParser(description='Speech Forensics Inference')
    parser.add_argument('--audio', type=str, required=True, help='Path to audio file')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to model checkpoint')
    parser.add_argument('--output', type=str, default=None, help='Output directory for results')
    parser.add_argument('--localize', action='store_true', help='Enable tampered region localization')
    parser.add_argument('--visualize', action='store_true', help='Generate visualizations')
    parser.add_argument('--report', action='store_true', default=True, help='Generate final report')
    parser.add_argument('--decision-threshold', type=float, default=None,
                        help='Override decision threshold (default from config, typically 0.5)')
    parser.add_argument('--uncertainty-margin', type=float, default=None,
                        help='Margin around threshold to consider near-ties (default from config, typically 0.05)')
    parser.add_argument('--bias', type=str, default='neutral', choices=['neutral', 'real', 'fake'],
                        help='How to break near-tie decisions inside the uncertainty margin')
    parser.add_argument('--swap-classes', action='store_true',
                        help='Swap interpretation of auth logits if training used opposite label order')
    
    args = parser.parse_args()
    
    # Load config
    config_path = os.path.join(PROJECT_ROOT, args.config)
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load model
    print("Loading model...")
    net = model.MultiStreamForensics(config)
    net.eval()
    
    # Enforce checkpoint presence to avoid untrained predictions
    checkpoint_path = None
    if args.checkpoint:
        checkpoint_path = os.path.join(PROJECT_ROOT, args.checkpoint)
    else:
        checkpoint_path = os.path.join(PROJECT_ROOT, "checkpoint.pth")

    if not os.path.exists(checkpoint_path):
        raise SystemExit(
            f"Checkpoint not found at {checkpoint_path}. "
            "Provide a trained checkpoint via --checkpoint."
        )

    # Load checkpoint
    state = torch.load(checkpoint_path, map_location='cpu')
    if isinstance(state, dict) and 'model_state_dict' in state:
        missing, unexpected = net.load_state_dict(state['model_state_dict'], strict=False)
        ckpt_val_acc = state.get('val_accuracy', None)
        if ckpt_val_acc is not None:
            print(f"Checkpoint reported val_accuracy: {ckpt_val_acc:.2f}%")
    else:
        missing, unexpected = net.load_state_dict(state, strict=False)

    if missing:
        print(f"Warning: missing weights: {missing}")
    if unexpected:
        print(f"Warning: unexpected weights: {unexpected}")
    print(f"Loaded checkpoint from {checkpoint_path}")
    
    # Load audio
    print(f"Loading audio: {args.audio}")
    y, sr = librosa.load(args.audio, sr=config.get('sample_rate', 16000))
    total_duration = len(y) / float(sr)
    
    # Extract features
    print("Extracting features...")
    lfcc_feat = features.lfcc(y, sr)
    pfe_feat = features.pitch_formants_energy(y, sr)
    ppg_extractor = features.PPGExtractor()
    ppg_feat = ppg_extractor.extract(y, sr)
    
    # Decision thresholds (allow config override)
    decision_threshold = float(
        args.decision_threshold
        if args.decision_threshold is not None
        else config.get("model", {}).get("decision_threshold", 0.5)
    )
    uncertainty_margin = float(
        args.uncertainty_margin
        if args.uncertainty_margin is not None
        else config.get("model", {}).get("uncertainty_margin", 0.05)
    )  # 5% margin default
    tie_bias = args.bias  # neutral | real | fake

    # Prediction
    print("Running inference...")
    with torch.no_grad():
        pred_result = net.predict(
            lfcc_feat,
            pfe_feat,
            ppg_feat,
            return_probs=True
        )

    # Optional class swap if model was trained with flipped labels
    if args.swap_classes:
        orig_pred = pred_result['prediction']
        pred_result['prediction'] = 0 if orig_pred == 1 else 1
        fake_prob_old = pred_result['fake_prob']
        pred_result['fake_prob'] = pred_result['real_prob']
        pred_result['real_prob'] = fake_prob_old
        # Attack type unchanged; only real/fake swap

    # Apply thresholding to avoid near-50/50 flipping
    fake_prob = pred_result['fake_prob'] / 100.0
    real_prob = pred_result['real_prob'] / 100.0
    pred_adjusted = pred_result['prediction']
    uncertain = False

    # Decide with margin; if inside margin, apply bias
    if fake_prob >= decision_threshold + uncertainty_margin:
        pred_adjusted = 1
    elif real_prob >= decision_threshold + uncertainty_margin:
        pred_adjusted = 0
    else:
        uncertain = True
        if tie_bias == 'fake':
            pred_adjusted = 1
        elif tie_bias == 'real':
            pred_adjusted = 0
        else:
            pred_adjusted = pred_result['prediction']  # keep original argmax

    pred_result['prediction'] = pred_adjusted
    pred_result['uncertain'] = uncertain
    
    # Display results
    print("\n" + "=" * 60)
    print("SPEECH FORENSICS ANALYSIS RESULTS")
    print("=" * 60)
    print(f"\nInput File: {os.path.basename(args.audio)}")
    
    # Final Verdict
    print("\n" + "-" * 60)
    print("FINAL VERDICT")
    print("-" * 60)
    if pred_result.get('uncertain'):
        print("⚠️  UNCERTAIN / NEAR-TIE DECISION")
        print(f"(Real: {pred_result['real_prob']:.2f}% | Fake: {pred_result['fake_prob']:.2f}%)")
        print(f"Applied bias: {tie_bias}")
    if pred_result['prediction'] == 0:
        print("✅ REAL / GENUINE SPEECH")
        print("\nCharacteristics:")
        print("  ✓ Authentic human voice")
        print("  ✓ Natural phonetic transitions")
        print("  ✓ Stable formant structure")
    else:
        print("⚠️  FAKE / SPOOFED SPEECH DETECTED")
        print("\nCharacteristics:")
        print("  ✗ Synthesized (TTS/VC)")
        print("  ✗ Deepfake detected")
        print("  ✗ Possible replay attack or editing")
    
    # Confidence Scores
    print("\n" + "-" * 60)
    print("CONFIDENCE SCORES")
    print("-" * 60)
    print(f"Overall Confidence: {pred_result['confidence']:.2f}%")
    print(f"Real Speech Probability: {pred_result['real_prob']:.2f}%")
    print(f"Fake Speech Probability: {pred_result['fake_prob']:.2f}%")
    
    # Attack Type
    if pred_result['prediction'] == 1:
        print("\n" + "-" * 60)
        print("ATTACK TYPE IDENTIFICATION")
        print("-" * 60)
        attack_name = ATTACK_TYPES.get(pred_result['attack_type'], 
                                      f"Unknown (Type {pred_result['attack_type']})")
        print(f"Attack Type: {attack_name}")
        print(f"Attack Type Confidence: {pred_result['attack_confidence']:.2f}%")
    
    # Tampered Region Localization
    tampered_regions = None
    if args.localize or pred_result['prediction'] == 1:  # Auto-enable for fake audio
        print("\n" + "-" * 60)
        print("TAMPERED REGION LOCALIZATION")
        print("-" * 60)
        try:
            tampered_regions = localization.localize_tampered_regions(
                args.audio,
                net,
                sr=sr
            )
            timeline_text = localization.format_timeline(tampered_regions)
            print(timeline_text)
        except Exception as e:
            print(f"Localization analysis encountered an issue: {str(e)}")

    # Tampering summary (coverage + AI/Human duplicate type)
    tampering_summary = localization.summarize_tampering(
        tampered_regions or [],
        total_duration=total_duration,
        overall_prediction=pred_result['prediction']
    )

    print("\n" + "-" * 60)
    print("TAMPERING SUMMARY")
    print("-" * 60)
    print(f"Tampered Portion: {tampering_summary['tampered_percentage']:.2f}% "
          f"({tampering_summary['tampered_seconds']:.2f}s)")
    print(f"Genuine Portion: {tampering_summary['real_percentage']:.2f}% "
          f"({tampering_summary['real_seconds']:.2f}s)")
    print(f"Duplicate / Tamper Type: {tampering_summary['duplicate_type']}")
    
    # Generate visualizations
    if args.visualize:
        print("\n" + "-" * 60)
        print("GENERATING VISUALIZATIONS")
        print("-" * 60)
        
        output_dir = args.output or os.path.dirname(args.audio)
        os.makedirs(output_dir, exist_ok=True)
        
        base_name = os.path.splitext(os.path.basename(args.audio))[0]
        
        try:
            # All features
            fig = visualize.plot_all_features(y, sr, lfcc_feat, pfe_feat, ppg_feat)
            fig.savefig(os.path.join(output_dir, f"{base_name}_all_features.png"), 
                       dpi=150, bbox_inches='tight')
            print(f"Saved: {base_name}_all_features.png")
            
            # Individual plots
            fig = visualize.plot_spectrogram(y, sr)
            fig.savefig(os.path.join(output_dir, f"{base_name}_spectrogram.png"), 
                       dpi=150, bbox_inches='tight')
            print(f"Saved: {base_name}_spectrogram.png")
            
            f0 = pfe_feat[:, 0]
            fig = visualize.plot_pitch_contour(f0, sr)
            fig.savefig(os.path.join(output_dir, f"{base_name}_pitch.png"), 
                       dpi=150, bbox_inches='tight')
            print(f"Saved: {base_name}_pitch.png")
            
            F1, F2, F3 = pfe_feat[:, 2], pfe_feat[:, 3], pfe_feat[:, 4]
            fig = visualize.plot_formants(F1, F2, F3, sr)
            fig.savefig(os.path.join(output_dir, f"{base_name}_formants.png"), 
                       dpi=150, bbox_inches='tight')
            print(f"Saved: {base_name}_formants.png")
            
            energy = pfe_feat[:, 1]
            fig = visualize.plot_energy_curve(energy, sr)
            fig.savefig(os.path.join(output_dir, f"{base_name}_energy.png"), 
                       dpi=150, bbox_inches='tight')
            print(f"Saved: {base_name}_energy.png")
            
            # GENERATE TAMPERED REGION VISUALIZATIONS IF FAKE DETECTED
            if pred_result['prediction'] == 1 and tampered_regions:
                print("\nGenerating tampered region visualizations...")
                for i, region in enumerate(tampered_regions):
                    start_time = region['start_time']
                    end_time = region['end_time']
                    conf = region.get('confidence', 0)
                    
                    # Tampered region spectrogram
                    fig = visualize.plot_tampered_region_spectrogram(y, sr, start_time, end_time)
                    fig.savefig(os.path.join(output_dir, 
                               f"{base_name}_tampered_region_{i:02d}_spectrogram.png"), 
                               dpi=150, bbox_inches='tight')
                    print(f"Saved: {base_name}_tampered_region_{i:02d}_spectrogram.png")
                    
            elif pred_result['prediction'] == 1:
                print("\n(No specific tampered regions localized; enable --localize for detailed analysis)")
            
        except Exception as e:
            print(f"Visualization generation encountered an issue: {str(e)}")
    
    # Generate final report
    if args.report:
        print("\n" + "-" * 60)
        print("GENERATING FINAL REPORT")
        print("-" * 60)
        
        # Generate evidence
        evidence = report.generate_evidence_from_features(
            pred_result['prediction'],
            pred_result['confidence'],
            pfe_feat,
            lfcc_feat,
            ppg_feat
        )
        
        # Generate report
        output_dir = args.output or os.path.dirname(args.audio)
        os.makedirs(output_dir, exist_ok=True)
        
        report_path = os.path.join(output_dir, 
                                  f"forensic_report_{os.path.basename(args.audio)}.txt")
        
        report_text = report.generate_report(
            input_file=args.audio,
            prediction=pred_result['prediction'],
            confidence=pred_result['confidence'],
            real_prob=pred_result['real_prob'],
            fake_prob=pred_result['fake_prob'],
            attack_type=pred_result.get('attack_type') if pred_result['prediction'] == 1 else None,
            attack_confidence=pred_result.get('attack_confidence') if pred_result['prediction'] == 1 else None,
            tampered_regions=tampered_regions,
            tampering_summary=tampering_summary,
            evidence=evidence,
            output_path=report_path
        )
        
        print(f"Report saved to: {report_path}")
    
    print("\n" + "=" * 60)
    print("Analysis complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

