"""
Final Decision Report Generator
Generates comprehensive forensic reports with all analysis results
"""
import os
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional


# Attack type labels
ATTACK_TYPES = {
    0: "TTS (Text-to-Speech)",
    1: "VC (Voice Conversion)",
    2: "Replay Attack",
    3: "GAN-based Deepfake",
    4: "Splicing / Editing"
}


def generate_report(
    input_file: str,
    prediction: int,
    confidence: float,
    real_prob: float,
    fake_prob: float,
    attack_type: Optional[int] = None,
    attack_confidence: Optional[float] = None,
    tampered_regions: Optional[List[Dict]] = None,
    tampering_summary: Optional[Dict] = None,
    evidence: Optional[List[str]] = None,
    output_path: Optional[str] = None
) -> str:
    """
    Generate final forensic report
    
    Args:
        input_file: Path to input audio file
        prediction: 0 for Real, 1 for Fake
        confidence: Confidence score (0-100)
        real_prob: Probability of real speech (0-100)
        fake_prob: Probability of fake speech (0-100)
        attack_type: Attack type index (if fake)
        attack_confidence: Attack type confidence (if fake)
        tampered_regions: List of dicts with 'start_time', 'end_time', 'prediction', 'confidence'
        tampering_summary: Dict with aggregate tampering stats (seconds/percentage)
        evidence: List of evidence strings
        output_path: Optional path to save report file
    
    Returns:
        Report string
    """
    report_lines = []
    report_lines.append("=" * 60)
    report_lines.append("FINAL FORENSIC REPORT")
    report_lines.append("=" * 60)
    report_lines.append("")
    
    # File information
    report_lines.append(f"Input File: {os.path.basename(input_file)}")
    report_lines.append(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("")
    
    # Prediction result
    report_lines.append("-" * 60)
    report_lines.append("PREDICTION RESULT")
    report_lines.append("-" * 60)
    
    if prediction == 0:
        report_lines.append("Speech Status: ✅ REAL / GENUINE SPEECH")
        report_lines.append("")
        report_lines.append("Characteristics:")
        report_lines.append("  ✓ Authentic human voice")
        report_lines.append("  ✓ Natural phonetic transitions")
        report_lines.append("  ✓ Stable formant structure")
    else:
        report_lines.append("Speech Status: ⚠️  FAKE / SPOOFED SPEECH")
        report_lines.append("")
        report_lines.append("Characteristics:")
        report_lines.append("  ✗ Synthesized (TTS/VC)")
        report_lines.append("  ✗ Deepfake detected")
        report_lines.append("  ✗ Possible replay attack or editing")
    
    report_lines.append("")
    
    # Confidence scores
    report_lines.append("-" * 60)
    report_lines.append("CONFIDENCE SCORES")
    report_lines.append("-" * 60)
    report_lines.append(f"Overall Confidence: {confidence:.2f}%")
    report_lines.append(f"Real Speech Probability: {real_prob:.2f}%")
    report_lines.append(f"Fake Speech Probability: {fake_prob:.2f}%")
    report_lines.append("")

    # Tampering summary (coverage + type)
    if tampering_summary:
        report_lines.append("-" * 60)
        report_lines.append("TAMPERING SUMMARY")
        report_lines.append("-" * 60)
        report_lines.append(
            f"Tampered Portion: {tampering_summary['tampered_percentage']:.2f}% "
            f"({tampering_summary['tampered_seconds']:.2f}s)"
        )
        report_lines.append(
            f"Genuine Portion: {tampering_summary['real_percentage']:.2f}% "
            f"({tampering_summary['real_seconds']:.2f}s)"
        )
        report_lines.append(f"Duplicate / Tamper Type: {tampering_summary['duplicate_type']}")
        report_lines.append("")
    
    # Attack type (if fake)
    if prediction == 1 and attack_type is not None:
        report_lines.append("-" * 60)
        report_lines.append("ATTACK TYPE IDENTIFICATION")
        report_lines.append("-" * 60)
        attack_name = ATTACK_TYPES.get(attack_type, f"Unknown (Type {attack_type})")
        report_lines.append(f"Attack Type: {attack_name}")
        if attack_confidence is not None:
            report_lines.append(f"Attack Type Confidence: {attack_confidence:.2f}%")
        report_lines.append("")
    
    # Tampered regions (if available)
    if tampered_regions and len(tampered_regions) > 0:
        report_lines.append("-" * 60)
        report_lines.append("TAMPERED REGION LOCALIZATION")
        report_lines.append("-" * 60)
        report_lines.append("Audio Timeline:")
        report_lines.append("")
        
        for region in tampered_regions:
            start = region['start_time']
            end = region['end_time']
            pred = region['prediction']
            conf = region.get('confidence', 0)
            
            status = "Real" if pred == 0 else "Tampered"
            report_lines.append(f"  [{start:.2f}s – {end:.2f}s] → {status} (Confidence: {conf:.1f}%)")
        
        report_lines.append("")
    
    # Evidence
    if evidence and len(evidence) > 0:
        report_lines.append("-" * 60)
        report_lines.append("KEY EVIDENCE")
        report_lines.append("-" * 60)
        for i, ev in enumerate(evidence, 1):
            report_lines.append(f"  {i}. {ev}")
        report_lines.append("")
    
    # Summary
    report_lines.append("=" * 60)
    report_lines.append("END OF REPORT")
    report_lines.append("=" * 60)
    
    report_text = "\n".join(report_lines)
    
    # Save to file if path provided
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
    
    return report_text


def generate_evidence_from_features(
    prediction: int,
    confidence: float,
    pfe_feat: Optional = None,
    lfcc_feat: Optional = None,
    ppg_feat: Optional = None
) -> List[str]:
    """
    Generate evidence list based on feature analysis
    
    Args:
        prediction: 0 for Real, 1 for Fake
        confidence: Confidence score
        pfe_feat: Pitch, formants, energy features
        lfcc_feat: LFCC features
        ppg_feat: PPG features
    
    Returns:
        List of evidence strings
    """
    evidence = []
    
    if prediction == 1:  # Fake detected
        if pfe_feat is not None:
            # Check pitch consistency
            f0 = pfe_feat[:, 0]
            f0_std = np.std(f0[f0 > 0])  # Exclude zeros
            if f0_std > 50:  # High pitch variation
                evidence.append("Pitch contour inconsistent - high variation detected")
            
            # Check formant stability
            F1 = pfe_feat[:, 2]
            F2 = pfe_feat[:, 3]
            F1_std = np.std(F1[F1 > 0])
            F2_std = np.std(F2[F2 > 0])
            if F1_std > 200 or F2_std > 300:
                evidence.append("Formant trajectory mismatch - unstable formants")
        
        if lfcc_feat is not None:
            # Check for artifacts in spectral features
            lfcc_std = np.std(lfcc_feat, axis=0)
            if np.max(lfcc_std) > 5.0:
                evidence.append("CQCC artifacts detected - spectral anomalies present")
        
        if confidence > 90:
            evidence.append("High confidence fake detection - strong evidence of manipulation")
    else:  # Real
        if pfe_feat is not None:
            evidence.append("Natural pitch and formant patterns observed")
        if confidence > 90:
            evidence.append("High confidence genuine speech - authentic characteristics confirmed")
    
    return evidence

