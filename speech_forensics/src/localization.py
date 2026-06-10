"""
Tampered Region Localization
Identifies which portions of audio are manipulated
"""
import numpy as np
import librosa
from typing import List, Dict, Tuple
from . import features
from . import model


def localize_tampered_regions(
    audio_path: str,
    model_instance,
    sr: int = 16000,
    segment_duration: float = 1.0,
    hop_duration: float = 0.5,
    threshold: float = 0.5,
    device=None
) -> List[Dict]:
    """
    Localize tampered regions in audio using sliding window analysis
    
    Args:
        audio_path: Path to audio file
        model_instance: Trained model instance
        sr: Sample rate
        segment_duration: Duration of each analysis segment in seconds
        hop_duration: Hop size between segments in seconds
        threshold: Confidence threshold for tampering detection
    
    Returns:
        List of dictionaries with 'start_time', 'end_time', 'prediction', 'confidence'
    """
    # Load audio
    y, sr = librosa.load(audio_path, sr=sr)
    duration = len(y) / sr
    
    # Extract features for entire audio
    lfcc_feat = features.lfcc(y, sr)
    pfe_feat = features.pitch_formants_energy(y, sr)
    ppg_extractor = features.PPGExtractor()
    ppg_feat = ppg_extractor.extract(y, sr)

    # ------------------------------------------------------------------
    # SAFETY: enforce same temporal length across all feature streams
    # ------------------------------------------------------------------
    min_len = min(len(lfcc_feat), len(pfe_feat), len(ppg_feat))
    if min_len <= 0:
        # Very short / silent / invalid audio: nothing to localize
        return []

    # Truncate all streams to the same length to avoid empty slices
    lfcc_feat = lfcc_feat[:min_len]
    pfe_feat = pfe_feat[:min_len]
    ppg_feat = ppg_feat[:min_len]
    
    # Calculate frame rate (approximate)
    hop_length = 160  # From feature extraction
    frame_rate = sr / hop_length  # frames per second
    
    # Segment features
    seg_frames = int(segment_duration * frame_rate)
    hop_frames = int(hop_duration * frame_rate)
    
    results = []
    
    for start_frame in range(0, len(lfcc_feat), hop_frames):
        end_frame = min(start_frame + seg_frames, len(lfcc_feat))

        # Require full window; skip any partial/short segments to avoid padding issues
        if end_frame - start_frame < seg_frames:
            continue

        # Extract segment features
        lfcc_seg = lfcc_feat[start_frame:end_frame]
        pfe_seg = pfe_feat[start_frame:end_frame]
        ppg_seg = ppg_feat[start_frame:end_frame]

        # Skip segments where any stream is empty (extra safety)
        if (
            lfcc_seg.size == 0
            or pfe_seg.size == 0
            or ppg_seg.size == 0
        ):
            continue
        
        # Predict, guarding against zero-length errors in the RNN
        try:
            # Get device from model if not provided
            if device is None and hasattr(model_instance, 'parameters'):
                device = next(model_instance.parameters()).device
            pred_result = model_instance.predict(
                lfcc_seg, pfe_seg, ppg_seg, return_probs=True, device=device
            )
        except RuntimeError as e:
            # If the underlying LSTM still complains about sequence length,
            # skip this window instead of failing the whole localization.
            if "Expected sequence length to be larger than 0" in str(e):
                continue
            raise
        
        start_time = start_frame / frame_rate
        end_time = end_frame / frame_rate
        
        results.append({
            'start_time': start_time,
            'end_time': end_time,
            'prediction': pred_result['prediction'],
            'confidence': pred_result['confidence'],
            'real_prob': pred_result['real_prob'],
            'fake_prob': pred_result['fake_prob']
        })
    
    return results


def format_timeline(results: List[Dict]) -> str:
    """
    Format tampered region results as timeline string
    
    Args:
        results: List of region dictionaries
    
    Returns:
        Formatted timeline string
    """
    timeline_lines = []
    timeline_lines.append("Audio Timeline:")
    timeline_lines.append("")
    
    for region in results:
        start = region['start_time']
        end = region['end_time']
        pred = region['prediction']
        conf = region.get('confidence', 0)
        
        status = "Real" if pred == 0 else "Tampered"
        timeline_lines.append(f"  [{start:.2f}s – {end:.2f}s] → {status} (Confidence: {conf:.1f}%)")
    
    return "\n".join(timeline_lines)


def merge_adjacent_regions(results: List[Dict], same_prediction: bool = True) -> List[Dict]:
    """
    Merge adjacent regions with same prediction
    
    Args:
        results: List of region dictionaries
        same_prediction: If True, merge only same predictions; if False, merge all adjacent
    
    Returns:
        Merged list of regions
    """
    if not results:
        return []
    
    merged = [results[0].copy()]
    
    for current in results[1:]:
        last = merged[-1]
        
        # Check if should merge
        should_merge = False
        if same_prediction:
            should_merge = (last['prediction'] == current['prediction'] and 
                          abs(last['end_time'] - current['start_time']) < 0.1)
        else:
            should_merge = abs(last['end_time'] - current['start_time']) < 0.1
        
        if should_merge:
            # Merge regions
            merged[-1]['end_time'] = current['end_time']
            merged[-1]['confidence'] = (last['confidence'] + current['confidence']) / 2
        else:
            merged.append(current.copy())
    
    return merged


def summarize_tampering(
    results: List[Dict],
    total_duration: float,
    overall_prediction: int = None
) -> Dict:
    """
    Summarize tampered vs real coverage over the full audio.

    Args:
        results: List of region dictionaries from localization
        total_duration: Total audio duration in seconds
        overall_prediction: Optional global model decision (0 real, 1 fake)

    Returns:
        Dictionary with seconds/percentages and an overall duplicate type label
    """
    if total_duration is None or total_duration <= 0:
        return {
            'tampered_seconds': 0.0,
            'tampered_percentage': 0.0,
            'real_seconds': 0.0,
            'real_percentage': 0.0,
            'duplicate_type': "Unknown"
        }

    tampered_seconds = 0.0
    real_seconds = 0.0

    for region in results or []:
        start = max(0.0, float(region.get('start_time', 0.0)))
        end = min(float(region.get('end_time', 0.0)), total_duration)
        duration = max(0.0, end - start)
        if region.get('prediction', 0) == 1:
            tampered_seconds += duration
        else:
            real_seconds += duration

    # Clamp to total duration
    tampered_seconds = min(tampered_seconds, total_duration)
    real_seconds = min(max(total_duration - tampered_seconds, 0.0), total_duration)

    tampered_pct = (tampered_seconds / total_duration) * 100.0
    real_pct = (real_seconds / total_duration) * 100.0

    duplicate_type = (
        "AI / synthetic manipulation"
        if (overall_prediction == 1 or tampered_seconds > 0)
        else "Human / genuine speech"
    )

    return {
        'tampered_seconds': tampered_seconds,
        'tampered_percentage': tampered_pct,
        'real_seconds': real_seconds,
        'real_percentage': real_pct,
        'duplicate_type': duplicate_type
    }
