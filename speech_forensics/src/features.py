import numpy as np
import librosa
import parselmouth
try:
    from gammatone.gtgram import gtgram
    HAS_GAMMATONE = True
except ImportError:
    HAS_GAMMATONE = False
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torch

# ---------- Spectral: LFCC ----------
def lfcc(y, sr, n_fft=1024, hop_length=160, n_mels=40, n_lfcc=20):
    S = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=n_fft,
        hop_length=hop_length, n_mels=n_mels, power=2.0
    )
    S = np.log(np.maximum(S, 1e-10))
    c = librosa.feature.mfcc(
        S=librosa.power_to_db(S),
        n_mfcc=n_lfcc
    )
    return c.T  # [frames, n_lfcc]

# ---------- Acoustic-phonetic: pitch + formants + energy ----------
def pitch_formants_energy(y, sr, hop_length=160):
    try:
        # Pitch
        f0 = librosa.yin(y, fmin=50, fmax=500, sr=sr, hop_length=hop_length)
        f0 = np.nan_to_num(f0)
    except:
        # If yin fails, use pyin as fallback
        try:
            f0, _, _ = librosa.pyin(y, fmin=50, fmax=500, sr=sr, hop_length=hop_length)
            f0 = np.nan_to_num(f0)
        except:
            # If both fail, return zeros
            f0 = np.zeros(len(y) // hop_length + 1)

    # Energy
    energy = librosa.feature.rms(
        y=y,
        frame_length=hop_length * 2,
        hop_length=hop_length
    )[0]

    # Formants (Praat)
    try:
        snd = parselmouth.Sound(y, sampling_frequency=sr)
        formant = snd.to_formant_burg(time_step=hop_length / sr)
    except:
        formant = None

    n_frames = len(f0)
    F1 = np.zeros(n_frames)
    F2 = np.zeros(n_frames)
    F3 = np.zeros(n_frames)

    if formant is not None:
        for i in range(n_frames):
            t = i * (hop_length / sr)
            try:
                f1_val = formant.get_value_at_time(1, t)
                f2_val = formant.get_value_at_time(2, t)
                f3_val = formant.get_value_at_time(3, t)
                
                # Handle NaN values from formant extraction
                F1[i] = f1_val if np.isfinite(f1_val) else 0
                F2[i] = f2_val if np.isfinite(f2_val) else 0
                F3[i] = f3_val if np.isfinite(f3_val) else 0
            except:
                pass

    # Ensure all values are finite
    F1 = np.nan_to_num(F1, nan=0.0, posinf=0.0, neginf=0.0)
    F2 = np.nan_to_num(F2, nan=0.0, posinf=0.0, neginf=0.0)
    F3 = np.nan_to_num(F3, nan=0.0, posinf=0.0, neginf=0.0)
    
    feats = np.stack([f0, energy, F1, F2, F3], axis=-1)
    # Final safety check
    feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
    return feats  # [frames, 5]

# ---------- Phonetic stream: PPG via Wav2Vec2 ----------
class PPGExtractor:
    def __init__(self, model_id="facebook/wav2vec2-base-960h"):
        self.processor = Wav2Vec2Processor.from_pretrained(model_id)
        self.model = Wav2Vec2ForCTC.from_pretrained(model_id)
        self.model.eval()

    def ppg(self, y, sr):
        # Wav2Vec2 expects 16kHz
        inputs = self.processor(
            y,
            sampling_rate=sr,
            return_tensors="pt",
            padding=True
        )

        with torch.no_grad():
            logits = self.model(**inputs).logits.squeeze(0)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()

        return probs  # [frames, vocab_size]
    
    def extract(self, y, sr, target_dim=32):
        """
        Extract PPG features and reduce to target dimension
        
        Args:
            y: Audio signal
            sr: Sample rate
            target_dim: Target feature dimension (default 32)
        
        Returns:
            PPG features of shape [frames, target_dim]
        """
        probs = self.ppg(y, sr)  # [frames, vocab_size]
        
        # Wav2Vec2-base vocab size is 32, so we can just return it
        # If vocab_size > target_dim, take first target_dim dimensions
        # If vocab_size < target_dim, pad with zeros
        if probs.shape[-1] > target_dim:
            return probs[:, :target_dim]
        elif probs.shape[-1] < target_dim:
            padding = np.zeros((probs.shape[0], target_dim - probs.shape[-1]))
            return np.concatenate([probs, padding], axis=-1)
        else:
            return probs

# ---------- Gammatone features ----------
def gammatone_features(y, sr, n_filters=64, window_time=0.025, hop_time=0.010):
    if not HAS_GAMMATONE:
        # Fallback: return zeros if gammatone is not available
        n_frames = int((len(y) / sr - window_time) / hop_time) + 1
        return np.zeros((n_frames, n_filters))
    gt = gtgram(y, sr, window_time, hop_time, n_filters)
    return gt.T  # [frames, n_filters]


# ---------- Simple augmentations (on-waveform) ----------
def augment_pitch_shift(y, sr, n_steps=2):
    try:
        return librosa.effects.pitch_shift(y, sr, n_steps)
    except Exception:
        return y


def augment_time_stretch(y, rate=1.1):
    try:
        # Time stretch requires a minimum length; guard it
        if len(y) < 100:
            return y
        return librosa.effects.time_stretch(y, rate)
    except Exception:
        return y


def augment_add_noise(y, snr_db=20):
    try:
        rms = np.sqrt(np.mean(y ** 2)) + 1e-9
        snr_linear = 10 ** (snr_db / 20.0)
        noise_std = rms / snr_linear
        noise = np.random.normal(0.0, noise_std, size=y.shape)
        return y + noise
    except Exception:
        return y


def apply_augmentations(y, sr, cfg=None):
    """Apply a set of randomized augmentations based on cfg dict

    cfg keys (examples):
      - pitch_prob: probability to apply pitch shift
      - pitch_steps: max steps for pitch shift (uniform +/-)
      - stretch_prob: probability to apply time stretch
      - stretch_rates: tuple (min, max)
      - noise_prob: probability to add noise
      - noise_snr_db: min snr, max snr
    """
    if cfg is None:
        cfg = {}

    y_out = y.copy()

    # Pitch
    if np.random.rand() < cfg.get('pitch_prob', 0.3):
        steps = cfg.get('pitch_steps', 2)
        n_steps = np.random.uniform(-steps, steps)
        y_out = augment_pitch_shift(y_out, sr, n_steps)

    # Time stretch
    if np.random.rand() < cfg.get('stretch_prob', 0.3):
        min_r, max_r = cfg.get('stretch_rates', (0.9, 1.15))
        rate = np.random.uniform(min_r, max_r)
        y_out = augment_time_stretch(y_out, rate)

    # Add noise
    if np.random.rand() < cfg.get('noise_prob', 0.5):
        min_snr, max_snr = cfg.get('noise_snr_db', (10, 30))
        if isinstance(min_snr, (list, tuple)) or isinstance(min_snr, np.ndarray):
            snr = np.random.uniform(min_snr[0], min_snr[1])
        elif isinstance(max_snr, (list, tuple)) or isinstance(max_snr, np.ndarray):
            snr = np.random.uniform(min_snr, max_snr)
        else:
            snr = cfg.get('noise_snr_db', 20)
        y_out = augment_add_noise(y_out, snr)

    # Ensure no NaNs
    y_out = np.nan_to_num(y_out)
    return y_out
