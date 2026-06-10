"""
Visualization module for acoustic-phonetic cues
Generates spectrograms, pitch contours, formant graphs,
energy curves, and PPG visualizations
"""

import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import io
import base64


# --------------------------------------------------
# SPECTROGRAM
# --------------------------------------------------
def plot_spectrogram(y, sr, hop_length=160, n_fft=1024):
    fig, ax = plt.subplots(figsize=(12, 4))

    D = librosa.amplitude_to_db(
        np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length)),
        ref=np.max
    )

    img = librosa.display.specshow(
        D, y_axis='hz', x_axis='time',
        sr=sr, hop_length=hop_length, ax=ax
    )

    ax.set_title("Spectrogram", fontsize=14, fontweight="bold")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    plt.colorbar(img, ax=ax, format="%+2.0f dB")

    plt.tight_layout()
    return fig


def plot_tampered_region_spectrogram(
    y,
    sr,
    start_time: float,
    end_time: float,
    hop_length: int = 160,
    n_fft: int = 1024,
):
    """
    Spectrogram focused on a specific (tampered) time region.

    Args:
        y: Full audio samples
        sr: Sample rate
        start_time: Region start (seconds)
        end_time: Region end (seconds)
    """
    start_sample = int(max(0.0, start_time) * sr)
    end_sample = int(min(len(y) / sr, end_time) * sr)

    if end_sample <= start_sample:
        # Fallback: show whole signal if region is invalid
        return plot_spectrogram(y, sr, hop_length=hop_length, n_fft=n_fft)

    y_region = y[start_sample:end_sample]

    title = f"Region Spectrogram [{start_time:.2f}s – {end_time:.2f}s]"
    fig, ax = plt.subplots(figsize=(8, 3))

    D = librosa.amplitude_to_db(
        np.abs(librosa.stft(y_region, n_fft=n_fft, hop_length=hop_length)),
        ref=np.max,
    )

    img = librosa.display.specshow(
        D,
        y_axis="hz",
        x_axis="time",
        sr=sr,
        hop_length=hop_length,
        ax=ax,
    )

    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    plt.colorbar(img, ax=ax, format="%+2.0f dB")

    plt.tight_layout()
    return fig


# --------------------------------------------------
# CQCC
# --------------------------------------------------
def plot_cqcc(y, sr, hop_length=160):
    CQT = librosa.cqt(y, sr=sr, hop_length=hop_length)
    CQT_db = librosa.amplitude_to_db(np.abs(CQT), ref=np.max)

    fig, ax = plt.subplots(figsize=(12, 4))
    img = librosa.display.specshow(
        CQT_db, x_axis="time", y_axis="cqt_hz",
        sr=sr, hop_length=hop_length, ax=ax
    )

    ax.set_title("Constant-Q Cepstral Coefficients (CQCC)", fontweight="bold")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    plt.colorbar(img, ax=ax, format="%+2.0f dB")

    plt.tight_layout()
    return fig


# --------------------------------------------------
# PITCH
# --------------------------------------------------
def plot_pitch_contour(f0, sr, hop_length=160):
    times = librosa.frames_to_time(
        np.arange(len(f0)), sr=sr, hop_length=hop_length
    )

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(times, f0, linewidth=2)
    ax.set_title("Pitch Contour (F0)", fontweight="bold")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    return fig


# --------------------------------------------------
# FORMANTS
# --------------------------------------------------
def plot_formants(F1, F2, F3, sr, hop_length=160):
    times = librosa.frames_to_time(
        np.arange(len(F1)), sr=sr, hop_length=hop_length
    )

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(times, F1, label="F1", linewidth=2)
    ax.plot(times, F2, label="F2", linewidth=2)
    ax.plot(times, F3, label="F3", linewidth=2)

    ax.set_title("Formant Trajectories", fontweight="bold")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    return fig


# --------------------------------------------------
# ENERGY
# --------------------------------------------------
def plot_energy_curve(energy, sr, hop_length=160):
    times = librosa.frames_to_time(
        np.arange(len(energy)), sr=sr, hop_length=hop_length
    )

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(times, energy, linewidth=2)
    ax.set_title("Energy Curve", fontweight="bold")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Energy")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    return fig


# --------------------------------------------------
# PPG
# --------------------------------------------------
def plot_ppg_visualization(ppg, sr, hop_length=160, top_k=10):
    times = librosa.frames_to_time(
        np.arange(ppg.shape[0]), sr=sr, hop_length=hop_length
    )

    fig, ax = plt.subplots(figsize=(12, 6))
    img = ax.imshow(
        ppg.T, aspect="auto", origin="lower",
        extent=[times[0], times[-1], 0, ppg.shape[1]]
    )

    ax.set_title("Phoneme Posteriorgram (PPG)", fontweight="bold")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Phoneme Index")
    plt.colorbar(img, ax=ax, label="Probability")

    plt.tight_layout()
    return fig


# --------------------------------------------------
# ALL FEATURES (FINAL DASHBOARD VIEW)
# --------------------------------------------------
def plot_all_features(y, sr, lfcc_feat, pfe_feat, ppg_feat):
    hop_length = 160

    f0 = pfe_feat[:, 0]
    energy = pfe_feat[:, 1]
    F1, F2, F3 = pfe_feat[:, 2], pfe_feat[:, 3], pfe_feat[:, 4]

    times = librosa.frames_to_time(
        np.arange(len(f0)), sr=sr, hop_length=hop_length
    )

    fig = plt.figure(figsize=(16, 12))

    # Spectrogram
    ax1 = plt.subplot(3, 2, 1)
    D = librosa.amplitude_to_db(
        np.abs(librosa.stft(y, hop_length=hop_length)), ref=np.max
    )
    img1 = librosa.display.specshow(
        D, y_axis="hz", x_axis="time",
        sr=sr, hop_length=hop_length, ax=ax1
    )
    ax1.set_title("Spectrogram", fontweight="bold")
    plt.colorbar(img1, ax=ax1, format="%+2.0f dB")

    # Pitch
    ax2 = plt.subplot(3, 2, 2)
    ax2.plot(times, f0)
    ax2.set_title("Pitch Contour (F0)", fontweight="bold")
    ax2.grid(alpha=0.3)

    # Formants
    ax3 = plt.subplot(3, 2, 3)
    ax3.plot(times, F1, label="F1")
    ax3.plot(times, F2, label="F2")
    ax3.plot(times, F3, label="F3")
    ax3.legend()
    ax3.set_title("Formant Trajectories", fontweight="bold")

    # Energy
    ax4 = plt.subplot(3, 2, 4)
    ax4.plot(times, energy)
    ax4.set_title("Energy Curve", fontweight="bold")

    # CQCC
    ax5 = plt.subplot(3, 2, 5)
    CQT = librosa.cqt(y, sr=sr, hop_length=hop_length)
    CQT_db = librosa.amplitude_to_db(np.abs(CQT), ref=np.max)
    img5 = librosa.display.specshow(
        CQT_db, x_axis="time", y_axis="cqt_hz",
        sr=sr, hop_length=hop_length, ax=ax5
    )
    ax5.set_title("CQCC", fontweight="bold")
    plt.colorbar(img5, ax=ax5, format="%+2.0f dB")

    # PPG summary
    ax6 = plt.subplot(3, 2, 6)
    ppg_mean = np.mean(ppg_feat, axis=0)
    top = np.argsort(ppg_mean)[-20:]
    ax6.barh(range(len(top)), ppg_mean[top])
    ax6.set_title("Top PPG Phonemes", fontweight="bold")

    plt.tight_layout()
    return fig


# --------------------------------------------------
# STREAMLIT SUPPORT
# --------------------------------------------------
def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return img_base64
