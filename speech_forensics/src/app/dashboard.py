import streamlit as st
import numpy as np
import librosa
import torch
import yaml
import os
import sys

# ----------------------------------------------------

# FIX PROJECT ROOT

# ----------------------------------------------------
# ----------------------------------------------------
# FIX PROJECT ROOT (VERY IMPORTANT)
# ----------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


from src import features
from src import model
from src.visualize import plot_tampered_region_spectrogram, plot_all_features

# ----------------------------------------------------
# STREAMLIT CONFIG + GLOBAL STYLES
# ----------------------------------------------------

st.set_page_config(page_title="Speech Forensics Dashboard", layout="wide")

# Custom CSS for attractive, modern UI with better navigation
st.markdown(
    """
    <style>
        /* Global page style */
        .main {
            background: linear-gradient(135deg, #0f0f1e 0%, #1a1a2e 100%);
            color: #e8e8f5;
        }

        /* Hero header */
        .hero-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 2.5rem 2rem;
            border-radius: 16px;
            margin-bottom: 2rem;
            box-shadow: 0 20px 60px rgba(102, 126, 234, 0.25);
            color: white;
            position: relative;
            overflow: hidden;
        }

        .hero-header::before {
            content: '';
            position: absolute;
            top: -50%;
            right: -10%;
            width: 300px;
            height: 300px;
            background: rgba(255,255,255,0.05);
            border-radius: 50%;
            pointer-events: none;
        }

        .hero-title {
            font-size: 2.2rem;
            font-weight: 800;
            margin: 0;
            position: relative;
            z-index: 1;
        }

        .hero-subtitle {
            font-size: 1.05rem;
            margin-top: 0.5rem;
            opacity: 0.95;
            position: relative;
            z-index: 1;
        }

        /* Card-style container */
        .sf-card {
            background: linear-gradient(135deg, #1e1e3f 0%, #2d2d4a 100%);
            border-radius: 16px;
            padding: 1.6rem;
            margin-bottom: 1.5rem;
            border: 1px solid rgba(102, 126, 234, 0.3);
            transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease;
            position: relative;
        }

        .sf-card:hover {
            transform: translateY(-4px);
            box-shadow: 0 25px 50px rgba(102, 126, 234, 0.2);
            border-color: #667eea;
        }

        .sf-card-title {
            font-size: 1.25rem;
            font-weight: 700;
            color: #667eea;
            margin-bottom: 0.8rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        /* Block containers */
        .info-block {
            background: rgba(102, 126, 234, 0.08);
            border-left: 4px solid #667eea;
            padding: 1.2rem;
            border-radius: 8px;
            margin: 1rem 0;
        }

        .success-block {
            background: rgba(34, 197, 94, 0.08);
            border-left: 4px solid #22c55e;
            padding: 1.2rem;
            border-radius: 8px;
            margin: 1rem 0;
        }

        .error-block {
            background: rgba(239, 68, 68, 0.08);
            border-left: 4px solid #ef4444;
            padding: 1.2rem;
            border-radius: 8px;
            margin: 1rem 0;
        }

        /* Navigation tabs styling */
        .nav-tabs {
            display: flex;
            gap: 0.5rem;
            margin-bottom: 2rem;
            border-bottom: 2px solid rgba(102, 126, 234, 0.2);
            overflow-x: auto;
        }

        .nav-tab {
            padding: 0.8rem 1.5rem;
            background: rgba(102, 126, 234, 0.1);
            border: 1px solid rgba(102, 126, 234, 0.3);
            border-radius: 8px 8px 0 0;
            cursor: pointer;
            transition: all 0.3s ease;
            font-weight: 600;
            white-space: nowrap;
        }

        .nav-tab:hover {
            background: rgba(102, 126, 234, 0.2);
            border-color: #667eea;
        }

        /* Metric styling - improved */
        .metric-container {
            background: linear-gradient(135deg, #1e1e3f 0%, #2d2d4a 100%);
            padding: 1.5rem;
            border-radius: 12px;
            border: 1px solid rgba(102, 126, 234, 0.3);
            text-align: center;
            transition: all 0.3s ease;
        }

        .metric-container:hover {
            transform: translateY(-3px);
            box-shadow: 0 15px 35px rgba(102, 126, 234, 0.15);
            border-color: #667eea;
        }

        .metric-label {
            font-size: 0.9rem;
            color: #9b9bbb;
            margin-bottom: 0.5rem;
            font-weight: 500;
        }

        .metric-value {
            font-size: 2.2rem;
            font-weight: 800;
            color: #667eea;
        }

        /* Button hover effects - enhanced */
        .stButton>button {
            border-radius: 10px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 0.7rem 2rem;
            font-weight: 700;
            box-shadow: 0 10px 25px rgba(102, 126, 234, 0.3);
            transition: transform 0.15s ease, box-shadow 0.15s ease, filter 0.15s ease;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            font-size: 0.9rem;
        }

        .stButton>button:hover {
            filter: brightness(1.1);
            transform: translateY(-2px);
            box-shadow: 0 15px 35px rgba(102, 126, 234, 0.4);
        }

        /* Tabs styling - improved */
        .stTabs [data-baseweb="tab"] {
            font-size: 1rem;
            font-weight: 700;
            color: #9b9bbb;
            border-radius: 10px 10px 0 0;
        }

        .stTabs [aria-selected="true"] {
            color: #667eea;
            border-color: #667eea !important;
        }

        /* Section titles */
        .section-title {
            font-size: 1.6rem;
            font-weight: 800;
            margin: 1.5rem 0 1rem 0;
            color: #e8e8f5;
            border-bottom: 2px solid rgba(102, 126, 234, 0.3);
            padding-bottom: 0.8rem;
        }

        .subsection-title {
            font-size: 1.2rem;
            font-weight: 700;
            color: #667eea;
            margin: 1.2rem 0 0.8rem 0;
        }

        /* Sidebar styling */
        .sidebar-section {
            background: rgba(102, 126, 234, 0.08);
            padding: 1rem;
            border-radius: 10px;
            margin-bottom: 1rem;
            border-left: 4px solid #667eea;
        }

        .status-badge {
            display: inline-block;
            padding: 0.4rem 1rem;
            border-radius: 20px;
            font-weight: 600;
            font-size: 0.85rem;
        }

        .status-success {
            background: linear-gradient(135deg, #34d399, #22c55e);
            color: white;
        }

        .status-error {
            background: linear-gradient(135deg, #f87171, #ef4444);
            color: white;
        }

        .status-warning {
            background: linear-gradient(135deg, #fbbf24, #f59e0b);
            color: white;
        }

        /* Timeline styling */
        .timeline-bar {
            border-radius: 8px;
            transition: all 0.3s ease;
        }

        /* Expander styling */
        .streamlit-expanderHeader {
            background: rgba(102, 126, 234, 0.1);
            border-radius: 10px;
            border: 1px solid rgba(102, 126, 234, 0.3);
        }

        .streamlit-expanderHeader:hover {
            background: rgba(102, 126, 234, 0.15);
            border-color: #667eea;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero-header">
        <h1 class="hero-title">🎙️ Speech Forensics Analysis</h1>
        <p class="hero-subtitle">
            Advanced audio authentication system | Detect real vs. fake speech | Identify tampered regions with precision
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------

# LOAD CONFIG

# ----------------------------------------------------

CONFIG_PATH = os.path.join(PROJECT_ROOT, "config.yaml")
with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

# ----------------------------------------------------

# LOAD MODEL

# ----------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
clf_model = model.MultiStreamForensics(config).to(device)

# Try several common checkpoint locations (train script may save under different names)
possible_checkpoints = [
    os.path.join(PROJECT_ROOT, "speech_forensics", "checkpoints", "audio_forensics.pt"),
    os.path.join(PROJECT_ROOT, "speech_forensics", "checkpoints", "checkpoint.pth"),
    os.path.join(PROJECT_ROOT, "checkpoints", "audio_forensics.pt"),
    os.path.join(PROJECT_ROOT, "checkpoints", "checkpoint.pth"),
    os.path.join(PROJECT_ROOT, "outputs", "checkpoints", "audio_forensics.pt"),
    os.path.join(PROJECT_ROOT, "outputs", "checkpoints", "checkpoint.pth"),
]

found_ckpt = None
for p in possible_checkpoints:
    if os.path.exists(p):
        found_ckpt = p
        break

if found_ckpt:
    try:
        checkpoint = torch.load(found_ckpt, map_location=device)
        state = checkpoint.get("model_state_dict", checkpoint)
        clf_model.load_state_dict(state)
        clf_model.eval()
        st.sidebar.markdown("### ✅ Model Status")
        st.sidebar.markdown(f"<div class='sidebar-section'><span class='status-badge status-success'>✓ Model Loaded</span><br><br>**File:** {os.path.basename(found_ckpt)}<br>**Device:** {str(device).upper()}<br>**Validation Accuracy:** {checkpoint.get('val_accuracy', 'N/A')}</div>", unsafe_allow_html=True)
    except Exception as e:
        st.sidebar.markdown("### ❌ Model Error")
        st.sidebar.markdown(f"<div class='sidebar-section'><span class='status-badge status-error'>✗ Load Failed</span><br><br>**Error:** {str(e)}</div>", unsafe_allow_html=True)
        st.stop()
else:
    st.sidebar.markdown("### ❌ Model Not Found")
    st.sidebar.markdown("<div class='sidebar-section'><span class='status-badge status-error'>✗ Missing Checkpoint</span><br><br>**Action Required:** Train the model first to generate a checkpoint.</div>", unsafe_allow_html=True)
    st.stop()


# ====== SECTION 1: FILE UPLOAD ======
st.markdown("<div class='section-title'>📤 Step 1: Upload & Configure</div>", unsafe_allow_html=True)

col_upload, col_controls = st.columns([2, 1])

with col_upload:
    st.markdown(
        """
        <div class='info-block'>
        <strong>📁 Upload your audio file</strong> in any of these formats: WAV, MP3, or FLAC
        </div>
        """,
        unsafe_allow_html=True,
    )
    uploaded_file = st.file_uploader("", type=["wav", "mp3", "flac"], label_visibility="collapsed")

with col_controls:
    st.markdown(
        """
        <div class='info-block'>
        <strong>⚙️ Analysis Settings</strong>
        </div>
        """,
        unsafe_allow_html=True,
    )
    conf_threshold = st.slider(
        "Confidence Threshold (%)",
        min_value=40,
        max_value=90,
        value=55,
        step=1,
        help="Only regions above this confidence will be marked as tampered.",
    )

if uploaded_file is None:
    st.info("👆 **Please upload an audio file to begin analysis**")
    st.stop()

# Audio player
st.markdown("<div class='subsection-title'>🔊 Audio Playback</div>", unsafe_allow_html=True)
st.audio(uploaded_file)

# Load audio
y, sr = librosa.load(uploaded_file, sr=16000)

# ====== FEATURE EXTRACTION ======
st.markdown("<div class='section-title'>🔍 Step 2: Feature Extraction</div>", unsafe_allow_html=True)

with st.spinner("🔄 Extracting audio features..."):
    lfcc_feat = features.lfcc(y, sr)
    pfe_feat = features.pitch_formants_energy(y, sr)
    ppg_feat = features.PPGExtractor().extract(y, sr)

lfcc_feat = np.nan_to_num(lfcc_feat)
pfe_feat = np.nan_to_num(pfe_feat)
ppg_feat = np.nan_to_num(ppg_feat)

st.markdown(
    """
    <div class='success-block'>
    ✅ <strong>Features extracted successfully!</strong> Analyzed acoustic patterns, pitch, formants, and phonetic features.
    </div>
    """,
    unsafe_allow_html=True,
)

# ====== GLOBAL PREDICTION ======
st.markdown("<div class='section-title'>🎯 Step 3: Analysis Results</div>", unsafe_allow_html=True)

with st.spinner("🔄 Running model inference..."):
    result = clf_model.predict(lfcc_feat, pfe_feat, ppg_feat, return_probs=True, device=device)

real_prob = result["real_prob"]
fake_prob = result["fake_prob"]
prediction = result["prediction"]

# Display prediction results in attractive cards
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(
        f"""
        <div class='metric-container'>
            <div class='metric-label'>🟢 Real Speech Probability</div>
            <div class='metric-value'>{real_prob:.1f}%</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col2:
    st.markdown(
        f"""
        <div class='metric-container'>
            <div class='metric-label'>🔴 Fake Speech Probability</div>
            <div class='metric-value'>{fake_prob:.1f}%</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col3:
    confidence = max(real_prob, fake_prob)
    st.markdown(
        f"""
        <div class='metric-container'>
            <div class='metric-label'>📊 Prediction Confidence</div>
            <div class='metric-value'>{confidence:.1f}%</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Main verdict
st.markdown("<hr style='border: 1px solid rgba(102, 126, 234, 0.3); margin: 1.5rem 0;'>", unsafe_allow_html=True)

if prediction == 0:
    st.markdown(
        """
        <div class='success-block' style='padding: 2rem; text-align: center; font-size: 1.2rem;'>
        <span style='font-size: 2.5rem;'>✅</span><br>
        <strong>AUTHENTIC SPEECH DETECTED</strong><br>
        This audio appears to be genuine without signs of tampering.
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        f"""
        <div class='error-block' style='padding: 2rem; text-align: center; font-size: 1.2rem;'>
        <span style='font-size: 2.5rem;'>⚠️</span><br>
        <strong>SUSPICIOUS AUDIO DETECTED</strong><br>
        This audio shows signs of manipulation or synthetic generation ({fake_prob:.1f}% probability).
        </div>
        """,
        unsafe_allow_html=True,
    )

# ====== TEMPORAL LOCALIZATION ======
st.markdown("<div class='section-title'>📍 Step 4: Tampered Region Localization</div>", unsafe_allow_html=True)

st.markdown(
    """
    <div class='info-block'>
    <strong>🔍 Analyzing the audio timeline</strong> to identify specific regions that may contain tampering or synthetic speech.
    </div>
    """,
    unsafe_allow_html=True,
)

with st.spinner("🔄 Analyzing timeline..."):
    regions = clf_model.predict_temporal(
        lfcc_feat,
        pfe_feat,
        ppg_feat,
        segment_duration=1.0,
        hop_duration=0.5,
        device=device
    )

tampered = [r for r in regions if r["prediction"] == 1 and r["confidence"] > conf_threshold]

if len(tampered) == 0:
    st.markdown(
        """
        <div class='success-block'>
        ✅ <strong>No tampered regions detected</strong> | All analyzed segments appear to be authentic audio.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("### 📊 Timeline Overview")
    st.info("View the complete timeline below showing all analyzed segments!")
else:
    st.markdown(
        f"""
        <div class='error-block'>
        ⚠️ <strong>{len(tampered)} Tampered Region(s) Detected</strong> | Review the details below for analysis.
        </div>
        """,
        unsafe_allow_html=True,
    )

# ====== VISUAL TIMELINE ======
st.markdown("<div class='subsection-title'>📊 Complete Audio Timeline</div>", unsafe_allow_html=True)

import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(14, 2.5))

for r in regions:
    start = r["start_time"]
    end = r["end_time"]
    conf = r["confidence"]

    if r["prediction"] == 1 and conf > conf_threshold:
        color = "#ef4444"
        label = "Tampered"
    else:
        color = "#22c55e"
        label = "Authentic"

    ax.barh(0, end - start, left=start, color=color, alpha=0.8, edgecolor="white", linewidth=1)

# Style the timeline
ax.set_xlabel("Time (seconds)", fontsize=11, fontweight="bold")
ax.set_yticks([])
ax.set_ylim(-0.5, 0.5)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.set_facecolor((0.118, 0.118, 0.247, 0.5))
fig.patch.set_facecolor((0.059, 0.059, 0.118, 0))

# Custom legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor="#22c55e", edgecolor="white", label="🟢 Authentic Regions"),
    Patch(facecolor="#ef4444", edgecolor="white", label="🔴 Tampered Regions")
]
ax.legend(handles=legend_elements, loc="upper right", framealpha=0.9)

st.pyplot(fig, use_container_width=True)

if len(tampered) > 0:
    # ====== DETECTED TAMPERED SEGMENTS ======
    st.markdown("<div class='subsection-title'>🔍 Detected Tampered Segments</div>", unsafe_allow_html=True)
    
    # Text list of tampered regions
    for idx, r in enumerate(tampered, start=1):
        start = float(r["start_time"])
        end = float(r["end_time"])
        conf = float(r["confidence"])
        
        st.markdown(
            f"""
            <div class='sf-card'>
            <span style='font-size: 1.2rem;'>🔴</span> <strong>Suspicious Region {idx}</strong><br>
            <strong>Time Range:</strong> {start:.2f}s → {end:.2f}s<br>
            <strong>Duration:</strong> {end-start:.2f}s<br>
            <strong>Confidence:</strong> <span style='color: #ef4444; font-weight: bold;'>{conf:.1f}%</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

if len(tampered) > 0:
    # ====== INTERACTIVE SPECTROGRAMS & DETAILED ANALYSIS ======
    st.markdown("<div class='section-title'>🔬 Step 5: Detailed Region Analysis</div>", unsafe_allow_html=True)
    
    st.markdown(
        """
        <div class='info-block'>
        <strong>📊 Inspect suspicious segments</strong> with zoomed spectrograms and focused timeline analysis.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Sort by start time for a clean ordering
    tampered_sorted = sorted(tampered, key=lambda r: r["start_time"])

    # Build labels for selection
    region_labels = [
        f"Region {i}: {r['start_time']:.2f}s – {r['end_time']:.2f}s (Conf: {r['confidence']:.1f}%)"
        for i, r in enumerate(tampered_sorted, start=1)
    ]

    st.markdown("<div class='subsection-title'>🎯 Choose a Region</div>", unsafe_allow_html=True)
    selected_label = st.selectbox(
        "Select a tampered region to view detailed analysis:",
        region_labels,
        help="View spectrogram and timeline for each flagged region"
    )

    # Map selection back to region
    selected_index = region_labels.index(selected_label)
    selected_region = tampered_sorted[selected_index]

    sel_start = float(selected_region["start_time"])
    sel_end = float(selected_region["end_time"])
    sel_conf = float(selected_region["confidence"])

    st.markdown(
        f"""
        <div class='sf-card'>
        <strong>📍 Region Analysis: {selected_index + 1}</strong><br>
        Time: {sel_start:.2f}s – {sel_end:.2f}s | Duration: {sel_end-sel_start:.2f}s | Confidence: {sel_conf:.1f}%
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Spectrogram focused on the selected region
    st.markdown("<div class='subsection-title'>🌊 Spectrogram Visualization</div>", unsafe_allow_html=True)
    fig_reg = plot_tampered_region_spectrogram(y, sr, sel_start, sel_end)
    st.pyplot(fig_reg, use_container_width=True)

    # Focused timeline highlighting only the selected region
    st.markdown("<div class='subsection-title'>📍 Position in Full Audio</div>", unsafe_allow_html=True)
    total_duration = len(y) / sr
    fig_sel, ax_sel = plt.subplots(figsize=(14, 2))

    # Background bar for full audio
    ax_sel.barh(
        0,
        total_duration,
        left=0.0,
        color="#646464",
        alpha=0.4,
        label="Full Audio",
    )

    # Highlight real regions before tampered
    ax_sel.barh(
        0,
        sel_start,
        left=0.0,
        color="#22c55e",
        alpha=0.7,
        linewidth=1,
        edgecolor="white",
        label="Authentic Audio",
    )
    
    # Highlight tampered region
    ax_sel.barh(
        0,
        sel_end - sel_start,
        left=sel_start,
        color="#ef4444",
        alpha=0.9,
        linewidth=2,
        edgecolor="white",
        label="Selected Tampered Region",
    )
    
    # Highlight real regions after tampered
    if total_duration > sel_end:
        ax_sel.barh(
            0,
            total_duration - sel_end,
            left=sel_end,
            color="#22c55e",
            alpha=0.7,
            linewidth=1,
            edgecolor="white",
        )

    ax_sel.set_xlim(0, total_duration)
    ax_sel.set_xlabel("Time (seconds)", fontweight="bold")
    ax_sel.set_yticks([])
    ax_sel.spines['top'].set_visible(False)
    ax_sel.spines['right'].set_visible(False)
    ax_sel.spines['left'].set_visible(False)
    ax_sel.set_facecolor((0.118, 0.118, 0.247, 0.5))
    
    handles, labels = ax_sel.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax_sel.legend(by_label.values(), by_label.keys(), loc="upper right", framealpha=0.9)

    st.pyplot(fig_sel, use_container_width=True)

# ====== ACOUSTIC–PHONETIC FEATURE EXPLORER ======
st.markdown("<div class='section-title'>🎵 Step 6: Acoustic & Phonetic Features</div>", unsafe_allow_html=True)

st.markdown(
    """
    <div class='info-block'>
    <strong>🔍 Deep dive into acoustic features</strong> that help identify synthetic or manipulated speech.
    </div>
    """,
    unsafe_allow_html=True,
)

# Create tabs for different feature views
tab1, tab2 = st.tabs(["📊 Full Recording Analysis", "🔎 Detailed Feature Breakdown"])

with tab1:
    st.markdown("<div class='subsection-title'>Complete Audio - All Features</div>", unsafe_allow_html=True)
    with st.spinner("🔄 Generating feature visualizations..."):
        fig_all = plot_all_features(y, sr, lfcc_feat, pfe_feat, ppg_feat)
        st.pyplot(fig_all, use_container_width=True)
    
    st.markdown(
        """
        <div class='info-block'>
        This visualization shows three types of features extracted from the entire audio:<br>
        • <strong>LFCC:</strong> Linear Frequency Cepstral Coefficients<br>
        • <strong>PFE:</strong> Pitch, Formants & Energy<br>
        • <strong>PPG:</strong> Phonetic Posteriorgram features
        </div>
        """,
        unsafe_allow_html=True,
    )

with tab2:
    st.markdown("<div class='subsection-title'>Feature Statistics</div>", unsafe_allow_html=True)
    
    # Display feature statistics in columns
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "📈 LFCC Frames",
            f"{lfcc_feat.shape[0]}",
            help="Number of time frames in LFCC features",
        )
    
    with col2:
        st.metric(
            "🎵 Coefficients",
            f"{lfcc_feat.shape[1]}",
            help="Dimensions of each feature vector",
        )
    
    with col3:
        duration = len(y) / sr
        st.metric(
            "⏱️ Duration",
            f"{duration:.2f}s",
            help="Total audio duration",
        )

if len(tampered) > 0:
    with st.expander("🔍 Tampered Region Feature Details", expanded=False):
        st.markdown("<div class='subsection-title'>Feature Analysis for Suspicious Regions</div>", unsafe_allow_html=True)
        
        # Reuse region labels & sorted list from localization section
        tampered_sorted = sorted(tampered, key=lambda r: r["start_time"])
        region_labels_feat = [
            f"Region {i}: {r['start_time']:.2f}s – {r['end_time']:.2f}s (Conf: {r['confidence']:.1f}%)"
            for i, r in enumerate(tampered_sorted, start=1)
        ]

        sel_region_label = st.selectbox(
            "Select a tampered region to view its acoustic profile:",
            region_labels_feat,
            key="feature_region_selector",
        )

        region_idx_feat = region_labels_feat.index(sel_region_label)
        region_feat = tampered_sorted[region_idx_feat]

        r_start = float(region_feat["start_time"])
        r_end = float(region_feat["end_time"])
        r_conf = float(region_feat["confidence"])

        st.markdown(
            f"""
            <div class='sf-card'>
            <strong>🎯 Analyzing Region {region_idx_feat + 1}</strong><br>
            Time: {r_start:.2f}s – {r_end:.2f}s | Duration: {r_end-r_start:.2f}s | Confidence: {r_conf:.1f}%
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Map time window to frame indices for LFCC/PFE/PPG (hop_length = 160 @ 16 kHz → 10 ms per frame)
        hop_length = 160
        start_frame = int(r_start * sr / hop_length)
        end_frame = int(r_end * sr / hop_length)

        lfcc_seg = lfcc_feat[start_frame:end_frame]
        pfe_seg = pfe_feat[start_frame:end_frame]
        ppg_seg = ppg_feat[start_frame:end_frame]
        y_seg = y[int(r_start * sr): int(r_end * sr)]

        if lfcc_seg.shape[0] < 5:
            st.warning("⚠️ Selected region is too short for accurate feature visualization.")
        else:
            with st.spinner("🔄 Generating region features..."):
                fig_region = plot_all_features(y_seg, sr, lfcc_seg, pfe_seg, ppg_seg)
                st.pyplot(fig_region, use_container_width=True)

# ====== FOOTER & INFORMATION ======
st.markdown("<hr style='border: 1px solid rgba(102, 126, 234, 0.3); margin: 2rem 0;'>", unsafe_allow_html=True)

st.markdown(
    """
    <div class='sf-card'>
    <h3 style='color: #667eea; margin-top: 0;'>📚 About This System</h3>
    <p><strong>Speech Forensics Dashboard</strong> uses advanced machine learning to detect synthetic, manipulated, or tampered speech.</p>
    
    <strong>🔬 Analysis Methods:</strong><br>
    • <strong>LFCC Features:</strong> Linear Frequency Cepstral Coefficients capture spectral characteristics<br>
    • <strong>Pitch & Energy:</strong> Identify unnatural pitch variations and energy patterns<br>
    • <strong>Phonetic Analysis:</strong> PPG features reveal phonetic inconsistencies<br>
    • <strong>Temporal Localization:</strong> Pinpoint exact regions of potential manipulation<br>
    
    <strong>⚠️ Important Notes:</strong><br>
    • This tool is for forensic analysis and educational purposes<br>
    • Always verify results with expert human review for critical applications<br>
    • Confidence scores indicate model certainty, not absolute truth<br>
    
    <strong>📖 Model Information:</strong> MultiStream Forensics Network trained on diverse audio datasets
    </div>
    """,
    unsafe_allow_html=True,
)

# Footer
st.markdown(
    """
    <div style='text-align: center; padding: 2rem 0; color: #7a7a9e; font-size: 0.9rem; border-top: 1px solid rgba(102, 126, 234, 0.2); margin-top: 2rem;'>
    <p>🎙️ <strong>Speech Forensics Analysis System</strong> | Advanced Audio Authentication<br>
    Built with Streamlit, PyTorch, and Librosa | © 2025 Audio Forensics Research</p>
    </div>
    """,
    unsafe_allow_html=True,
)