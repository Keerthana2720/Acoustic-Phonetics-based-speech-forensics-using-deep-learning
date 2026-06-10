Acoustic-Phonetic Based Speech Forensics using Deep Learning
Overview

Acoustic-Phonetic Based Speech Forensics using Deep Learning is an intelligent forensic speech analysis system designed to identify speakers, verify speech authenticity, detect tampered audio segments, and generate forensic reports. The system leverages acoustic-phonetic feature extraction and Transformer-based deep learning models to provide accurate and reliable speech forensic analysis.

The project processes speech recordings through multiple stages including preprocessing, feature extraction, speaker embedding generation, classification, tampered speech detection, and forensic report generation. The developed framework can be used in digital forensics, cybersecurity, voice authentication, criminal investigations, and anti-spoofing applications.

Features
Speaker Identification and Verification
Acoustic-Phonetic Speech Analysis
Tampered Speech Detection
Deep Learning-Based Classification
Transformer-Based Speaker Embedding Generation
Audio Preprocessing and Noise Reduction
Interactive Streamlit Dashboard
Forensic Report Generation
Multi-Language Speech Dataset Support
Visualization of Speech Analysis Results
System Architecture

The system consists of the following modules:

Speech Data Acquisition
Audio Preprocessing
Acoustic-Phonetic Feature Extraction
Deep Learning Model Training
Speaker Identification
Tampered Speech Localization
Report Generation
Interactive Dashboard

Technologies Used:
Programming Languages
Python 3.10+
Deep Learning Frameworks
PyTorch
NumPy
Audio Processing
Librosa
SoundFile
Web Framework
Streamlit
Data Processing
Pandas
Scikit-learn
Visualization
Matplotlib
Seaborn

Acoustic-Phonetic Features
MFCC (Mel Frequency Cepstral Coefficients)
Spectrogram Features
Pitch and Energy Features
Temporal Speech Features
Acoustic-Phonetic Embeddings
Deep Learning Models
Transformer Networks – Context-aware speech representation learning.
Self-Attention Mechanism – Captures long-range dependencies in speech signals.
Feed Forward Neural Networks (FFN) – Used within Transformer architectures.
Speaker Embedding Networks – Generate speaker-specific representations for forensic analysis.
Dataset Technologies
Real Speech Dataset
Tampered/Fake Speech Dataset
Indian Languages Speech Dataset
Custom Labeled Forensic Audio Dataset

Methodology
Step 1: Audio Acquisition
Speech recordings are collected from multiple speakers and stored in the dataset repository.
Step 2: Audio Preprocessing
Noise Reduction
Silence Removal
Normalization
Resampling
Step 3: Feature Extraction
The system extracts acoustic-phonetic features including:

CQCC (Constant-Q Cepstral Coefficients)
Spectral Features
Pitch Features
Temporal Speech Features
Step 4: Deep Learning Analysis
A Transformer-based neural network learns speaker-specific characteristics and generates speaker embeddings.

Step 5: Tampered Speech Detection
The model analyzes speech patterns to identify manipulated, edited, or synthetic audio regions.

Step 6: Report Generation
The system generates a forensic report containing:
Speaker Prediction
Confidence Score
Tampered Region Analysis
Audio Statistics
Installation
Clone Repository
git clone https://github.com/Keerthana2720/Acoustic-Phonetics-based-speech-forensics-using-deep-learning.git
Navigate to Project
cd Acoustic-Phonetics-based-speech-forensics-using-deep-learning
Install Dependencies
pip install -r requirements.txt
Running the Project
Train Model
python speech_forensics/src/train.py
Test Model
python speech_forensics/test_model.py
Launch Dashboard
python -m streamlit run speech_forensics/src/app/dashboard.py
Applications
Digital Speech Forensics
Criminal Investigations
Speaker Verification
Voice Authentication
Cybersecurity
Audio Tampering Detection
Anti-Spoofing Systems
Deepfake Voice Detection
Performance

The developed system demonstrates:
High Speaker Identification Accuracy
Robust Performance under Noisy Conditions
Effective Tampered Speech Detection
Efficient Deep Learning-Based Analysis
Real-Time Dashboard Visualization
Future Enhancements
Real-Time Speaker Verification
Deepfake Voice Detection
Multilingual Speaker Recognition
Explainable AI (XAI) Integration
Cloud Deployment
Mobile Forensic Applications
Advanced Tampered Region Localization
Hybrid Transformer-CNN Architectures
Author
Keerthana M
Bachelor of Technology (B.Tech)

Project Title:
Acoustic-Phonetic Based Speech Forensics using Deep Learning

License
This project is developed for academic and research purposes. Use and modification are permitted with proper attribution.
