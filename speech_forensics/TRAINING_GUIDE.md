# Speech Forensics Model Training Guide

This guide explains how to train the speech forensics model for accurate real/fake speech detection.

## Prerequisites

1. **Prepare Your Dataset**
   
   Organize your audio files in the following structure:
   ```
   data/processed/
     ├── real/          # Real/genuine speech audio files
     │   ├── real_001.wav
     │   ├── real_002.wav
     │   └── ...
     └── fake/          # Fake/spoofed speech audio files
         ├── fake_001.wav
         ├── fake_002_tts.wav      # TTS samples
         ├── fake_003_vc.wav       # Voice conversion samples
         ├── fake_004_replay.wav   # Replay attack samples
         └── ...
   ```

   **Note**: The model automatically detects attack types from filenames:
   - `tts` → Text-to-Speech
   - `vc` or `voice_conversion` → Voice Conversion
   - `replay` → Replay Attack
   - `gan` or `deepfake` → GAN-based Deepfake
   - `splice` or `edit` → Splicing/Editing
   - Otherwise defaults to TTS

2. **Dataset Requirements**
   - Audio format: WAV, FLAC, or MP3
   - Sample rate: Will be resampled to 16kHz automatically
   - Minimum duration: At least 0.5 seconds
   - Balanced dataset: Try to have similar number of real and fake samples

## Training the Model

### Step 1: Install Dependencies

Ensure you have all required packages:
```bash
pip install torch torchvision torchaudio
pip install librosa parselmouth gammatone transformers scikit-learn
```

### Step 2: Configure Training Parameters

Edit `config.yaml` to adjust training parameters:

```yaml
training:
  batch_size: 16        # Increase if you have GPU memory
  lr: 0.0001           # Learning rate
  epochs: 10           # Number of training epochs
  weight_auth: 1.0     # Weight for authentication loss (real/fake)
  weight_attack: 0.7   # Weight for attack type classification loss

model:
  hidden_size: 256     # LSTM hidden size
  lstm_layers: 2       # Number of LSTM layers
  dropout: 0.2         # Dropout rate
```

### Step 3: Run Training

From the project root directory:

```bash
# Using Python module
python -m speech_forensics.src.train

# Or if you're in the speech_forensics directory
python -m src.train
```

### Step 4: Monitor Training

The training script will display:
- Training loss and accuracy per batch
- Validation metrics per epoch:
  - Authentication accuracy (real vs fake)
  - Precision, Recall, F1-score
  - Attack type classification accuracy
  - Confusion matrix

The best model (based on validation accuracy) will be saved automatically to:
```
outputs/checkpoints/checkpoint.pth
```

## Training Tips

1. **Dataset Size**: For good results, aim for at least:
   - 500+ real samples
   - 500+ fake samples (across different attack types)

2. **Balanced Classes**: Ensure your dataset has balanced real/fake samples

3. **Epochs**: Start with 10 epochs, increase if validation loss is still decreasing

4. **Learning Rate**: If loss isn't decreasing, try:
   - Lower learning rate (e.g., 0.00005)
   - The script includes automatic learning rate reduction

5. **Batch Size**: 
   - GPU: Use 16-32
   - CPU: Use 8-16

6. **Overfitting**: If validation accuracy plateaus or decreases:
   - Increase dropout (e.g., 0.3)
   - Use data augmentation
   - Get more training data

## Troubleshooting

### "No training samples found"
- Check that `data/processed/real/` and `data/processed/fake/` directories exist
- Ensure audio files have valid extensions (.wav, .flac, .mp3)

### "CUDA out of memory"
- Reduce `batch_size` in config.yaml
- Use CPU instead: The script automatically detects GPU/CPU

### Low Accuracy
- Ensure dataset has good quality audio
- Check for balanced real/fake samples
- Try training for more epochs
- Verify audio files aren't corrupted

### Slow Training
- Enable GPU if available
- Reduce batch size
- The first epoch is slower (feature extraction cache)

## Model Evaluation

After training, the model is saved and can be used for inference:

```bash
python infer.py --audio path/to/audio.wav --checkpoint outputs/checkpoints/checkpoint.pth
```

Or use the dashboard:
```bash
streamlit run src/app/dashboard.py
```

## Next Steps

1. **Fine-tuning**: After initial training, fine-tune on your specific domain data
2. **Data Augmentation**: Add noise, speed variation, or pitch shifting
3. **Ensemble**: Train multiple models and average predictions
4. **Transfer Learning**: Start with pre-trained weights if available

## Support

For issues or questions, check:
- Model architecture: `src/model.py`
- Feature extraction: `src/features.py`
- Dataset loading: `src/dataset.py`
