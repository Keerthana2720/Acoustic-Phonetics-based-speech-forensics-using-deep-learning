# Troubleshooting Guide - Model Always Predicting "Real"

## Problem
The model is always predicting "Real Speech" even when given fake/spoofed audio files.

## Root Causes & Solutions

### 1. **Model Not Trained (MOST COMMON)**
**Symptom**: Model always predicts "Real" regardless of input.

**Cause**: The model is using random/untrained weights.

**Solution**: 
1. Prepare your dataset:
   ```
   data/processed/
     ├── real/    (place real speech audio files here)
     └── fake/    (place fake/spoofed audio files here)
   ```

2. Train the model:
   ```bash
   python -m speech_forensics.src.train
   ```

3. The trained checkpoint will be saved to:
   ```
   outputs/checkpoints/checkpoint.pth
   ```

4. The dashboard will automatically load this checkpoint on next run.

**Verification**: Check the sidebar in the dashboard. It should show:
- ✅ "Loaded trained model (Val Acc: XX.XX%)" if checkpoint exists
- ⚠️ "No trained checkpoint found" if model is untrained

---

### 2. **Dataset Issues**

#### Not Enough Data
- **Requirement**: Minimum 100-200 samples per class (real/fake)
- **Recommendation**: 500+ samples per class for good results
- **Solution**: Collect more training data

#### Unbalanced Dataset
- **Problem**: Too many real samples vs fake samples (or vice versa)
- **Symptom**: Model biased towards the majority class
- **Solution**: Balance your dataset (equal number of real/fake)

#### Poor Quality Data
- **Problem**: Audio files are corrupted, too short, or poor quality
- **Solution**: 
  - Use audio files > 0.5 seconds
  - Ensure clear audio without heavy noise
  - Use consistent sample rates (16kHz recommended)

---

### 3. **Feature Extraction Issues**

#### Missing Dependencies
If you see errors during feature extraction:
```bash
pip install parselmouth  # For formant extraction
pip install gammatone    # For gammatone features
pip install transformers # For PPG features (Wav2Vec2)
```

#### PPG Model Download
On first run, Wav2Vec2 model will download (~300MB). This is normal.

---

### 4. **Model Architecture Issues**

#### Checkpoint Not Loading
**Symptoms**:
- Dashboard shows "No trained checkpoint found"
- Predictions are random/always same

**Solutions**:
- Verify checkpoint exists: `outputs/checkpoints/checkpoint.pth`
- Check file permissions
- Ensure checkpoint was saved correctly during training

#### Model Mismatch
**Problem**: Checkpoint from different model architecture
**Solution**: Re-train with current model architecture

---

### 5. **Prediction Logic (FIXED)**

The dashboard had a bug where it would show "Real" even when model predicted "Fake" if real_prob was slightly higher. This has been fixed in the latest version.

**Fixed Logic**:
- Now uses model's direct prediction (argmax)
- Only uses probability comparison as fallback

---

## Quick Diagnosis Steps

1. **Check if model is trained**:
   - Look at dashboard sidebar message
   - If "No trained checkpoint found" → Train the model

2. **Check prediction probabilities**:
   - Expand "Debug Info" in dashboard
   - If probabilities are close to 50/50 → Model likely untrained
   - If probabilities are extreme (99%+) but wrong → Dataset issue

3. **Check feature extraction**:
   - Look at Debug Info → Feature Shapes
   - Should show: LFCC=[frames, 20], PFE=[frames, 5], PPG=[frames, 32]
   - If shapes are wrong → Feature extraction issue

4. **Check training logs**:
   - When training, watch validation accuracy
   - Should increase over epochs (target: >70% accuracy)
   - If stuck at ~50% → Dataset or model architecture issue

---

## Training Best Practices

1. **Start with small dataset**:
   - 100-200 samples per class
   - Train for 5-10 epochs
   - Verify model learns (accuracy improves)

2. **Gradually increase data**:
   - Add more samples as needed
   - Monitor validation accuracy
   - Stop if overfitting (val accuracy decreases)

3. **Hyperparameter tuning**:
   - If low accuracy: Increase `lr` (0.0001 → 0.0005)
   - If overfitting: Increase `dropout` (0.2 → 0.3-0.5)
   - If slow training: Increase `batch_size` (16 → 32)

4. **Data quality**:
   - Use consistent audio formats (WAV, 16kHz)
   - Similar duration samples (3-10 seconds)
   - Clear audio without background noise

---

## Expected Results After Training

- **Validation Accuracy**: Should be >70% for good performance
- **Prediction Confidence**: Should be >60% for clear cases
- **Balance**: Should correctly identify both real and fake samples

---

## Still Having Issues?

1. Verify your dataset structure matches requirements
2. Check training logs for errors
3. Ensure all dependencies are installed
4. Try with a small test dataset first
5. Check that audio files can be loaded (use librosa.load() test)

---

## Common Error Messages

### "No training samples found"
- **Fix**: Create `data/processed/real/` and `data/processed/fake/` directories
- Add audio files to these directories

### "CUDA out of memory"
- **Fix**: Reduce `batch_size` in config.yaml (16 → 8 or 4)

### "Feature extraction failed"
- **Fix**: Install missing dependencies (parselmouth, gammatone, transformers)
- Check audio file format (should be WAV, FLAC, or MP3)

### "Model always predicts same class"
- **Fix**: This means model is untrained or dataset is unbalanced
- Train the model or balance your dataset
