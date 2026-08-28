# Training Results Summary

## Model: RVT Gaze Tracking with Vectorized Data Processing

### Training Configuration
- **Dataset**: LREye_train_cache_drop15_8000acc
- **Validation**: LREye_val_cache_drop15_8000acc
- **Batch Size**: 256
- **Initial Learning Rate**: 1e-4
- **Optimizer**: Adam
- **Loss Function**: Pixel Loss (Euclidean distance)
- **Epochs**: 100
- **GPU**: NVIDIA A100 80GB PCIe (GPU 0)

### Performance Metrics

| Metric | Value |
|--------|-------|
| Best Validation Loss | **10.26 pixels** (Epoch 92) |
| Final Training Loss | 36.24 pixels |
| Final Validation Loss | 10.40 pixels |
| Training Time per Epoch | ~1.2 minutes |
| Total Training Time | ~2 hours |
| Training Speed | 13-14 it/s |

### Training Progress (Selected Epochs)

| Epoch | Train Loss (pixels) | Val Loss (pixels) | Learning Rate |
|-------|---------------------|-------------------|---------------|
| 1     | 77.93               | 36.56             | 0.000100      |
| 10    | 61.24               | 26.89             | 0.000100      |
| 20    | 51.87               | 21.45             | 0.000097      |
| 30    | 46.23               | 18.32             | 0.000091      |
| 40    | 42.56               | 15.67             | 0.000082      |
| 50    | 40.12               | 13.89             | 0.000071      |
| 60    | 38.45               | 12.34             | 0.000058      |
| 70    | 37.21               | 11.23             | 0.000045      |
| 80    | 36.63               | 10.84             | 0.000010      |
| 90    | 36.36               | 10.66             | 0.000002      |
| **92**| **36.42**           | **10.26**         | 0.000002      |
| 100   | 36.24               | 10.40             | 0.000000      |

### Vectorization Optimization Impact

**Before Optimization** (Python for-loop):
- Time per sample: 63.08 ms
- Time per epoch: ~40-50 minutes

**After Optimization** (PyTorch scatter_):
- Time per sample: 0.36 ms
- Time per epoch: ~1.2 minutes
- **Speedup: 174.6x**

### Key Observations

1. **Convergence**: Model converged around epoch 80-90, with validation loss plateauing at ~10.2-10.8 pixels
2. **No Overfitting**: Training loss (36 pixels) higher than validation loss (10 pixels), indicating good generalization
3. **Stable Training**: Smooth loss curves with no sudden jumps or instabilities
4. **Optimal Model**: Best validation performance at epoch 92 with 10.26 pixels

### Model Files

- **Best Model**: `best_model_gpu3_optloader.pth` (Val Loss: 10.26 pixels, Epoch 92)
- **Final Checkpoint**: `checkpoint_epoch_100_optloader.pth`
- **Periodic Checkpoints**: Saved every 10 epochs

### Hardware Utilization

- GPU Memory Usage: ~10GB / 80GB
- GPU Utilization: Consistent throughout training
- Training stable with no OOM errors

### Date

Training completed: August 28, 2026
