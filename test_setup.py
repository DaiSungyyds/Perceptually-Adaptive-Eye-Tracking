"""
Quick Test Script

Tests all core components without requiring full data download
"""

import sys
import os

# Test 1: Import check
print("="*60)
print("Test 1: Checking imports...")
print("="*60)

try:
    import torch
    print(f"✓ PyTorch: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA device: {torch.cuda.get_device_name(0)}")
except ImportError as e:
    print(f"✗ PyTorch not installed: {e}")
    print("  Run: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
    sys.exit(1)

try:
    import timm
    print(f"✓ timm: {timm.__version__}")
except ImportError:
    print("✗ timm not installed")
    print("  Run: pip install timm")
    sys.exit(1)

try:
    import cv2
    print(f"✓ opencv-python: {cv2.__version__}")
except ImportError:
    print("⚠ opencv-python not installed (optional for now)")

print()

# Test 2: Model creation
print("="*60)
print("Test 2: Creating Multi-Exit ViT model...")
print("="*60)

sys.path.append('src')

try:
    from models.multi_exit_vit import MultiExitViT

    model = MultiExitViT(
        model_name='vit_small_patch16_224',
        pretrained=False,  # Don't download weights for quick test
        exit_points=[3, 6, 12]
    )

    print("✓ Model created successfully")
    print(f"  Total parameters: {model.get_num_params():,}")

except Exception as e:
    print(f"✗ Model creation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()

# Test 3: Forward pass
print("="*60)
print("Test 3: Testing forward pass...")
print("="*60)

try:
    model.eval()

    # Create dummy input
    batch_size = 2
    dummy_input = torch.randn(batch_size, 1, 224, 224)

    # Test single-exit inference
    with torch.no_grad():
        for state in ['saccade', 'pursuit', 'fixation']:
            output = model(dummy_input, state=state)
            assert output.shape == (batch_size, 2), f"Wrong output shape: {output.shape}"
            print(f"✓ {state:10s} -> Exit {model.get_exit_for_state(state)+1}: {output.shape}")

    # Test multi-exit training forward
    with torch.no_grad():
        outputs = model.forward_all_exits(dummy_input)
        assert len(outputs) == 3, f"Wrong number of exits: {len(outputs)}"
        print(f"✓ Multi-exit forward: {len(outputs)} exits")

except Exception as e:
    print(f"✗ Forward pass failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()

# Test 4: Loss functions
print("="*60)
print("Test 4: Testing loss functions...")
print("="*60)

try:
    from train.losses import MultiExitLoss, angular_loss

    # Test angular loss
    pred = torch.randn(batch_size, 2)
    target = torch.randn(batch_size, 2)

    loss = angular_loss(pred, target)
    print(f"✓ Angular loss: {loss.item():.4f} degrees")

    # Test multi-exit loss
    criterion = MultiExitLoss()
    predictions = {
        'exit_1': torch.randn(batch_size, 2),
        'exit_2': torch.randn(batch_size, 2),
        'exit_3': torch.randn(batch_size, 2)
    }
    states = ['fixation', 'saccade']

    loss, loss_dict = criterion(predictions, target, states)
    print(f"✓ Multi-exit loss: {loss.item():.4f}")
    for key, val in loss_dict.items():
        print(f"    {key}: {val:.4f}")

except Exception as e:
    print(f"✗ Loss function test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()

# Test 5: Check data availability
print("="*60)
print("Test 5: Checking data availability...")
print("="*60)

data_root = 'eye_data'
if os.path.exists(data_root):
    users = [d for d in os.listdir(data_root) if d.startswith('user') and os.path.isdir(os.path.join(data_root, d))]
    print(f"✓ Data directory found: {data_root}")
    print(f"  Available users: {len(users)}")
    print(f"  Users: {users[:5]}{'...' if len(users) > 5 else ''}")

    if len(users) >= 2:
        print(f"✓ Sufficient data for training ({len(users)} users)")
    else:
        print(f"⚠ Only {len(users)} users available (need at least 2)")
else:
    print(f"⚠ Data directory not found: {data_root}")
    print("  Data is still downloading or not yet extracted")

print()

# Summary
print("="*60)
print("SUMMARY")
print("="*60)
print("✓ All core components working!")
print()
print("Next steps:")
print("1. Wait for data download to complete")
print("2. Test with mock data: python src/data/mock_dataset.py")
print("3. Start training: python src/train/train_multi_exit_vit.py")
print()
print("See README_MODEL.md for full documentation")
print("="*60)
