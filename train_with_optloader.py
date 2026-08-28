"""
使用师兄的opt_dataloader训练 - GPU 3
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import sys
from tqdm import tqdm

# 设置GPU 0
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

sys.path.append('src')
sys.path.append('new_dataset/sel_valid_label_0815')

from models.rvt_gaze import RVT_Gaze
from cache_dataset import LoadFromCache


def pixel_loss(pred, target, screen_size=(1920, 1080)):
    """Pixel loss (欧氏距离)"""
    screen_width, screen_height = screen_size

    pred_x = (pred[:, 0] + 1) * (screen_width / 2)
    pred_y = (pred[:, 1] + 1) * (screen_height / 2)

    target_x = (target[:, 0] + 1) * (screen_width / 2)
    target_y = (target[:, 1] + 1) * (screen_height / 2)

    diff_x = pred_x - target_x
    diff_y = pred_y - target_y
    pixel_error = torch.sqrt(diff_x**2 + diff_y**2)

    return pixel_error.mean()


def normalize_gaze(gaze, screen_size=(1080, 1920)):
    """归一化gaze坐标到[-1, 1]"""
    screen_height, screen_width = screen_size

    # gaze shape: [1, 2] where [y, x]
    gaze_y = gaze[0, 0].item()
    gaze_x = gaze[0, 1].item()

    gaze_x = max(0, min(gaze_x, screen_width))
    gaze_y = max(0, min(gaze_y, screen_height))

    gaze_x_norm = (gaze_x / screen_width) * 2 - 1
    gaze_y_norm = (gaze_y / screen_height) * 2 - 1

    return torch.tensor([gaze_x_norm, gaze_y_norm], dtype=torch.float32)


def events_to_polarity_image(events, img_size=(260, 346)):
    """
    向量化版本 - 将events转换为polarity图像
    使用scatter_保证后面的event覆盖前面的event
    """
    height, width = img_size
    polarity_img = torch.zeros((height, width), dtype=torch.float32)

    if len(events) == 0:
        return polarity_img.unsqueeze(0)

    # 提取坐标和极性
    x = events[:, 1].long()
    y = events[:, 2].long()
    p = events[:, 3]

    # 边界检查
    mask = (x >= 0) & (x < width) & (y >= 0) & (y < height)
    x_valid = x[mask]
    y_valid = y[mask]
    p_valid = p[mask]

    if len(x_valid) == 0:
        return polarity_img.unsqueeze(0)

    # 转换极性值
    polarity_values = torch.where(p_valid == 1.0,
                                  torch.ones_like(p_valid),
                                  -torch.ones_like(p_valid))

    # 使用scatter_: PyTorch按顺序处理，后面的自动覆盖前面的
    flat_indices = y_valid * width + x_valid
    flat_img = polarity_img.view(-1)
    flat_img.scatter_(0, flat_indices, polarity_values)

    return flat_img.view(height, width).unsqueeze(0)


class CacheDatasetWithTransform(LoadFromCache):
    """师兄的LoadFromCache + 转换为模型输入格式"""

    def __init__(self, cache_dir, target_img_size=(160, 176)):
        super().__init__(cache_dir)
        self.target_img_size = target_img_size

    def __getitem__(self, index):
        # 从pkl加载 (events, gaze, timestamp)
        events, gaze_label, timestamp = super().__getitem__(index)

        # Events → Polarity Image
        polarity_img = events_to_polarity_image(events, img_size=(260, 346))

        # Resize to target size
        if polarity_img.shape[1:] != self.target_img_size:
            import torch.nn.functional as F
            polarity_img = polarity_img.unsqueeze(0)
            polarity_img = F.interpolate(
                polarity_img, size=self.target_img_size,
                mode='bilinear', align_corners=False
            )
            polarity_img = polarity_img.squeeze(0)

        # Normalize gaze
        gaze_norm = normalize_gaze(gaze_label, screen_size=(1080, 1920))

        return {
            'image': polarity_img,
            'gaze': gaze_norm,
            'timestamp': timestamp
        }


def main():
    print("="*70)
    print("          Training with 师兄's opt_dataloader - GPU 3")
    print("="*70)

    device = torch.device('cuda:0')
    print(f"\nDevice: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # 配置
    CHECKPOINT_PATH = '/data/wh_srt/gaze_project/best_model_gpu1.pth'
    TRAIN_CACHE = '/home/jzj24/Eye_Tracking/dataset_cache/EVB_Eye_New/LREye_train_cache_drop15_8000acc'
    VAL_CACHE = '/home/jzj24/Eye_Tracking/dataset_cache/EVB_Eye_New/LREye_val_cache_drop15_8000acc'
    BATCH_SIZE = 256
    EPOCHS = 100
    LR = 1e-4

    print(f"\nConfiguration:")
    print(f"  Batch Size: {BATCH_SIZE}")
    print(f"  Epochs: {EPOCHS}")

    # 加载数据（使用师兄的LoadFromCache）
    print(f"\nLoading data with LoadFromCache...")
    train_dataset = CacheDatasetWithTransform(TRAIN_CACHE)
    val_dataset = CacheDatasetWithTransform(VAL_CACHE)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        prefetch_factor=4,
        persistent_workers=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(f"Train batches: {len(train_loader)}")

    # 创建模型
    print(f"\nLoading model...")
    model = RVT_Gaze(
        img_size=(160, 176),
        patch_size=16,
        in_chans=1,
        embed_dim=192,
        depth=3,
        num_heads=6,
        mlp_ratio=1.0,
        qkv_bias=True,
        drop_rate=0.0,
        drop_path_rate=0.1
    )

    if os.path.exists(CHECKPOINT_PATH):
        checkpoint = torch.load(CHECKPOINT_PATH, map_location='cpu', weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded checkpoint from Epoch {checkpoint['epoch']}")
        best_val_loss = float('inf')
        start_epoch = 0
    else:
        print("No checkpoint, starting from scratch")
        best_val_loss = float('inf')
        start_epoch = 0

    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    print(f"\n{'='*70}")
    print(f"Starting Training")
    print(f"{'='*70}")

    for epoch in range(start_epoch, EPOCHS):
        # 训练
        model.train()
        train_loss = 0.0
        train_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{EPOCHS} [Train]')

        for batch in train_bar:
            images = batch['image'].to(device)
            labels = batch['gaze'].to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = pixel_loss(outputs, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            train_bar.set_postfix(loss=f'{loss.item():.2f}px')

        train_loss /= len(train_loader)

        # 验证
        model.eval()
        val_loss = 0.0
        val_bar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{EPOCHS} [Val]')

        with torch.no_grad():
            for batch in val_bar:
                images = batch['image'].to(device)
                labels = batch['gaze'].to(device)

                outputs = model(images)
                loss = pixel_loss(outputs, labels)

                val_loss += loss.item()
                val_bar.set_postfix(loss=f'{loss.item():.2f}px')

        val_loss /= len(val_loader)
        scheduler.step()

        print(f"\nEpoch {epoch+1}/{EPOCHS}:")
        print(f"  Train Loss: {train_loss:.2f} pixels")
        print(f"  Val Loss: {val_loss:.2f} pixels")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = '/data/wh_srt/gaze_project/best_model_gpu3_optloader.pth'
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'train_loss': train_loss,
            }, save_path)
            print(f"  New best model saved! Val Loss: {val_loss:.2f} pixels")

        # 定期保存
        if (epoch + 1) % 10 == 0:
            checkpoint_path = f'/data/wh_srt/gaze_project/checkpoint_epoch_{epoch+1}_optloader.pth'
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'train_loss': train_loss,
            }, checkpoint_path)
            print(f"  Checkpoint saved")

    print(f"\n{'='*70}")
    print(f"Training Complete!")
    print(f"Best Val Loss: {best_val_loss:.2f} pixels")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
