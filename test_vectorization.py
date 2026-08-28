"""
向量化版本的events_to_polarity_image + 严格验证 + Benchmark
"""

import torch
import numpy as np
import time
import pickle


def events_to_polarity_image_old(events, img_size=(260, 346)):
    """原始版本 - Python循环（保证后面的event覆盖前面的）"""
    height, width = img_size
    polarity_img = torch.zeros((height, width), dtype=torch.float32)

    if len(events) == 0:
        return polarity_img.unsqueeze(0)

    # Python循环 - 后面的覆盖前面的
    for event in events:
        t, x, y, p = event
        x, y = int(x.item()), int(y.item())
        if 0 <= x < width and 0 <= y < height:
            polarity_img[y, x] = 1.0 if p.item() == 1.0 else -1.0

    return polarity_img.unsqueeze(0)


def events_to_polarity_image_vectorized(events, img_size=(260, 346)):
    """
    向量化版本 - 保证语义一致（后面的event覆盖前面的）

    使用scatter_方法，关键是确保后面的event覆盖前面的
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

    # 关键：使用scatter_，PyTorch按顺序处理，后面的自动覆盖前面的
    # 将2D坐标展平为1D索引
    flat_indices = y_valid * width + x_valid

    # 展平图像并scatter
    flat_img = polarity_img.view(-1)
    flat_img.scatter_(0, flat_indices, polarity_values)

    return flat_img.view(height, width).unsqueeze(0)


def test_correctness(num_samples=100):
    """测试向量化版本和原始版本输出是否完全一致"""
    print("="*70)
    print("测试正确性：向量化版本 vs 原始版本")
    print("="*70)

    # 生成随机测试数据
    print(f"\n使用随机生成的 {num_samples} 个样本测试...")

    total_equal = 0
    total_diff_pixels = 0
    max_diff = 0

    for i in range(num_samples):
        # 生成随机events (t, x, y, p)
        num_events = 8000
        events = torch.zeros((num_events, 4), dtype=torch.float32)
        events[:, 0] = torch.rand(num_events) * 1000  # timestamp
        events[:, 1] = torch.randint(0, 346, (num_events,)).float()  # x
        events[:, 2] = torch.randint(0, 260, (num_events,)).float()  # y
        events[:, 3] = torch.randint(0, 2, (num_events,)).float()  # polarity

        # 两种方法生成图像
        old_img = events_to_polarity_image_old(events)
        new_img = events_to_polarity_image_vectorized(events)

        # 检查是否完全相等
        is_equal = torch.equal(old_img, new_img)
        diff_pixels = (old_img != new_img).sum().item()

        if is_equal:
            total_equal += 1
        else:
            total_diff_pixels += diff_pixels
            max_diff = max(max_diff, diff_pixels)
            if i < 5:  # 只打印前几个差异
                print(f"  样本 {i}: 不一致! 差异像素数: {diff_pixels}")
                print(f"    Old shape: {old_img.shape}, New shape: {new_img.shape}")
                print(f"    Old unique values: {torch.unique(old_img)}")
                print(f"    New unique values: {torch.unique(new_img)}")

    print(f"\n{'='*70}")
    print(f"结果:")
    print(f"  完全一致的样本: {total_equal}/{num_samples} ({total_equal/num_samples*100:.1f}%)")
    print(f"  不一致的样本: {num_samples-total_equal}/{num_samples}")
    if total_diff_pixels > 0 and num_samples > total_equal:
        print(f"  平均差异像素: {total_diff_pixels/(num_samples-total_equal):.1f}")
    print(f"  最大差异像素: {max_diff}")
    print(f"{'='*70}\n")

    return total_equal == num_samples


def benchmark(num_iterations=1000):
    """Benchmark: 比较两种方法的速度"""
    print("="*70)
    print("性能测试：Benchmark")
    print("="*70)

    # 生成随机测试数据
    print(f"\n使用随机样本进行 {num_iterations} 次迭代测试...")

    num_events = 8000
    events = torch.zeros((num_events, 4), dtype=torch.float32)
    events[:, 0] = torch.rand(num_events) * 1000
    events[:, 1] = torch.randint(0, 346, (num_events,)).float()
    events[:, 2] = torch.randint(0, 260, (num_events,)).float()
    events[:, 3] = torch.randint(0, 2, (num_events,)).float()

    print(f"\n使用随机样本:")
    print(f"  Events数量: {len(events)}")
    print(f"  Events shape: {events.shape}")

    # Warm up
    for _ in range(10):
        _ = events_to_polarity_image_old(events)
        _ = events_to_polarity_image_vectorized(events)

    # Benchmark 原始版本
    print(f"\n测试原始版本 (Python循环)...")
    t0 = time.perf_counter()
    for _ in range(num_iterations):
        _ = events_to_polarity_image_old(events)
    old_time = time.perf_counter() - t0
    old_per_sample = old_time / num_iterations * 1000  # ms

    # Benchmark 向量化版本
    print(f"测试向量化版本...")
    t0 = time.perf_counter()
    for _ in range(num_iterations):
        _ = events_to_polarity_image_vectorized(events)
    new_time = time.perf_counter() - t0
    new_per_sample = new_time / num_iterations * 1000  # ms

    speedup = old_time / new_time

    print(f"\n{'='*70}")
    print(f"结果 ({num_iterations} iterations):")
    print(f"  原始版本 (Python循环):")
    print(f"    总时间: {old_time:.2f}s")
    print(f"    每样本: {old_per_sample:.2f}ms")
    print(f"  向量化版本:")
    print(f"    总时间: {new_time:.2f}s")
    print(f"    每样本: {new_per_sample:.2f}ms")
    print(f"  加速比: {speedup:.1f}x")
    print(f"{'='*70}\n")

    # 估算训练加速
    samples_per_epoch = 251328
    old_epoch_time = old_per_sample * samples_per_epoch / 1000 / 60  # minutes
    new_epoch_time = new_per_sample * samples_per_epoch / 1000 / 60  # minutes

    print(f"预计训练加速:")
    print(f"  原始版本 - 单epoch数据处理时间: ~{old_epoch_time:.1f}分钟")
    print(f"  向量化版本 - 单epoch数据处理时间: ~{new_epoch_time:.1f}分钟")
    print(f"  节省时间: ~{old_epoch_time - new_epoch_time:.1f}分钟/epoch")
    print(f"{'='*70}\n")


def main():
    print("\n" + "="*70)
    print("  向量化 events_to_polarity_image 验证和性能测试")
    print("="*70 + "\n")

    # 1. 测试正确性
    all_correct = test_correctness(num_samples=100)

    if not all_correct:
        print("⚠️  警告: 向量化版本和原始版本有差异！")
        print("    建议检查实现逻辑")
        return
    else:
        print("✓ 正确性验证通过！向量化版本和原始版本完全一致！\n")

    # 2. Benchmark
    benchmark(num_iterations=1000)

    print("="*70)
    print("测试完成！")
    print("="*70)


if __name__ == '__main__':
    main()
