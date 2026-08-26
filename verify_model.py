"""
完整的模型验证脚本
验证Multi-Exit ViT的所有功能
"""

import sys
sys.path.append('src')

import torch
import numpy as np
from models.multi_exit_vit import MultiExitViT
from train.losses import MultiExitLoss, angular_loss, StateAwareLoss

print("=" * 70)
print("Multi-Exit ViT 完整验证")
print("=" * 70)
print()

# ============================================
# 测试1：模型创建
# ============================================
print("测试1：模型创建")
print("-" * 70)

try:
    model = MultiExitViT(
        model_name='vit_small_patch16_224',
        pretrained=False,  # 快速测试不下载权重
        exit_points=[3, 6, 12]
    )
    print("✓ 模型创建成功")
    print(f"  - Backbone: vit_small_patch16_224")
    print(f"  - Embed dim: {model.embed_dim}")
    print(f"  - Exit points: {model.exit_points}")
    print(f"  - Total params: {model.get_num_params():,}")

    # 验证Exit数量
    assert len(model.exit_heads) == 3, "Exit数量应该是3"
    print("✓ Exit数量正确: 3个")

    # 验证状态映射
    assert model.state_to_exit == {'saccade': 0, 'pursuit': 1, 'fixation': 2}
    print("✓ 状态映射正确")

except Exception as e:
    print(f"✗ 模型创建失败: {e}")
    sys.exit(1)

print()

# ============================================
# 测试2：单Exit推理模式
# ============================================
print("测试2：单Exit推理模式（Inference）")
print("-" * 70)

model.eval()
batch_size = 4
test_image = torch.randn(batch_size, 1, 224, 224)

try:
    with torch.no_grad():
        # 测试每个状态
        for state in ['saccade', 'pursuit', 'fixation']:
            gaze = model(test_image, state=state)

            # 验证输出shape
            assert gaze.shape == (batch_size, 2), f"{state}: shape错误"

            # 验证输出范围（应该在合理范围内）
            assert not torch.isnan(gaze).any(), f"{state}: 包含NaN"

            # 获取使用的Exit
            exit_idx = model.get_exit_for_state(state)
            exit_name = f"Exit {exit_idx + 1} (Block {model.exit_points[exit_idx]})"

            print(f"✓ {state:10s} -> {exit_name}")
            print(f"    输出shape: {gaze.shape}")
            print(f"    输出范围: [{gaze.min().item():.2f}, {gaze.max().item():.2f}]")

    print("✓ 单Exit推理模式工作正常")

except Exception as e:
    print(f"✗ 单Exit推理失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()

# ============================================
# 测试3：多Exit训练模式
# ============================================
print("测试3：多Exit训练模式（Training）")
print("-" * 70)

model.train()

try:
    outputs = model.forward_all_exits(test_image)

    # 验证返回的Exit数量
    assert len(outputs) == 3, f"应该返回3个Exit，实际{len(outputs)}"
    print(f"✓ 返回了3个Exit的输出")

    # 验证每个Exit的输出
    for exit_name, gaze in outputs.items():
        assert gaze.shape == (batch_size, 2), f"{exit_name}: shape错误"
        assert not torch.isnan(gaze).any(), f"{exit_name}: 包含NaN"
        print(f"  {exit_name}: {gaze.shape} ✓")

    print("✓ 多Exit训练模式工作正常")

except Exception as e:
    print(f"✗ 多Exit训练失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()

# ============================================
# 测试4：验证计算效率（不同Exit的参数量）
# ============================================
print("测试4：验证计算效率")
print("-" * 70)

try:
    for i, exit_point in enumerate(model.exit_points):
        n_params = model.get_num_params(exit_index=i)
        print(f"  Exit {i+1} (Block {exit_point}): {n_params:,} params")

    total_params = model.get_num_params()
    print(f"  Full model: {total_params:,} params")

    # 验证Exit 1的参数量 < Exit 3
    params_exit1 = model.get_num_params(exit_index=0)
    params_exit3 = model.get_num_params(exit_index=2)
    assert params_exit1 < params_exit3, "Exit 1应该比Exit 3参数少"

    saving = (1 - params_exit1 / params_exit3) * 100
    print(f"\n✓ Exit 1比Exit 3节省 {saving:.1f}% 参数")

except Exception as e:
    print(f"✗ 参数量计算失败: {e}")
    sys.exit(1)

print()

# ============================================
# 测试5：损失函数
# ============================================
print("测试5：损失函数")
print("-" * 70)

try:
    # 生成测试数据
    pred = torch.randn(batch_size, 2)
    target = torch.randn(batch_size, 2)
    states = ['fixation', 'saccade', 'pursuit', 'fixation']

    # 测试Angular Loss
    loss_angular = angular_loss(pred, target)
    assert not torch.isnan(loss_angular), "Angular loss包含NaN"
    print(f"✓ Angular Loss: {loss_angular.item():.4f}°")

    # 测试Multi-Exit Loss
    criterion_multi = MultiExitLoss()
    loss_multi, loss_dict = criterion_multi(outputs, target, states)
    assert not torch.isnan(loss_multi), "Multi-exit loss包含NaN"
    print(f"✓ Multi-Exit Loss: {loss_multi.item():.4f}")
    for key, val in loss_dict.items():
        print(f"    {key}: {val:.4f}")

    # 测试State-Aware Loss
    criterion_state = StateAwareLoss()
    loss_state = criterion_state(pred, target, states)
    assert not torch.isnan(loss_state), "State-aware loss包含NaN"
    print(f"✓ State-Aware Loss: {loss_state.item():.4f}")

except Exception as e:
    print(f"✗ 损失函数测试失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()

# ============================================
# 测试6：反向传播
# ============================================
print("测试6：反向传播与参数更新")
print("-" * 70)

try:
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # 保存初始参数
    initial_params = [p.clone() for p in model.parameters() if p.requires_grad]

    # 前向传播
    outputs = model.forward_all_exits(test_image)

    # 计算损失
    criterion = MultiExitLoss()
    loss, _ = criterion(outputs, target, states)

    # 反向传播
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # 验证参数已更新
    updated = False
    for initial_p, current_p in zip(initial_params,
                                    [p for p in model.parameters() if p.requires_grad]):
        if not torch.equal(initial_p, current_p):
            updated = True
            break

    assert updated, "参数没有更新"
    print("✓ 反向传播成功")
    print(f"✓ 参数已更新")
    print(f"✓ Loss: {loss.item():.4f}")

except Exception as e:
    print(f"✗ 反向传播失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()

# ============================================
# 测试7：状态路由正确性
# ============================================
print("测试7：状态路由正确性")
print("-" * 70)

try:
    model.eval()

    # 测试相同输入，不同状态应该产生不同输出
    test_img = torch.randn(1, 1, 224, 224)

    with torch.no_grad():
        gaze_saccade = model(test_img, state='saccade')
        gaze_pursuit = model(test_img, state='pursuit')
        gaze_fixation = model(test_img, state='fixation')

    # 验证不同状态产生不同输出（因为使用了不同的Exit）
    diff_sp = torch.norm(gaze_saccade - gaze_pursuit).item()
    diff_sf = torch.norm(gaze_saccade - gaze_fixation).item()
    diff_pf = torch.norm(gaze_pursuit - gaze_fixation).item()

    print(f"  Saccade vs Pursuit: {diff_sp:.4f}")
    print(f"  Saccade vs Fixation: {diff_sf:.4f}")
    print(f"  Pursuit vs Fixation: {diff_pf:.4f}")

    # 应该有差异（但可能很小）
    assert diff_sp > 0 or diff_sf > 0 or diff_pf > 0, "不同状态应该产生不同输出"
    print("✓ 状态路由工作正常（不同状态产生不同输出）")

except Exception as e:
    print(f"✗ 状态路由测试失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()

# ============================================
# 总结
# ============================================
print("=" * 70)
print("✓✓✓ 所有测试通过！Multi-Exit ViT模型实现正确 ✓✓✓")
print("=" * 70)
print()
print("验证内容总结:")
print("  ✓ 模型创建与架构")
print("  ✓ 单Exit推理模式（状态感知路由）")
print("  ✓ 多Exit训练模式")
print("  ✓ 计算效率（参数量差异）")
print("  ✓ 损失函数（Angular, Multi-Exit, State-Aware）")
print("  ✓ 反向传播与参数更新")
print("  ✓ 状态路由正确性")
print()
print("下一步: 运行训练脚本")
print("  python src/train/train_multi_exit_vit.py")
print("=" * 70)
