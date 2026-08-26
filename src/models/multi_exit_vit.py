"""
Multi-Exit Vision Transformer for State-Aware Gaze Estimation

State-Exit Mapping:
- Saccade (fast eye movement) -> Exit 1 (Early, Block 3)
- Pursuit (tracking) -> Exit 2 (Medium, Block 6)
- Fixation (steady gaze) -> Exit 3 (Deep, Block 12)
"""

import torch
import torch.nn as nn
import timm
from typing import Dict, Optional, List


class GazeRegressionHead(nn.Module):
    """Gaze regression head for predicting 2D gaze coordinates"""

    def __init__(self, embed_dim: int, hidden_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.fc = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2)  # (x, y) coordinates
        )

    def forward(self, x):
        """
        Args:
            x: (B, N, D) token embeddings from ViT
        Returns:
            gaze: (B, 2) normalized gaze coordinates in [-1, 1]
        """
        cls_token = x[:, 0]  # Extract CLS token
        gaze = self.fc(cls_token)
        return gaze


class MultiExitViT(nn.Module):
    """
    Multi-Exit Vision Transformer with State-Aware Routing

    Architecture:
        Input (224x224)
        -> Patch Embedding
        -> Transformer Blocks 1-3 -> Exit 1 (Saccade)
        -> Transformer Blocks 4-6 -> Exit 2 (Pursuit)
        -> Transformer Blocks 7-12 -> Exit 3 (Fixation)
    """

    def __init__(
        self,
        model_name: str = 'vit_small_patch16_224',
        pretrained: bool = True,
        exit_points: List[int] = [3, 6, 12],
        hidden_dim: int = 256,
        dropout: float = 0.1
    ):
        """
        Args:
            model_name: timm model name
            pretrained: use pretrained weights
            exit_points: which transformer blocks to exit at [early, medium, late]
            hidden_dim: hidden dimension in regression heads
            dropout: dropout rate
        """
        super().__init__()

        self.exit_points = sorted(exit_points)
        assert len(self.exit_points) == 3, "Must have exactly 3 exit points"

        # Load pretrained ViT backbone
        self.vit = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,  # Remove classification head
            global_pool=''  # Keep all tokens
        )

        self.embed_dim = self.vit.embed_dim
        self.blocks = self.vit.blocks
        self.num_blocks = len(self.blocks)

        # Verify exit points are valid
        for exit_idx in self.exit_points:
            assert 1 <= exit_idx <= self.num_blocks, \
                f"Exit point {exit_idx} out of range [1, {self.num_blocks}]"

        # Create exit heads
        self.exit_heads = nn.ModuleDict({
            f'exit_{i+1}': GazeRegressionHead(self.embed_dim, hidden_dim, dropout)
            for i in range(len(self.exit_points))
        })

        # State to exit mapping
        self.state_to_exit = {
            'saccade': 0,   # Early exit
            'pursuit': 1,   # Medium exit
            'fixation': 2   # Late exit (full model)
        }

        print(f"Multi-Exit ViT initialized:")
        print(f"  Backbone: {model_name}")
        print(f"  Embed dim: {self.embed_dim}")
        print(f"  Exit points: {self.exit_points}")
        print(f"  State mapping: {self.state_to_exit}")

    def forward_features(self, x: torch.Tensor, exit_after: Optional[int] = None):
        """
        Forward pass through backbone up to a specific block

        Args:
            x: (B, C, H, W) input images
            exit_after: exit after this block index (0-indexed), None for full forward

        Returns:
            x: (B, N, D) token embeddings
        """
        # Patch embedding
        x = self.vit.patch_embed(x)

        # Add positional encoding
        x = self.vit._pos_embed(x)

        # Patch dropout
        x = self.vit.patch_drop(x)

        # Pre-normalization (if exists)
        x = self.vit.norm_pre(x)

        # Transformer blocks
        num_blocks = exit_after + 1 if exit_after is not None else self.num_blocks
        for i in range(num_blocks):
            x = self.blocks[i](x)

        return x

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[str] = None,
        exit_index: Optional[int] = None
    ):
        """
        Single-exit forward pass (inference mode)

        Args:
            x: (B, C, H, W) input images
            state: eye movement state ('saccade', 'pursuit', 'fixation')
            exit_index: explicit exit index (0, 1, 2), overrides state

        Returns:
            gaze: (B, 2) predicted gaze coordinates
        """
        # Handle grayscale images
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)

        # Determine which exit to use
        if exit_index is None:
            if state is None:
                state = 'fixation'  # Default to full model
            exit_index = self.state_to_exit.get(state, 2)

        # Forward to the selected exit point
        exit_block_idx = self.exit_points[exit_index] - 1  # Convert to 0-indexed
        tokens = self.forward_features(x, exit_after=exit_block_idx)

        # Predict gaze
        exit_name = f'exit_{exit_index + 1}'
        gaze = self.exit_heads[exit_name](tokens)

        return gaze

    def forward_all_exits(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Multi-exit forward pass (training mode)
        Computes predictions at all exit points

        Args:
            x: (B, C, H, W) input images

        Returns:
            outputs: dict with keys 'exit_1', 'exit_2', 'exit_3'
                     each containing (B, 2) gaze predictions
        """
        # Handle grayscale images
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)

        outputs = {}

        for i, exit_block_idx in enumerate(self.exit_points):
            # Forward to this exit point
            tokens = self.forward_features(x, exit_after=exit_block_idx - 1)

            # Predict gaze
            exit_name = f'exit_{i + 1}'
            outputs[exit_name] = self.exit_heads[exit_name](tokens)

        return outputs

    def get_exit_for_state(self, state: str) -> int:
        """Get exit index for a given state"""
        return self.state_to_exit.get(state, 2)

    def get_num_params(self, exit_index: Optional[int] = None):
        """
        Get number of parameters

        Args:
            exit_index: if specified, count params up to that exit
                       if None, count all params
        """
        if exit_index is None:
            return sum(p.numel() for p in self.parameters())

        # Count backbone params up to exit
        exit_block_idx = self.exit_points[exit_index]
        backbone_params = sum(
            p.numel() for i, block in enumerate(self.blocks[:exit_block_idx])
            for p in block.parameters()
        )

        # Add patch embedding and pos encoding
        backbone_params += sum(p.numel() for p in self.vit.patch_embed.parameters())
        if hasattr(self.vit, 'pos_embed'):
            backbone_params += self.vit.pos_embed.numel()

        # Add exit head params
        exit_name = f'exit_{exit_index + 1}'
        head_params = sum(p.numel() for p in self.exit_heads[exit_name].parameters())

        return backbone_params + head_params


# Utility functions
def create_model(
    model_name: str = 'vit_small_patch16_224',
    pretrained: bool = True,
    exit_points: List[int] = [3, 6, 12]
) -> MultiExitViT:
    """Factory function to create Multi-Exit ViT model"""
    return MultiExitViT(
        model_name=model_name,
        pretrained=pretrained,
        exit_points=exit_points
    )


if __name__ == '__main__':
    print("=== Multi-Exit ViT Model Test ===\n")

    # Create model
    model = create_model(pretrained=False)
    model.eval()

    # Test input
    batch_size = 4
    x = torch.randn(batch_size, 1, 224, 224)  # Grayscale eye images

    print("\n1. Single-exit inference (state-aware)")
    for state in ['saccade', 'pursuit', 'fixation']:
        with torch.no_grad():
            gaze = model(x, state=state)
        print(f"   {state:10s} -> {gaze.shape} | Exit {model.get_exit_for_state(state) + 1}")

    print("\n2. Multi-exit training forward")
    with torch.no_grad():
        outputs = model.forward_all_exits(x)
    for exit_name, gaze in outputs.items():
        print(f"   {exit_name}: {gaze.shape}")

    print("\n3. Model statistics")
    for i in range(3):
        n_params = model.get_num_params(exit_index=i)
        print(f"   Exit {i+1} (Block {model.exit_points[i]}): {n_params:,} parameters")
    total_params = model.get_num_params()
    print(f"   Total: {total_params:,} parameters")

    print("\n✓ Model test passed!")
