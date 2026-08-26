"""
Loss functions for Multi-Exit Vision Transformer

Includes:
- Angular error loss
- Multi-exit loss with weighting
- State-specific loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


def angular_loss(pred: torch.Tensor, target: torch.Tensor, reduction: str = 'mean') -> torch.Tensor:
    """
    Angular error loss in degrees

    Args:
        pred: (B, 2) predicted gaze coordinates in [-1, 1]
        target: (B, 2) ground truth gaze coordinates in [-1, 1]
        reduction: 'mean', 'sum', or 'none'

    Returns:
        loss: angular error in degrees
    """
    # Compute Euclidean distance
    diff = pred - target
    angular_error = torch.norm(diff, dim=1)

    # Convert to degrees (approximate)
    # Assuming normalized coordinates correspond to screen space
    # This is a simplified version; real angular error requires screen geometry
    angular_error_deg = angular_error * 90.0  # Scale factor

    if reduction == 'mean':
        return angular_error_deg.mean()
    elif reduction == 'sum':
        return angular_error_deg.sum()
    else:
        return angular_error_deg


def mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean squared error loss"""
    return F.mse_loss(pred, target)


def l1_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """L1 loss"""
    return F.l1_loss(pred, target)


class MultiExitLoss(nn.Module):
    """
    Multi-exit loss with weighted combination

    L_total = λ1·L_exit1 + λ2·L_exit2 + λ3·L_exit3
    """

    def __init__(
        self,
        exit_weights: Optional[Dict[str, float]] = None,
        loss_type: str = 'angular',
        temperature: float = 1.0
    ):
        """
        Args:
            exit_weights: weights for each exit {'exit_1': 0.3, 'exit_2': 0.3, 'exit_3': 0.4}
            loss_type: 'angular', 'mse', or 'l1'
            temperature: temperature for softening weights
        """
        super().__init__()

        # Default weights: emphasize deeper exits
        if exit_weights is None:
            exit_weights = {
                'exit_1': 0.3,
                'exit_2': 0.3,
                'exit_3': 0.4
            }

        self.exit_weights = exit_weights
        self.loss_type = loss_type
        self.temperature = temperature

        # Select base loss function
        if loss_type == 'angular':
            self.base_loss = angular_loss
        elif loss_type == 'mse':
            self.base_loss = mse_loss
        elif loss_type == 'l1':
            self.base_loss = l1_loss
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")

    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: torch.Tensor,
        states: Optional[list] = None
    ):
        """
        Compute multi-exit loss

        Args:
            predictions: dict of predictions from each exit
                        {'exit_1': (B, 2), 'exit_2': (B, 2), 'exit_3': (B, 2)}
            targets: (B, 2) ground truth gaze coordinates
            states: optional list of states for adaptive weighting

        Returns:
            total_loss: weighted combination of exit losses
            loss_dict: individual losses for logging
        """
        loss_dict = {}
        total_loss = 0.0

        for exit_name, pred in predictions.items():
            # Compute loss for this exit
            loss = self.base_loss(pred, targets)
            loss_dict[f'loss_{exit_name}'] = loss.item()

            # Add weighted loss to total
            weight = self.exit_weights.get(exit_name, 1.0)
            total_loss += weight * loss

        loss_dict['loss_total'] = total_loss.item()

        return total_loss, loss_dict


class StateAwareLoss(nn.Module):
    """
    State-aware loss that applies different weights based on eye movement state

    Intuition:
    - Saccade: prioritize speed over precision (lower weight)
    - Pursuit: balance speed and precision
    - Fixation: maximize precision (higher weight)
    """

    def __init__(
        self,
        state_weights: Optional[Dict[str, float]] = None,
        loss_type: str = 'angular'
    ):
        """
        Args:
            state_weights: weights for each state
            loss_type: 'angular', 'mse', or 'l1'
        """
        super().__init__()

        # Default: weight fixation more heavily
        if state_weights is None:
            state_weights = {
                'saccade': 0.5,
                'pursuit': 1.0,
                'fixation': 1.5
            }

        self.state_weights = state_weights

        if loss_type == 'angular':
            self.base_loss = angular_loss
        elif loss_type == 'mse':
            self.base_loss = mse_loss
        elif loss_type == 'l1':
            self.base_loss = l1_loss
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        states: list
    ):
        """
        Compute state-aware loss

        Args:
            predictions: (B, 2) predicted gaze
            targets: (B, 2) ground truth gaze
            states: list of B state strings

        Returns:
            loss: weighted loss
        """
        # Compute per-sample loss
        losses = self.base_loss(predictions, targets, reduction='none')

        # Apply state-specific weights
        weights = torch.tensor(
            [self.state_weights.get(state, 1.0) for state in states],
            device=losses.device
        )

        weighted_loss = (losses * weights).mean()

        return weighted_loss


class CombinedLoss(nn.Module):
    """
    Combined multi-exit + state-aware loss
    """

    def __init__(
        self,
        exit_weights: Optional[Dict[str, float]] = None,
        state_weights: Optional[Dict[str, float]] = None,
        loss_type: str = 'angular'
    ):
        super().__init__()

        self.multi_exit_loss = MultiExitLoss(exit_weights, loss_type)
        self.state_aware_loss = StateAwareLoss(state_weights, loss_type)

    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: torch.Tensor,
        states: list
    ):
        """
        Compute combined loss

        Args:
            predictions: dict of predictions from each exit
            targets: (B, 2) ground truth gaze
            states: list of B state strings

        Returns:
            total_loss: combined loss
            loss_dict: detailed loss breakdown
        """
        # Multi-exit loss
        multi_exit_loss, loss_dict = self.multi_exit_loss(predictions, targets, states)

        # State-aware loss for the final exit
        final_pred = predictions['exit_3']
        state_loss = self.state_aware_loss(final_pred, targets, states)

        # Combine
        total_loss = multi_exit_loss + 0.1 * state_loss

        loss_dict['loss_state_aware'] = state_loss.item()
        loss_dict['loss_combined'] = total_loss.item()

        return total_loss, loss_dict


if __name__ == '__main__':
    print("=== Loss Functions Test ===\n")

    # Create dummy data
    batch_size = 4
    pred = torch.randn(batch_size, 2)
    target = torch.randn(batch_size, 2)
    states = ['fixation', 'saccade', 'pursuit', 'fixation']

    # Test multi-exit predictions
    predictions = {
        'exit_1': torch.randn(batch_size, 2),
        'exit_2': torch.randn(batch_size, 2),
        'exit_3': torch.randn(batch_size, 2)
    }

    print("1. Angular loss")
    loss = angular_loss(pred, target)
    print(f"   Loss: {loss.item():.4f} degrees")

    print("\n2. Multi-exit loss")
    criterion = MultiExitLoss()
    loss, loss_dict = criterion(predictions, target, states)
    print(f"   Total loss: {loss.item():.4f}")
    for k, v in loss_dict.items():
        print(f"   {k}: {v:.4f}")

    print("\n3. State-aware loss")
    criterion = StateAwareLoss()
    loss = criterion(pred, target, states)
    print(f"   Loss: {loss.item():.4f}")

    print("\n4. Combined loss")
    criterion = CombinedLoss()
    loss, loss_dict = criterion(predictions, target, states)
    print(f"   Total loss: {loss.item():.4f}")
    for k, v in loss_dict.items():
        print(f"   {k}: {v:.4f}")

    print("\n✓ Loss functions test passed!")
