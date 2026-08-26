"""
Training script for Multi-Exit ViT

Usage:
    python src/train/train_multi_exit_vit.py --config configs/default.yaml
"""

import sys
import os
sys.path.append('src')

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import argparse
from tqdm import tqdm
import json
from datetime import datetime

from models.multi_exit_vit import MultiExitViT
from data.mock_dataset import MockGazeDataset
from train.losses import MultiExitLoss, CombinedLoss


class Trainer:
    """Training manager for Multi-Exit ViT"""

    def __init__(self, config: dict):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        print(f"Device: {self.device}")

        # Create model
        self.model = MultiExitViT(
            model_name=config['model']['name'],
            pretrained=config['model']['pretrained'],
            exit_points=config['model']['exit_points']
        ).to(self.device)

        # Create loss
        if config['loss']['type'] == 'multi_exit':
            self.criterion = MultiExitLoss(
                exit_weights=config['loss']['exit_weights'],
                loss_type=config['loss']['base_loss']
            )
        elif config['loss']['type'] == 'combined':
            self.criterion = CombinedLoss(
                exit_weights=config['loss']['exit_weights'],
                state_weights=config['loss'].get('state_weights'),
                loss_type=config['loss']['base_loss']
            )

        # Create optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config['training']['lr'],
            weight_decay=config['training']['weight_decay']
        )

        # Create scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config['training']['epochs'],
            eta_min=config['training']['lr'] * 0.01
        )

        # Logging
        self.writer = SummaryWriter(config['logging']['log_dir'])
        self.best_val_loss = float('inf')
        self.global_step = 0

    def train_epoch(self, train_loader, epoch):
        """Train for one epoch"""
        self.model.train()

        epoch_losses = {
            'total': 0.0,
            'exit_1': 0.0,
            'exit_2': 0.0,
            'exit_3': 0.0
        }

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")

        for batch_idx, batch in enumerate(pbar):
            images = batch['image'].to(self.device)
            gazes = batch['gaze'].to(self.device)
            states = batch['state']

            # Forward pass (all exits)
            outputs = self.model.forward_all_exits(images)

            # Compute loss
            loss, loss_dict = self.criterion(outputs, gazes, states)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Update metrics
            epoch_losses['total'] += loss.item()
            for key in ['exit_1', 'exit_2', 'exit_3']:
                if f'loss_{key}' in loss_dict:
                    epoch_losses[key] += loss_dict[f'loss_{key}']

            # Log to tensorboard
            if batch_idx % 10 == 0:
                for key, val in loss_dict.items():
                    self.writer.add_scalar(f'train/{key}', val, self.global_step)
                self.writer.add_scalar('train/lr', self.optimizer.param_groups[0]['lr'], self.global_step)

            self.global_step += 1

            # Update progress bar
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        # Average losses
        num_batches = len(train_loader)
        for key in epoch_losses:
            epoch_losses[key] /= num_batches

        return epoch_losses

    def validate(self, val_loader, epoch):
        """Validate on validation set"""
        self.model.eval()

        val_losses = {
            'total': 0.0,
            'exit_1': 0.0,
            'exit_2': 0.0,
            'exit_3': 0.0
        }

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                images = batch['image'].to(self.device)
                gazes = batch['gaze'].to(self.device)
                states = batch['state']

                # Forward pass
                outputs = self.model.forward_all_exits(images)

                # Compute loss
                loss, loss_dict = self.criterion(outputs, gazes, states)

                val_losses['total'] += loss.item()
                for key in ['exit_1', 'exit_2', 'exit_3']:
                    if f'loss_{key}' in loss_dict:
                        val_losses[key] += loss_dict[f'loss_{key}']

        # Average losses
        num_batches = len(val_loader)
        for key in val_losses:
            val_losses[key] /= num_batches

        # Log to tensorboard
        for key, val in val_losses.items():
            self.writer.add_scalar(f'val/loss_{key}', val, epoch)

        return val_losses

    def save_checkpoint(self, epoch, val_loss, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss': val_loss,
            'config': self.config
        }

        # Save latest
        latest_path = os.path.join(self.config['logging']['checkpoint_dir'], 'latest.pth')
        torch.save(checkpoint, latest_path)

        # Save best
        if is_best:
            best_path = os.path.join(self.config['logging']['checkpoint_dir'], 'best.pth')
            torch.save(checkpoint, best_path)
            print(f"  ✓ Saved best model (val_loss: {val_loss:.4f})")

    def train(self, train_loader, val_loader):
        """Main training loop"""
        print("\n" + "="*50)
        print("Starting Training")
        print("="*50)

        for epoch in range(1, self.config['training']['epochs'] + 1):
            print(f"\nEpoch {epoch}/{self.config['training']['epochs']}")

            # Train
            train_losses = self.train_epoch(train_loader, epoch)

            # Validate
            val_losses = self.validate(val_loader, epoch)

            # Update scheduler
            self.scheduler.step()

            # Print summary
            print(f"\n  Train Loss: {train_losses['total']:.4f}")
            print(f"    Exit 1: {train_losses['exit_1']:.4f}")
            print(f"    Exit 2: {train_losses['exit_2']:.4f}")
            print(f"    Exit 3: {train_losses['exit_3']:.4f}")

            print(f"  Val Loss: {val_losses['total']:.4f}")
            print(f"    Exit 1: {val_losses['exit_1']:.4f}")
            print(f"    Exit 2: {val_losses['exit_2']:.4f}")
            print(f"    Exit 3: {val_losses['exit_3']:.4f}")

            # Save checkpoint
            is_best = val_losses['total'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_losses['total']

            self.save_checkpoint(epoch, val_losses['total'], is_best)

        print("\n" + "="*50)
        print("Training Complete!")
        print(f"Best Val Loss: {self.best_val_loss:.4f}")
        print("="*50)

        self.writer.close()


def create_dataloaders(config):
    """Create train and validation dataloaders"""

    # Get available users
    data_root = config['data']['root']
    all_users = [f'user{i}' for i in range(4, 28)]  # user4-27

    # Filter to only existing users
    existing_users = [
        u for u in all_users
        if os.path.exists(os.path.join(data_root, u))
    ]

    print(f"Found {len(existing_users)} users: {existing_users[:5]}...")

    # Split users
    n_train = int(len(existing_users) * config['data']['train_split'])
    n_val = int(len(existing_users) * config['data']['val_split'])

    train_users = existing_users[:n_train]
    val_users = existing_users[n_train:n_train + n_val]

    print(f"Train users: {len(train_users)}")
    print(f"Val users: {len(val_users)}")

    # Create datasets
    train_dataset = MockGazeDataset(
        data_root=data_root,
        users=train_users,
        eye=config['data']['eye'],
        img_size=config['data']['img_size']
    )

    val_dataset = MockGazeDataset(
        data_root=data_root,
        users=val_users,
        eye=config['data']['eye'],
        img_size=config['data']['img_size']
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['training']['num_workers'],
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training']['num_workers'],
        pin_memory=True
    )

    return train_loader, val_loader


def main():
    # Default configuration
    config = {
        'model': {
            'name': 'vit_small_patch16_224',
            'pretrained': True,
            'exit_points': [3, 6, 12]
        },
        'loss': {
            'type': 'multi_exit',  # or 'combined'
            'base_loss': 'angular',
            'exit_weights': {
                'exit_1': 0.3,
                'exit_2': 0.3,
                'exit_3': 0.4
            }
        },
        'training': {
            'epochs': 50,
            'batch_size': 32,
            'lr': 1e-4,
            'weight_decay': 1e-5,
            'num_workers': 0  # Windows compatibility
        },
        'data': {
            'root': 'eye_data',
            'eye': 'left',
            'img_size': 224,
            'train_split': 0.7,
            'val_split': 0.15
        },
        'logging': {
            'log_dir': 'experiments/logs/' + datetime.now().strftime('%Y%m%d_%H%M%S'),
            'checkpoint_dir': 'experiments/checkpoints'
        }
    }

    # Create directories
    os.makedirs(config['logging']['log_dir'], exist_ok=True)
    os.makedirs(config['logging']['checkpoint_dir'], exist_ok=True)

    # Save config
    config_path = os.path.join(config['logging']['log_dir'], 'config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    print("Configuration:")
    print(json.dumps(config, indent=2))

    # Create dataloaders
    train_loader, val_loader = create_dataloaders(config)

    # Create trainer
    trainer = Trainer(config)

    # Train
    trainer.train(train_loader, val_loader)


if __name__ == '__main__':
    main()
