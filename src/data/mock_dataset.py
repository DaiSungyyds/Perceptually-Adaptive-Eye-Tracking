"""
Mock Dataset for Multi-Exit ViT Development

This allows training to proceed before the I-VT state labeling is complete.
Generates synthetic state labels that follow realistic distributions.
"""

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import List, Optional
import json


class MockGazeDataset(Dataset):
    """
    Mock dataset with synthetic state labels

    Uses actual eye images from downloaded data, but generates:
    - Random (but realistic) state labels
    - Random (but bounded) gaze coordinates

    This allows model development to proceed independently.
    """

    def __init__(
        self,
        data_root: str,
        users: List[str],
        eye: str = 'left',
        img_size: int = 224,
        screen_size: tuple = (1920, 1080),
        transform=None,
        state_distribution: dict = None
    ):
        """
        Args:
            data_root: path to eye_data/ directory
            users: list of user folders to include
            eye: 'left' or 'right'
            img_size: resize images to this size
            screen_size: (width, height) for coordinate normalization
            transform: optional transforms
            state_distribution: probabilities for each state
        """
        self.data_root = data_root
        self.eye = eye
        self.eye_idx = '0' if eye == 'left' else '1'
        self.img_size = img_size
        self.screen_width, self.screen_height = screen_size
        self.transform = transform

        # Default state distribution (similar to real data)
        if state_distribution is None:
            self.state_distribution = {
                'fixation': 0.30,
                'pursuit': 0.60,
                'saccade': 0.10
            }
        else:
            self.state_distribution = state_distribution

        self.samples = []

        print(f"Loading Mock Dataset...")
        print(f"  Eye: {eye}")
        print(f"  Users: {users}")

        for user in users:
            user_dir = os.path.join(data_root, user)

            if not os.path.exists(user_dir):
                print(f"  Warning: {user} does not exist, skipping")
                continue

            # Find frames directory
            eye_dir = os.path.join(user_dir, self.eye_idx)
            frames_dir = os.path.join(eye_dir, 'frames')

            if not os.path.exists(frames_dir):
                print(f"  Warning: {user}/{eye} frames not found, skipping")
                continue

            # Get all frame files
            frame_files = sorted([
                f for f in os.listdir(frames_dir)
                if f.endswith('.png')
            ])

            # Create samples with mock labels
            for i, frame_file in enumerate(frame_files):
                self.samples.append({
                    'image_path': os.path.join(frames_dir, frame_file),
                    'user': user,
                    'frame_idx': i
                })

            print(f"  ✓ {user}/{eye}: {len(frame_files)} frames")

        print(f"\n✓ Mock Dataset loaded: {len(self.samples)} total samples")
        self._print_mock_statistics()

    def _generate_mock_state(self, idx: int) -> str:
        """Generate mock state label with realistic distribution"""
        # Use deterministic random seed based on index for reproducibility
        rng = np.random.RandomState(idx)

        states = list(self.state_distribution.keys())
        probs = list(self.state_distribution.values())

        state = rng.choice(states, p=probs)
        return state

    def _generate_mock_gaze(self, idx: int, state: str) -> tuple:
        """
        Generate mock gaze coordinates with state-dependent characteristics

        - Fixation: clustered around screen center
        - Saccade: more dispersed, can be anywhere
        - Pursuit: smooth trajectories (simulated with noise)
        """
        rng = np.random.RandomState(idx)

        if state == 'fixation':
            # Clustered around center with small variance
            center_x = self.screen_width / 2
            center_y = self.screen_height / 2
            gaze_x = center_x + rng.normal(0, self.screen_width * 0.1)
            gaze_y = center_y + rng.normal(0, self.screen_height * 0.1)

        elif state == 'saccade':
            # Uniform distribution (can look anywhere)
            gaze_x = rng.uniform(0, self.screen_width)
            gaze_y = rng.uniform(0, self.screen_height)

        else:  # pursuit
            # Smooth movement (simulated)
            t = (idx % 100) / 100.0  # pseudo-temporal component
            gaze_x = self.screen_width * (0.3 + 0.4 * np.sin(2 * np.pi * t))
            gaze_y = self.screen_height * (0.4 + 0.2 * np.cos(2 * np.pi * t))
            # Add noise
            gaze_x += rng.normal(0, self.screen_width * 0.05)
            gaze_y += rng.normal(0, self.screen_height * 0.05)

        # Clip to screen bounds
        gaze_x = np.clip(gaze_x, 0, self.screen_width)
        gaze_y = np.clip(gaze_y, 0, self.screen_height)

        return gaze_x, gaze_y

    def _normalize_gaze(self, gaze_x: float, gaze_y: float) -> tuple:
        """Normalize gaze coordinates to [-1, 1]"""
        norm_x = (gaze_x / self.screen_width) * 2 - 1
        norm_y = (gaze_y / self.screen_height) * 2 - 1
        return norm_x, norm_y

    def _print_mock_statistics(self):
        """Print expected state distribution"""
        print("\nMock State Distribution:")
        for state, prob in self.state_distribution.items():
            expected_count = int(len(self.samples) * prob)
            print(f"  {state}: {prob*100:.1f}% (~{expected_count} samples)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Load image
        try:
            img = cv2.imread(sample['image_path'], cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise ValueError(f"Failed to load: {sample['image_path']}")

            img = cv2.resize(img, (self.img_size, self.img_size))
            img = img.astype(np.float32) / 255.0
            img = torch.from_numpy(img).unsqueeze(0)  # (1, H, W)

            if self.transform:
                img = self.transform(img)

        except Exception as e:
            print(f"Error loading image: {e}")
            # Return zero image on error
            img = torch.zeros(1, self.img_size, self.img_size)

        # Generate mock labels
        state = self._generate_mock_state(idx)
        gaze_x, gaze_y = self._generate_mock_gaze(idx, state)
        norm_x, norm_y = self._normalize_gaze(gaze_x, gaze_y)

        gaze = torch.tensor([norm_x, norm_y], dtype=torch.float32)

        return {
            'image': img,
            'gaze': gaze,
            'state': state,
            'user': sample['user'],
            'frame_idx': sample['frame_idx']
        }


def create_mock_state_labels(
    data_root: str,
    output_path: str,
    users: List[str],
    eye: str = 'left'
):
    """
    Create a mock ivt_state_labels.json file

    This mimics the output from the I-VT classifier (同学A's work)
    so you can test the data pipeline before it's ready.
    """
    print("Generating mock state labels...")

    mock_labels = {}

    for user in users:
        user_dir = os.path.join(data_root, user)

        if not os.path.exists(user_dir):
            continue

        # Count frames
        eye_idx = '0' if eye == 'left' else '1'
        frames_dir = os.path.join(user_dir, eye_idx, 'frames')

        if not os.path.exists(frames_dir):
            continue

        frame_files = [f for f in os.listdir(frames_dir) if f.endswith('.png')]
        n_frames = len(frame_files)

        # Generate mock data
        states = []
        timestamps = []
        gaze_coords = []

        for i in range(n_frames):
            # Mock state (deterministic)
            rng = np.random.RandomState(i)
            state = rng.choice(
                ['fixation', 'saccade', 'pursuit'],
                p=[0.30, 0.10, 0.60]
            )
            states.append(state)

            # Mock timestamp (microseconds)
            timestamps.append(237060314 + i * 10000)

            # Mock gaze coordinates
            gaze_x = rng.uniform(0, 1920)
            gaze_y = rng.uniform(0, 1080)
            gaze_coords.append([int(gaze_x), int(gaze_y)])

        # Compute statistics
        from collections import Counter
        state_counts = Counter(states)
        statistics = {
            state: count / n_frames * 100
            for state, count in state_counts.items()
        }

        mock_labels[user] = {
            eye: {
                'n_samples': n_frames,
                'statistics': statistics,
                'states': states,
                'timestamps': timestamps,
                'gaze_coords': gaze_coords
            }
        }

        print(f"  ✓ {user}: {n_frames} samples")

    # Save to file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(mock_labels, f, indent=2)

    print(f"\n✓ Mock labels saved to: {output_path}")
    return mock_labels


if __name__ == '__main__':
    print("=== Mock Dataset Test ===\n")

    # Test mock dataset
    dataset = MockGazeDataset(
        data_root='eye_data',
        users=['user4', 'user5'],
        eye='left'
    )

    print(f"\n1. Dataset size: {len(dataset)}")

    print("\n2. Sample a few items:")
    for i in [0, 100, 500]:
        if i < len(dataset):
            sample = dataset[i]
            print(f"   Sample {i}:")
            print(f"     Image: {sample['image'].shape}")
            print(f"     Gaze: {sample['gaze'].numpy()}")
            print(f"     State: {sample['state']}")

    print("\n3. Test DataLoader:")
    from torch.utils.data import DataLoader

    loader = DataLoader(dataset, batch_size=4, shuffle=True)
    batch = next(iter(loader))

    print(f"   Batch images: {batch['image'].shape}")
    print(f"   Batch gazes: {batch['gaze'].shape}")
    print(f"   Batch states: {batch['state']}")

    print("\n4. Generate mock state labels file:")
    create_mock_state_labels(
        data_root='eye_data',
        output_path='data/mock_ivt_state_labels.json',
        users=['user4', 'user5']
    )

    print("\n✓ Mock dataset test passed!")
