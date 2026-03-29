"""PyTorch Dataset for target detection with Gaussian heatmap generation."""
import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
import cv2
from config import INPUT_SIZE, HEATMAP_SIZE, GAUSSIAN_SIGMA


def generate_heatmap(targets, img_size, heatmap_size=None, sigma=None):
    """Create a float32 heatmap with Gaussian peaks at target locations.

    Args:
        targets: List of {"x": int, "y": int, "radius": int} in img_size coordinates.
        img_size: (width, height) of the source image.
        heatmap_size: (width, height) of the output heatmap.
        sigma: Gaussian sigma for blobs.

    Returns:
        numpy float32 array of shape (heatmap_height, heatmap_width), values [0, 1].
    """
    if heatmap_size is None:
        heatmap_size = HEATMAP_SIZE
    if sigma is None:
        sigma = GAUSSIAN_SIGMA

    hm_w, hm_h = heatmap_size
    img_w, img_h = img_size
    heatmap = np.zeros((hm_h, hm_w), dtype=np.float32)

    for t in targets:
        # Scale target coordinates from image space to heatmap space
        cx = t["x"] * hm_w / img_w
        cy = t["y"] * hm_h / img_h

        # Generate 2D Gaussian
        y_grid, x_grid = np.mgrid[0:hm_h, 0:hm_w].astype(np.float32)
        gaussian = np.exp(-((x_grid - cx) ** 2 + (y_grid - cy) ** 2) / (2 * sigma ** 2))

        # Use max to prevent overlapping blobs from exceeding 1.0
        heatmap = np.maximum(heatmap, gaussian)

    return np.clip(heatmap, 0.0, 1.0)


class TargetDataset(Dataset):
    """Dataset of screenshots + auto-labeled target heatmaps."""

    def __init__(self, image_dir, label_dir, augment=False):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.augment = augment

        # Find matching image/label pairs
        image_files = set(os.path.splitext(f)[0] for f in os.listdir(image_dir) if f.endswith(".png"))
        label_files = set(os.path.splitext(f)[0] for f in os.listdir(label_dir) if f.endswith(".json"))
        self.keys = sorted(image_files & label_files)

        if len(self.keys) == 0:
            raise ValueError(f"No matching image/label pairs in {image_dir} and {label_dir}")

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        key = self.keys[idx]

        # Load image
        img_path = os.path.join(self.image_dir, f"{key}.png")
        img = cv2.imread(img_path)
        img = cv2.resize(img, INPUT_SIZE, interpolation=cv2.INTER_AREA)

        # Load labels
        label_path = os.path.join(self.label_dir, f"{key}.json")
        with open(label_path) as f:
            label_data = json.load(f)

        targets = label_data["targets"]
        img_w = label_data["width"]
        img_h = label_data["height"]

        # Augmentation: random horizontal flip
        if self.augment and np.random.random() > 0.5:
            img = cv2.flip(img, 1)  # Flip horizontally
            targets = [
                {"x": img_w - t["x"], "y": t["y"], "radius": t["radius"]}
                for t in targets
            ]

        # Augmentation: random brightness/contrast jitter
        if self.augment and np.random.random() > 0.5:
            alpha = np.random.uniform(0.85, 1.15)  # Contrast
            beta = np.random.randint(-15, 16)       # Brightness
            img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

        # Generate heatmap from targets (using original image dimensions for scaling)
        heatmap = generate_heatmap(targets, (img_w, img_h))

        # Convert image: BGR -> RGB, HWC -> CHW, normalize to [0, 1]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))  # (3, H, W)

        # Heatmap: add channel dim
        heatmap = heatmap[np.newaxis, :, :]  # (1, H, W)

        return torch.from_numpy(img), torch.from_numpy(heatmap)
