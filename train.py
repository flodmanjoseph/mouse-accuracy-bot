"""Training loop for the target detection CNN."""
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from model import TargetDetector, count_parameters
from dataset import TargetDataset
from config import (
    DEVICE, BATCH_SIZE, LEARNING_RATE, NUM_EPOCHS, TRAIN_SPLIT,
)

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")


def train(epochs=None, lr=None, quick=False):
    """Train the target detection model.

    Args:
        epochs: Number of training epochs (overrides config).
        lr: Learning rate (overrides config).
        quick: If True, train for only 10 epochs.
    """
    if epochs is None:
        epochs = 10 if quick else NUM_EPOCHS
    if lr is None:
        lr = LEARNING_RATE

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    print(f"Device: {DEVICE}")

    # Load dataset
    dataset = TargetDataset(
        os.path.join(DATA_DIR, "images"),
        os.path.join(DATA_DIR, "labels"),
        augment=True,
    )
    print(f"Dataset size: {len(dataset)} samples")

    # Split train/validation
    train_size = int(len(dataset) * TRAIN_SPLIT)
    val_size = len(dataset) - train_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    # Disable augmentation for validation
    val_dataset_no_aug = TargetDataset(
        os.path.join(DATA_DIR, "images"),
        os.path.join(DATA_DIR, "labels"),
        augment=False,
    )
    val_indices = val_set.indices
    val_set_clean = torch.utils.data.Subset(val_dataset_no_aug, val_indices)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set_clean, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print(f"Train: {train_size} | Validation: {val_size}")

    # Model
    model = TargetDetector().to(DEVICE)
    print(f"Model parameters: {count_parameters(model):,}")

    # Optimizer + scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=5, factor=0.5
    )

    # Loss: MSE on heatmap
    criterion = nn.MSELoss()

    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        # --- Train ---
        model.train()
        train_loss = 0.0
        for images, heatmaps in train_loader:
            images = images.to(DEVICE)
            heatmaps = heatmaps.to(DEVICE)

            pred = model(images)
            loss = criterion(pred, heatmaps)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)

        train_loss /= train_size

        # --- Validate ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, heatmaps in val_loader:
                images = images.to(DEVICE)
                heatmaps = heatmaps.to(DEVICE)

                pred = model(images)
                loss = criterion(pred, heatmaps)
                val_loss += loss.item() * images.size(0)

        val_loss /= val_size

        # Step scheduler
        scheduler.step(val_loss)

        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "best.pth"))
            marker = " * saved"
        else:
            marker = ""

        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch:3d}/{epochs} | "
              f"Train: {train_loss:.6f} | Val: {val_loss:.6f} | "
              f"LR: {current_lr:.1e}{marker}")

    # Save final
    torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "final.pth"))
    print(f"\nTraining complete. Best val loss: {best_val_loss:.6f}")
    print(f"Checkpoints saved to {CHECKPOINT_DIR}/")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train target detection model")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--quick", action="store_true", help="Quick training (10 epochs)")
    args = parser.parse_args()
    train(epochs=args.epochs, lr=args.lr, quick=args.quick)
