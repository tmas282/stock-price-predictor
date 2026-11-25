import os

import matplotlib
matplotlib.use("Agg")  # Headless backend for saving files without GUI
import matplotlib.pyplot as plt
import numpy as np


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def plot_training_metrics(
    train_losses: list[float],
    val_losses: list[float],
    val_accuracies: list[float],
    learning_rates: list[float],
    save_dir: str = "plots",
    filename: str = "training_metrics.png",
):
    """
    Plots training loss, validation loss, directional accuracy, and learning rate curves over epochs.
    """
    ensure_dir(save_dir)
    epochs = list(range(1, len(train_losses) + 1))

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

    # 1. Loss Curve
    ax1.plot(epochs, train_losses, "b-o", label="Train BCE Loss")
    ax1.plot(epochs, val_losses, "r-s", label="Val BCE Loss")
    ax1.axhline(float(np.log(2.0)), color="gray", linestyle="--", label="Coin Flip (ln 2 = 0.693)")
    ax1.set_title("BCE Loss Progression")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("BCE Loss")
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend()

    # 2. Directional Accuracy Curve
    ax2.plot(epochs, val_accuracies, "g-^", label="Val Directional Acc (%)")
    ax2.axhline(50.0, color="gray", linestyle="--", label="Baseline (50% Coin Flip)")
    ax2.set_title("Directional Accuracy (%)")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend()

    # 3. Learning Rate Progression
    ax3.plot(epochs, learning_rates, "m-d", label="Learning Rate")
    ax3.set_title("Learning Rate Schedule")
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("Learning Rate")
    ax3.grid(True, linestyle="--", alpha=0.6)
    ax3.legend()

    plt.tight_layout()
    output_path = os.path.join(save_dir, filename)
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved training metrics plot -> {output_path}")
