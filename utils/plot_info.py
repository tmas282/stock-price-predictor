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
    ax1.plot(epochs, train_losses, "b-o", label="Train Loss")
    ax1.plot(epochs, val_losses, "r-s", label="Val Loss")
    ax1.set_title("Self-Supervised Loss Progression")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Huber Loss")
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


def plot_pair_accuracies(
    pair_accuracies: dict[str, float],
    save_dir: str = "plots",
    filename: str = "per_pair_accuracy.png",
):
    """
    Bar chart showing directional accuracy across all currency pairs with a 50% benchmark line.
    """
    ensure_dir(save_dir)
    pairs = list(pair_accuracies.keys())
    accs = [pair_accuracies[p] for p in pairs]

    # Sort pairs by accuracy descending
    sorted_indices = np.argsort(accs)[::-1]
    pairs = [pairs[i] for i in sorted_indices]
    accs = [accs[i] for i in sorted_indices]

    plt.figure(figsize=(14, 6))
    bars = plt.bar(pairs, accs, color="steelblue", edgecolor="black", alpha=0.85)

    # Highlight above vs below 50%
    for bar, acc in zip(bars, accs):
        bar.set_color("forestgreen" if acc >= 50.0 else "indianred")

    plt.axhline(50.0, color="black", linestyle="--", linewidth=1.2, label="50% Coin Flip Baseline")
    plt.title("Directional Accuracy by Currency Pair")
    plt.xlabel("Currency Pair")
    plt.ylabel("Accuracy (%)")
    plt.ylim(min(40.0, min(accs) - 2.0), max(60.0, max(accs) + 2.0))
    plt.xticks(rotation=45, ha="right")
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.legend()

    plt.tight_layout()
    output_path = os.path.join(save_dir, filename)
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved per-pair accuracy plot -> {output_path}")
