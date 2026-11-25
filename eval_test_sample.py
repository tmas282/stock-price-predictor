import argparse
import os
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from candles.daily import TARGET_CLOSE_IDX
from config import Config
from dataset import LEN_1D, PRED_DAYS, build_multi_pair_datasets
from model.itransformer import ITransformer
from utils.torch_device import get_device


def plot_confusion_matrix(cm: np.ndarray, save_path: str):
    """
    Plots a 2x2 confusion matrix heatmap with raw counts and row-normalized percentages.
    cm layout: [[TN, FP], [FN, TP]] for rows [Actual DN, Actual UP] and cols [Pred DN, Pred UP].
    """
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    row_sums = cm.sum(axis=1, keepdims=True)
    row_pcts = np.where(row_sums > 0, (cm / np.maximum(row_sums, 1)) * 100.0, 0.0)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(row_pcts, cmap="Blues", vmin=0.0, vmax=100.0)
    plt.colorbar(im, ax=ax, label="Row % (Recall)")

    labels = ["DN (0)", "UP (1)"]
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels([f"Pred {l}" for l in labels])
    ax.set_yticklabels([f"Actual {l}" for l in labels])
    ax.set_title("Test Set Confusion Matrix (D+1..D+7)")

    for r in range(2):
        for c in range(2):
            val_pct = row_pcts[r, c]
            txt_color = "white" if val_pct > 55.0 else "black"
            ax.text(
                c,
                r,
                f"{cm[r, c]:,}\n({val_pct:.1f}%)",
                ha="center",
                va="center",
                color=txt_color,
                fontsize=11,
                fontweight="bold",
            )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved confusion matrix plot -> {save_path}")


def evaluate_full_test_split(
    model: nn.Module,
    test_dataset,
    loaded_pairs: list[str],
    batch_size: int,
    num_workers: int,
    device: torch.device | str,
    plots_dir: str,
):
    loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    criterion = nn.BCEWithLogitsLoss()
    model.eval()

    total_loss = 0.0
    all_probs = []
    all_targets = []

    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            target_dir = batch["target_dir"].to(device)  # (B, 7, 1)

            logits = model(x)
            loss = criterion(logits, target_dir.to(logits.dtype))
            total_loss += loss.item()

            probs = torch.sigmoid(logits.float()).squeeze(-1).cpu().numpy()   # (B, 7)
            targets = target_dir.float().squeeze(-1).cpu().numpy()            # (B, 7)
            all_probs.append(probs)
            all_targets.append(targets)

    probs_arr = np.concatenate(all_probs, axis=0)      # (N, 7)
    targets_arr = np.concatenate(all_targets, axis=0)  # (N, 7)
    preds_arr = probs_arr >= 0.5
    true_arr = targets_arr == 1.0

    avg_bce = total_loss / max(len(loader), 1)
    total_checks = int(true_arr.size)
    correct_arr = preds_arr == true_arr
    overall_acc = float(correct_arr.mean() * 100.0) if total_checks > 0 else 0.0

    tp = int((preds_arr & true_arr).sum())
    tn = int((~preds_arr & ~true_arr).sum())
    fp = int((preds_arr & ~true_arr).sum())
    fn = int((~preds_arr & true_arr).sum())

    pred_up_pct = ((tp + fp) / total_checks) * 100.0 if total_checks > 0 else 0.0
    true_up_pct = ((tp + fn) / total_checks) * 100.0 if total_checks > 0 else 0.0
    up_prec = (tp / (tp + fp) * 100.0) if (tp + fp) > 0 else 0.0
    up_rec = (tp / (tp + fn) * 100.0) if (tp + fn) > 0 else 0.0
    dn_prec = (tn / (tn + fn) * 100.0) if (tn + fn) > 0 else 0.0
    dn_rec = (tn / (tn + fp) * 100.0) if (tn + fp) > 0 else 0.0
    bal_acc = 0.5 * (up_rec + dn_rec)

    print("\n================ FULL TEST SPLIT EVALUATION ================")
    print(f"Pairs: {len(loaded_pairs)} | Windows: {len(test_dataset):,} | Total Checks (N x {PRED_DAYS}d): {total_checks:,}")
    print(f"Test BCE Loss:     {avg_bce:.6f} (Coin-flip ln(2) = 0.693147)")
    print(f"Overall Accuracy:  {overall_acc:.2f}% | Balanced Accuracy: {bal_acc:.2f}%")
    print(f"Class Balance:     Pred UP = {pred_up_pct:.2f}% | Actual UP = {true_up_pct:.2f}%")
    print(f"Prob Stats P(UP):  min={probs_arr.min():.4f}, mean={probs_arr.mean():.4f}, std={probs_arr.std():.4f}, max={probs_arr.max():.4f}")

    print("\n--- 2x2 Confusion Matrix ---")
    print(f"{'':<14} | {'Pred DN (0)':>12} | {'Pred UP (1)':>12} | {'Recall':>8}")
    print("-" * 55)
    print(f"{'Actual DN (0)':<14} | {tn:12,d} | {fp:12,d} | {dn_rec:7.2f}%")
    print(f"{'Actual UP (1)':<14} | {fn:12,d} | {tp:12,d} | {up_rec:7.2f}%")
    print("-" * 55)
    print(f"{'Precision':<14} | {dn_prec:11.2f}% | {up_prec:11.2f}%")

    print("\n--- High-Confidence Accuracy ---")
    for margin in (0.02, 0.05, 0.10):
        mask = np.abs(probs_arr - 0.5) >= margin
        cov = float(mask.mean() * 100.0)
        cnt = int(mask.sum())
        acc = float(correct_arr[mask].mean() * 100.0) if cnt > 0 else 0.0
        print(f"  |P(UP) - 0.5| >= {margin:.2f} : {acc:6.2f}% acc  ({cnt:>6,}/{total_checks:,} = {cov:5.1f}% cov)")

    print("\n--- Per-Horizon Accuracy (D+1 .. D+7) ---")
    day_accs = (correct_arr.mean(axis=0) * 100.0).tolist()
    print("  " + " | ".join([f"D+{d + 1}: {acc:.2f}%" for d, acc in enumerate(day_accs)]))

    cm = np.array([[tn, fp], [fn, tp]], dtype=np.int64)
    plot_confusion_matrix(cm, os.path.join(plots_dir, "test_confusion_matrix.png"))


def inspect_single_sample(
    model: nn.Module,
    test_dataset,
    sample_idx: int,
    use_random: bool,
    device: torch.device | str,
):
    datasets = getattr(test_dataset, "datasets", [test_dataset])
    if use_random:
        ds = random.choice(datasets)
        idx = random.randrange(len(ds))
    else:
        ds = datasets[0]
        idx = sample_idx

    if not (0 <= idx < len(ds)):
        raise IndexError(f"sample-idx {idx} out of range for {ds.pair_name} test set (len={len(ds)})")

    sample = ds[idx]
    x = sample["x"].unsqueeze(0).to(device)
    actual_closes = sample["actual_next_ohlcv"].numpy()[:, TARGET_CLOSE_IDX]
    actual_dirs = sample["target_dir"].squeeze(-1).numpy().astype(int)
    prev_close = float(sample["prev_close"].item())
    n_feats = int(sample["x"].shape[1])

    model.eval()
    with torch.no_grad():
        logits = model(x).squeeze(0).squeeze(-1).float()
        pred_probs = torch.sigmoid(logits).cpu().numpy()
    pred_dirs = (pred_probs >= 0.5).astype(int)

    print("\n================ SINGLE SAMPLE INSPECTION ================")
    print(f"Pair: {ds.pair_name} | Test Sample: {idx}/{len(ds) - 1} | Window: {LEN_1D}d x {n_feats} ch | Entry Close (t=0): {prev_close:.5f}\n")

    header = f"{'Day':<5} | {'Actual Close':>12} | {'Move vs t=0':>11} | {'Actual':^6} | {'P(UP)':>7} | {'Pred':^6} | {'Result':^6}"
    print(header)
    print("-" * len(header))

    correct = 0
    for d in range(PRED_DAYS):
        move = (actual_closes[d] - prev_close) / prev_close
        ok = pred_dirs[d] == actual_dirs[d]
        correct += int(ok)
        print(
            f"D+{d + 1:<3} | {actual_closes[d]:12.5f} | {move:+10.3%} | "
            f"{'1(UP)' if actual_dirs[d] else '0(DN)':^6} | {pred_probs[d]:7.4f} | "
            f"{'1(UP)' if pred_dirs[d] else '0(DN)':^6} | {'OK' if ok else 'MISS':^6}"
        )

    print("-" * len(header))
    print(f"Sample Directional Accuracy: {correct}/{PRED_DAYS} ({correct / PRED_DAYS:.1%})")


def main():
    parser = argparse.ArgumentParser(description="Evaluate iTransformer 7D cumulative direction predictions on the test split.")
    parser.add_argument("--pair", type=str, default="ALL", help="Currency pair or 'ALL' for all 28 pairs (default: ALL)")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt", help="Model checkpoint path")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for full test split evaluation")
    parser.add_argument("--sample-idx", type=int, default=0, help="Test sample index for single-sample inspection (default: 0)")
    parser.add_argument("--random", action="store_true", help="Pick a random sample from the test split for single-sample inspection")
    parser.add_argument("--plots-dir", type=str, default="plots", help="Directory to save evaluation plots")
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found at: {args.checkpoint}")

    cfg = Config()
    device = get_device()
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    extra_config = ckpt["extra_config"]
    model_kwargs = extra_config["model_kwargs"]
    use_microstructure = extra_config.get("use_microstructure", False)
    use_news = extra_config["use_news"]

    pairs_arg = None if args.pair.upper() == "ALL" else [args.pair.upper()]
    _, _, test_dataset, loaded_pairs, feature_cols = build_multi_pair_datasets(
        pairs=pairs_arg,
        train_ratio=cfg.train_ratio,
        val_ratio=cfg.val_ratio,
        step_stride=cfg.step_stride,
        data_dir=cfg.data_dir,
        min_date=cfg.min_date,
        news_path=cfg.news_path if use_news else None,
        include_microstructure=use_microstructure,
    )
    assert len(feature_cols) == model_kwargs["input_dim"], (
        f"Feature count mismatch: dataset has {len(feature_cols)}, checkpoint expects {model_kwargs['input_dim']}"
    )

    model = ITransformer(**model_kwargs).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Loaded checkpoint: {args.checkpoint} (epoch {ckpt.get('epoch', '?')}) on {device} | use_microstructure={use_microstructure} | use_news={use_news} | arch={model_kwargs}")

    evaluate_full_test_split(
        model=model,
        test_dataset=test_dataset,
        loaded_pairs=loaded_pairs,
        batch_size=args.batch_size,
        num_workers=cfg.num_workers,
        device=device,
        plots_dir=args.plots_dir,
    )

    inspect_single_sample(
        model=model,
        test_dataset=test_dataset,
        sample_idx=args.sample_idx,
        use_random=args.random,
        device=device,
    )


if __name__ == "__main__":
    main()
