import torch
from torch import nn
from torch.utils.data import DataLoader

from config import Config
from dataset import FEATURE_COLS, build_multi_pair_datasets
from model.transformer import SelfSupervisedTransformer
from utils.model_checkpoint import save_model
from utils.plot_info import (
    plot_pair_accuracies,
    plot_price_predictions,
    plot_training_metrics,
)
from utils.torch_device import get_device

CLOSE_PCT_IDX = FEATURE_COLS.index("feat_close_pct")


def train_one_epoch(model, dataloader, optimizer, criterion, device, epoch: int, log_interval: int = 250):
    model.train()
    total_loss = 0.0
    total_batches = len(dataloader)

    for batch_idx, batch in enumerate(dataloader, start=1):
        x = batch["x"].to(device)                      # (B, L, 14)
        target_next = batch["target_next"].to(device)  # (B, 14)

        optimizer.zero_grad()
        
        # Forward pass:
        # src = x (L = 1440 multi-timeframe context)
        # tgt = x[:, -1:, :] (last 1H candle)
        pred_next = model(x)

        loss = criterion(pred_next, target_next)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Periodically log batch progress
        if batch_idx % log_interval == 0 or batch_idx == total_batches:
            avg_loss = total_loss / batch_idx
            pct_done = (batch_idx / total_batches) * 100
            print(
                f"  [Epoch {epoch:02d} | Batch {batch_idx:>6}/{total_batches} ({pct_done:5.1f}%)] "
                f"Batch Loss: {loss.item():.6f} | Running Avg: {avg_loss:.6f}"
            )

    return total_loss / total_batches


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct_direction = 0
    total_samples = 0
    total_price_error = 0.0

    pair_correct = {}
    pair_total = {}

    with torch.no_grad():
        for batch in dataloader:
            x = batch["x"].to(device)
            target_next = batch["target_next"].to(device)
            prev_close = batch["prev_close"].to(device)
            actual_next_close = batch["actual_next_close"].to(device)
            pairs = batch["pair"]

            pred_next = model(x)
            loss = criterion(pred_next, target_next)
            total_loss += loss.item()

            pred_pct = pred_next[:, CLOSE_PCT_IDX]
            true_pct = target_next[:, CLOSE_PCT_IDX]

            # Reconstruct nominal price: P(t) = pct * P(t-1) + P(t-1)
            pred_next_close = pred_pct * prev_close + prev_close

            is_correct = (torch.sign(pred_pct) == torch.sign(true_pct)).cpu().numpy()
            correct_direction += is_correct.sum()
            total_price_error += torch.abs(pred_next_close - actual_next_close).sum().item()
            total_samples += x.size(0)

            for i, p in enumerate(pairs):
                pair_correct[p] = pair_correct.get(p, 0) + int(is_correct[i])
                pair_total[p] = pair_total.get(p, 0) + 1

    avg_loss = total_loss / len(dataloader)
    dir_acc = (correct_direction / total_samples) * 100 if total_samples > 0 else 0.0
    mae_price = total_price_error / total_samples if total_samples > 0 else 0.0

    return avg_loss, dir_acc, mae_price, pair_correct, pair_total


def main(cfg: Config | None = None):
    if cfg is None:
        cfg = Config()

    device = get_device()
    print(f"Device: {device}")
    print(f"Hyperparameters: epochs={cfg.epochs}, batch_size={cfg.batch_size}, num_workers={cfg.num_workers}, lr={cfg.learning_rate}, weight_decay={cfg.weight_decay}")
    print(f"Data settings: dir='{cfg.data_dir}', step_stride={cfg.step_stride}h, plots_dir='{cfg.plots_dir}', checkpoint_dir='{cfg.checkpoint_dir}', train_ratio={cfg.train_ratio}, val_ratio={cfg.val_ratio}")

    # 1. Build Hierarchical Multi-Timeframe Datasets (L = 1440)
    train_dataset, val_dataset, test_dataset, loaded_pairs = build_multi_pair_datasets(
        train_ratio=cfg.train_ratio,
        val_ratio=cfg.val_ratio,
        step_stride=cfg.step_stride,
        data_dir=cfg.data_dir
    )

    print(f"\nLoaded {len(loaded_pairs)} pairs.")
    print(f"Total training windows:   {len(train_dataset):,}")
    print(f"Total validation windows: {len(val_dataset):,}")
    print(f"Total test windows:       {len(test_dataset):,}")

    # 2. Multiprocess PyTorch DataLoaders
    use_pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=use_pin_memory,
        persistent_workers=(cfg.num_workers > 0),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=use_pin_memory,
        persistent_workers=(cfg.num_workers > 0),
    )

    # 3. Initialize Transformer
    model = SelfSupervisedTransformer(feats=len(FEATURE_COLS)).to(device)

    criterion = nn.SmoothL1Loss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay
    )

    # Automatically adapt log interval so it logs ~10 to 20 times per epoch
    log_interval = max(50, len(train_loader) // 20)

    # 4. Self-Supervised Training Loop
    epochs = cfg.epochs
    history_train_loss = []
    history_val_loss = []
    history_val_acc = []
    best_val_loss = float("inf")
    last_pair_correct: dict[str, int] = {}
    last_pair_total: dict[str, int] = {}

    print(f"\nStarting multi-pair self-supervised training for {epochs} epochs (Logging every {log_interval} batches)...")
    for epoch in range(1, epochs + 1):
        print(f"\n--- Epoch [{epoch:02d}/{epochs:02d}] ---")
        train_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            epoch=epoch,
            log_interval=log_interval
        )
        val_loss, val_acc, val_mae, last_pair_correct, last_pair_total = evaluate(model, val_loader, criterion, device)
        
        history_train_loss.append(train_loss)
        history_val_loss.append(val_loss)
        history_val_acc.append(val_acc)

        print(
            f"End of Epoch {epoch:02d} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f} | "
            f"Val Directional Acc: {val_acc:.2f}% | "
            f"Val Price MAE: {val_mae:.5f}"
        )

        # Save checkpoint after every epoch
        epoch_path = f"{cfg.checkpoint_dir}/model_epoch_{epoch:02d}.pt"
        metrics_dict = {
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_mae": val_mae,
        }
        save_model(
            model=model,
            save_path=epoch_path,
            optimizer=optimizer,
            epoch=epoch,
            metrics=metrics_dict,
        )
        print(f"  [Checkpoint] Saved -> {epoch_path}")

        # Save copy if best validation loss achieved
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = f"{cfg.checkpoint_dir}/best_model.pt"
            save_model(
                model=model,
                save_path=best_path,
                optimizer=optimizer,
                epoch=epoch,
                metrics=metrics_dict,
            )
            print(f"  [Checkpoint] * New best validation loss! Saved -> {best_path}")

    # 5. Per-Pair Directional Accuracy Summary
    print("\n--- Per-Pair Validation Performance (Sample) ---")
    pair_accuracies = {}
    for pair in loaded_pairs:
        total = last_pair_total.get(pair, 0)
        corr = last_pair_correct.get(pair, 0)
        acc = (corr / total) * 100 if total > 0 else 50.0
        pair_accuracies[pair] = acc

    for pair in loaded_pairs[:8]:
        print(f"  {pair:<8}: {pair_accuracies[pair]:5.1f}% accuracy ({last_pair_total.get(pair, 0)} windows)")

    # 6. Single-Candle Multi-Pair Inference Demonstration
    model.eval()
    sample_batch = next(iter(val_loader))
    sample_x = sample_batch["x"].to(device)
    sample_prev_close = sample_batch["prev_close"].to(device)
    sample_actual_close = sample_batch["actual_next_close"].to(device)
    sample_actual_pct = sample_batch["target_next"][:, CLOSE_PCT_IDX].to(device)
    sample_pairs = sample_batch["pair"]

    with torch.no_grad():
        pred_vectors = model(sample_x)
        predicted_pcts = pred_vectors[:, CLOSE_PCT_IDX]
        predicted_closes = predicted_pcts * sample_prev_close + sample_prev_close

    print("\n--- Multi-Pair Inference Demonstration ---")
    for i in range(min(4, len(sample_pairs))):
        pair = sample_pairs[i]
        p_prev = sample_prev_close[i].item()
        p_pred = predicted_closes[i].item()
        p_act = sample_actual_close[i].item()
        pct_pred = predicted_pcts[i].item()
        pct_act = sample_actual_pct[i].item()

        print(f"[{pair}] P(t-1)={p_prev:.4f} | Pred pct={pct_pred:+.3%} -> Pred P(t)={p_pred:.4f} | Actual P(t)={p_act:.4f} ({pct_act:+.3%})")

    # 7. Generate & Save Diagnostic Plots
    print(f"\nGenerating and saving plots to '{cfg.plots_dir}'...")
    plot_training_metrics(
        train_losses=history_train_loss,
        val_losses=history_val_loss,
        val_accuracies=history_val_acc,
        save_dir=cfg.plots_dir,
    )
    plot_pair_accuracies(
        pair_accuracies=pair_accuracies,
        save_dir=cfg.plots_dir,
    )

    # Plot sample consecutive price predictions
    sample_size = min(60, len(sample_actual_close))
    plot_price_predictions(
        actual_prices=sample_actual_close[:sample_size].cpu().numpy(),
        predicted_prices=predicted_closes[:sample_size].cpu().numpy(),
        pair_name=sample_pairs[0] if sample_pairs else "Sample",
        save_dir=cfg.plots_dir,
    )
    print("Done!")


if __name__ == "__main__":
    main()
