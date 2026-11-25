import numpy as np
import torch
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader

from candles.daily import TARGET_CLOSE_IDX
from config import Config
from dataset import (
    LEN_1D,
    PRED_DAYS,
    TARGET_DIM,
    build_multi_pair_datasets,
)
from model.itransformer import ITransformer
from utils.model_checkpoint import save_model
from utils.plot_info import plot_training_metrics
from utils.torch_device import get_device


def train_one_epoch(
    model,
    dataloader,
    optimizer,
    criterion,
    device,
    epoch: int,
    scheduler=None,
    scaler=None,
    log_interval: int = 250,
):
    model.train()
    total_loss = 0.0
    total_batches = len(dataloader)

    for batch_idx, batch in enumerate(dataloader, start=1):
        x = batch["x"].to(device)                    # (B, 180, feats)
        target_dir = batch["target_dir"].to(device)  # (B, 7, 1) in {0.0, 1.0}

        optimizer.zero_grad()

        logits = model(x)                            # (B, 7, 1) raw logits
        loss = criterion(logits, target_dir.to(logits.dtype))

        if scaler is not None and scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()

        if batch_idx % log_interval == 0 or batch_idx == total_batches:
            avg_loss = total_loss / batch_idx
            pct_done = (batch_idx / total_batches) * 100
            cur_lr = optimizer.param_groups[0]["lr"]
            print(
                f"  [Epoch {epoch:02d} | Batch {batch_idx:>6}/{total_batches} ({pct_done:5.1f}%)] "
                f"BCE Loss: {loss.item():.6f} | Running Avg: {avg_loss:.6f} | LR: {cur_lr:.6g}"
            )

    return total_loss / total_batches


def evaluate(model, dataloader, criterion, device, conf_margin: float = 0.05):
    model.eval()
    total_loss = 0.0
    correct_direction = 0
    total_samples = 0

    tp = 0
    tn = 0
    fp = 0
    fn = 0

    hi_conf_correct = 0
    hi_conf_total = 0

    day_correct = np.zeros(PRED_DAYS, dtype=np.int64)
    day_total = np.zeros(PRED_DAYS, dtype=np.int64)

    with torch.no_grad():
        for batch in dataloader:
            x = batch["x"].to(device)
            target_dir = batch["target_dir"].to(device)  # (B, 7, 1)

            logits = model(x)                            # (B, 7, 1) raw logits
            loss = criterion(logits, target_dir.to(logits.dtype))
            total_loss += loss.item()

            probs = torch.sigmoid(logits.float()).squeeze(-1)   # (B, 7)
            targets = target_dir.float().squeeze(-1)            # (B, 7)
            preds = (probs >= 0.5).float()

            pred_up = (preds == 1.0)
            true_up = (targets == 1.0)
            tp += int((pred_up & true_up).sum().item())
            tn += int((~pred_up & ~true_up).sum().item())
            fp += int((pred_up & ~true_up).sum().item())
            fn += int((~pred_up & true_up).sum().item())

            is_correct = (preds == targets).cpu().numpy()       # (B, 7) bool
            correct_direction += int(is_correct.sum())
            total_samples += int(is_correct.size)

            day_correct += is_correct.sum(axis=0)
            day_total += is_correct.shape[0]

            hi_mask = ((probs - 0.5).abs() >= conf_margin).cpu().numpy()
            if hi_mask.any():
                hi_conf_correct += int(is_correct[hi_mask].sum())
                hi_conf_total += int(hi_mask.sum())

    avg_loss = total_loss / len(dataloader)
    dir_acc = (correct_direction / total_samples) * 100 if total_samples > 0 else 0.0
    pred_up_pct = ((tp + fp) / total_samples) * 100 if total_samples > 0 else 0.0
    true_up_pct = ((tp + fn) / total_samples) * 100 if total_samples > 0 else 0.0
    up_recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    dn_recall = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    bal_acc = 0.5 * (up_recall + dn_recall) * 100
    hi_conf_acc = (hi_conf_correct / hi_conf_total) * 100 if hi_conf_total > 0 else 0.0
    hi_conf_cov = (hi_conf_total / total_samples) * 100 if total_samples > 0 else 0.0
    day_accs = [(day_correct[d] / day_total[d]) * 100 if day_total[d] > 0 else 0.0 for d in range(PRED_DAYS)]

    return (
        avg_loss,
        dir_acc,
        bal_acc,
        pred_up_pct,
        true_up_pct,
        hi_conf_acc,
        hi_conf_cov,
        day_accs,
    )


def main(cfg: Config | None = None):
    if cfg is None:
        cfg = Config()

    device = get_device()
    print(f"Device: {device}")
    print(
        f"Hyperparameters: epochs={cfg.epochs}, batch_size={cfg.batch_size}, "
        f"num_workers={cfg.num_workers}, lr={cfg.learning_rate}, "
        f"weight_decay={cfg.weight_decay}, lr_min={cfg.lr_min}, warmup_ratio={cfg.warmup_ratio}"
    )
    print(
        f"Data settings: dir='{cfg.data_dir}', use_microstructure={cfg.use_microstructure}, "
        f"use_news={cfg.use_news}, news_path='{cfg.news_path}', "
        f"min_date='{cfg.min_date}', step_stride={cfg.step_stride}d, plots_dir='{cfg.plots_dir}', "
        f"checkpoint_dir='{cfg.checkpoint_dir}', train_ratio={cfg.train_ratio}, val_ratio={cfg.val_ratio}"
    )

    # 1. Build Cross-Pair Daily (1D) Datasets (T = 180 -> S = 7 binary directions)
    news_path = cfg.news_path if cfg.use_news else None
    train_dataset, val_dataset, test_dataset, loaded_pairs, feature_cols = build_multi_pair_datasets(
        train_ratio=cfg.train_ratio,
        val_ratio=cfg.val_ratio,
        step_stride=cfg.step_stride,
        data_dir=cfg.data_dir,
        min_date=cfg.min_date,
        news_path=news_path,
        include_microstructure=cfg.use_microstructure,
    )

    print(f"\nLoaded {len(loaded_pairs)} pairs | Input features: {len(feature_cols)}")
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

    # 3. Initialize iTransformer (out_dim=1 raw logits per day -> BCEWithLogitsLoss)
    model_kwargs = {
        "input_dim": len(feature_cols),
        "out_dim": TARGET_DIM,
        "T_in_len": LEN_1D,
        "S_out_len": PRED_DAYS,
        "n_layers": 2,
        "token_emb": 128,
        "input_projection": 32,
        "ffnn_dim": 256,
        "nhead": 4,
        "dropout": 0.5
    }
    model = ITransformer(**model_kwargs).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )

    total_steps = max(1, cfg.epochs * len(train_loader))
    warmup_steps = int(total_steps * cfg.warmup_ratio)
    if warmup_steps > 0 and warmup_steps < total_steps:
        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=1e-3,
            end_factor=1.0,
            total_iters=warmup_steps,
        )
        cosine_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=total_steps - warmup_steps,
            eta_min=cfg.lr_min,
        )
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_steps],
        )
    else:
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=total_steps,
            eta_min=cfg.lr_min,
        )

    print(f"LR Schedule: {warmup_steps:,} warmup steps -> {total_steps - warmup_steps:,} cosine decay steps (total={total_steps:,})")

    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        scaler = None
    elif torch.cuda.is_available():
        scaler = torch.amp.GradScaler("cuda", enabled=True)
    else:
        scaler = None

    log_interval = max(50, len(train_loader) // 20)

    # 4. Training Loop
    epochs = cfg.epochs
    history_train_loss = []
    history_val_loss = []
    history_val_acc = []
    history_lr = []
    best_val_loss = float("inf")

    extra_config = {
        "model_kwargs": model_kwargs,
        "use_microstructure": cfg.use_microstructure,
        "use_news": cfg.use_news,
        "feature_cols": feature_cols,
    }

    print(f"\nStarting multi-pair iTransformer BCE direction training for {epochs} epochs (Logging every {log_interval} batches)...")
    for epoch in range(1, epochs + 1):
        start_lr = optimizer.param_groups[0]["lr"]

        print(f"\n--- Epoch [{epoch:02d}/{epochs:02d}] (Start LR: {start_lr:.6g}) ---")
        train_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            epoch=epoch,
            scheduler=scheduler,
            scaler=scaler,
            log_interval=log_interval,
        )
        end_lr = optimizer.param_groups[0]["lr"]
        val_loss, val_acc, bal_acc, pred_up_pct, true_up_pct, hi_acc, hi_cov, day_accs = evaluate(
            model, val_loader, criterion, device
        )

        history_train_loss.append(train_loss)
        history_val_loss.append(val_loss)
        history_val_acc.append(val_acc)
        history_lr.append(end_lr)

        day_str = " ".join([f"D{d + 1}:{acc:.1f}%" for d, acc in enumerate(day_accs)])
        print(
            f"End of Epoch {epoch:02d} | "
            f"Train BCE: {train_loss:.6f} | "
            f"Val BCE: {val_loss:.6f} | "
            f"Val Acc: {val_acc:.2f}% | "
            f"Bal Acc: {bal_acc:.2f}% | "
            f"Pred UP: {pred_up_pct:.1f}% (Actual UP: {true_up_pct:.1f}%) | "
            f"Hi-Conf(|p-0.5|>=0.05): {hi_acc:.2f}% ({hi_cov:.1f}% cov) | "
            f"LR: {start_lr:.6g} -> {end_lr:.6g}"
        )
        print(f"  Per-Horizon Val Acc: {day_str}")

        epoch_path = f"{cfg.checkpoint_dir}/model_epoch_{epoch:02d}.pt"
        metrics_dict = {
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "bal_acc": bal_acc,
            "pred_up_pct": pred_up_pct,
            "true_up_pct": true_up_pct,
            "hi_conf_acc": hi_acc,
            "hi_conf_cov": hi_cov,
            "day_accs": day_accs,
            "lr": end_lr,
        }
        save_model(
            model=model,
            save_path=epoch_path,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            metrics=metrics_dict,
            extra_config=extra_config,
        )
        print(f"  [Checkpoint] Saved -> {epoch_path}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = f"{cfg.checkpoint_dir}/best_model.pt"
            save_model(
                model=model,
                save_path=best_path,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                metrics=metrics_dict,
                extra_config=extra_config,
            )
            print(f"  [Checkpoint] * New best validation BCE! Saved -> {best_path}")

    # 5. 7-Day Multi-Pair Direction Inference Demonstration
    model.eval()
    sample_batch = next(iter(val_loader))
    sample_x = sample_batch["x"].to(device)
    sample_prev_close = sample_batch["prev_close"].to(device)
    sample_actual_close = sample_batch["actual_next_ohlcv"][:, :, TARGET_CLOSE_IDX].to(device)
    sample_targets = sample_batch["target_dir"].squeeze(-1)  # (B, 7)
    sample_pairs = sample_batch["pair"]

    with torch.no_grad():
        probs = torch.sigmoid(model(sample_x).float()).squeeze(-1).cpu()  # (B, 7)
        preds = (probs >= 0.5).int()
        actual_pcts = (sample_actual_close - sample_prev_close.unsqueeze(1)) / sample_prev_close.unsqueeze(1)

    print("\n--- Multi-Pair Direction Inference (D+1 -> D+7 Cumulative P(UP)) ---")
    for i in range(min(4, len(sample_pairs))):
        pair = sample_pairs[i]
        p_prev = sample_prev_close[i].item()
        prob_str = " ".join([f"{probs[i, d].item():.2f}" for d in range(PRED_DAYS)])
        pred_bits = "".join([str(preds[i, d].item()) for d in range(PRED_DAYS)])
        true_bits = "".join([str(int(sample_targets[i, d].item())) for d in range(PRED_DAYS)])
        print(
            f"[{pair}] P(t)={p_prev:.4f} | P(UP)=[{prob_str}] | "
            f"Pred={pred_bits} Actual={true_bits} | D+7 Move={actual_pcts[i, -1].item():+.2%}"
        )

    # 6. Generate & Save Diagnostic Plots
    print(f"\nGenerating and saving plots to '{cfg.plots_dir}'...")
    plot_training_metrics(
        train_losses=history_train_loss,
        val_losses=history_val_loss,
        val_accuracies=history_val_acc,
        learning_rates=history_lr,
        save_dir=cfg.plots_dir,
    )

    print("Done!")


if __name__ == "__main__":
    main()
