import os
import torch
from torch import nn
from torch.optim.lr_scheduler import LRScheduler


def save_model(
    model: nn.Module,
    save_path: str,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: LRScheduler | None = None,
    epoch: int | None = None,
    metrics: dict | None = None,
    extra_config: dict | None = None,
) -> str:
    """
    Saves model weights, optimizer/scheduler state, and training metrics to disk.
    """
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "epoch": epoch,
        "metrics": metrics or {},
        "extra_config": extra_config or {},
    }

    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()

    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()

    torch.save(checkpoint, save_path)
    return os.path.abspath(save_path)


def load_model(
    model: nn.Module,
    checkpoint_path: str,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: LRScheduler | None = None,
    device: torch.device | str = "cpu",
) -> dict:
    """
    Loads weights and training state from a saved checkpoint into the model, optimizer, and scheduler.
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    model.to(device)
    return checkpoint
