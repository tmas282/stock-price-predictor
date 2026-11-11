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
    Saves a PyTorch model state_dict along with optional optimizer, scheduler state, and training metadata.

    Args:
        model: PyTorch nn.Module to save
        save_path: File path where the checkpoint will be saved (e.g. 'checkpoints/model_epoch_01.pt')
        optimizer: Optional optimizer whose state_dict will be saved for training resumption
        scheduler: Optional learning rate scheduler whose state_dict will be saved
        epoch: Current epoch number
        metrics: Dictionary of evaluation metrics (e.g. {'val_loss': 0.012, 'val_acc': 52.3})
        extra_config: Optional dictionary with model or data configurations

    Returns:
        The resolved absolute path where the file was saved.
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
    Loads weights from a saved checkpoint into the provided model (and optional optimizer/scheduler).

    Args:
        model: PyTorch nn.Module instance into which the weights will be loaded
        checkpoint_path: Path to the .pt or .pth checkpoint file
        optimizer: Optional optimizer instance to restore optimizer state for resumed training
        scheduler: Optional scheduler instance to restore scheduler state for resumed training
        device: Target torch device ('cpu', 'cuda', etc.)

    Returns:
        The checkpoint dictionary containing metadata (epoch, metrics, extra_config).
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Supports both structured checkpoint dicts and raw state_dicts
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if scheduler is not None and "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    return checkpoint if isinstance(checkpoint, dict) else {"model_state_dict": checkpoint}
