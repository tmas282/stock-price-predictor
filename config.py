import os
from dataclasses import dataclass
from dotenv import load_dotenv

# Load from .env if present
load_dotenv()


@dataclass
class Config:
    epochs: int = int(os.getenv("EPOCHS", 5))
    batch_size: int = int(os.getenv("BATCH_SIZE", 32))
    num_workers: int = int(os.getenv("NUM_WORKERS", 4))
    learning_rate: float = float(os.getenv("LEARNING_RATE", 5e-4))
    weight_decay: float = float(os.getenv("WEIGHT_DECAY", 1e-4))
    lr_min: float = float(os.getenv("LR_MIN", 1e-5))

    data_dir: str = os.getenv("DATA_DIR", "data")
    plots_dir: str = os.getenv("PLOTS_DIR", "plots")
    checkpoint_dir: str = os.getenv("CHECKPOINT_DIR", "checkpoints")
    train_ratio: float = float(os.getenv("TRAIN_RATIO", 0.70))
    val_ratio: float = float(os.getenv("VAL_RATIO", 0.15))
    step_stride: int = int(os.getenv("STEP_STRIDE", 4))
