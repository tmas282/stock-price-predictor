import os
from dataclasses import dataclass
from dotenv import load_dotenv

# Load from .env if present
load_dotenv()


@dataclass
class Config:
    epochs: int = int(os.getenv("EPOCHS", 10))
    batch_size: int = int(os.getenv("BATCH_SIZE", 32))
    num_workers: int = int(os.getenv("NUM_WORKERS", 4))
    learning_rate: float = float(os.getenv("LEARNING_RATE", 7e-4))
    weight_decay: float = float(os.getenv("WEIGHT_DECAY", 1e-4))
    lr_min: float = float(os.getenv("LR_MIN", 1e-5))
    warmup_ratio: float = float(os.getenv("WARMUP_RATIO", 0.10))

    data_dir: str = os.getenv("DATA_DIR", "data")
    use_microstructure: bool = os.getenv("USE_MICROSTRUCTURE", "true").strip().lower() in {"1", "true", "yes"}
    use_news: bool = os.getenv("USE_NEWS", "true").strip().lower() in {"1", "true", "yes"}
    news_path: str = os.getenv("NEWS_PATH", "data/news_sentiment.parquet")
    min_date: str | None = os.getenv("MIN_DATE") or None
    plots_dir: str = os.getenv("PLOTS_DIR", "plots")
    checkpoint_dir: str = os.getenv("CHECKPOINT_DIR", "checkpoints")
    train_ratio: float = float(os.getenv("TRAIN_RATIO", 0.80))
    val_ratio: float = float(os.getenv("VAL_RATIO", 0.10))
    step_stride: int = int(os.getenv("STEP_STRIDE_DAYS", 7))
