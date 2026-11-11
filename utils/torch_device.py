from functools import lru_cache
import torch

@lru_cache
def get_device() -> torch.device | str:
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"