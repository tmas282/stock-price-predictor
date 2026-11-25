import torch
from torch import nn

class ProjectionInput(nn.Module):
    def __init__(
        self,
        feats: int = 28,
        emb_size: int = 512,
    ):
        super().__init__()
        self.proj_candle_linear = nn.Linear(feats, emb_size)
        self.emb = nn.Parameter(torch.zeros(1, 1, emb_size))
        self.acti_func = nn.LeakyReLU()
    
    @torch.autocast(device_type="cuda", enabled=torch.cuda.is_available(),
                    dtype=torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16)
    def forward(self, candle: torch.Tensor):
        return self.acti_func(self.proj_candle_linear(candle)) + self.emb

class OutputProjection(nn.Module):
    def __init__(
        self,
        feats: int = 12,
        emb_size: int = 512,
    ):
        super().__init__()
        self.proj_output_linear = nn.Linear(emb_size, feats)
    
    @torch.autocast(device_type="cuda", enabled=torch.cuda.is_available(),
                    dtype=torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16)
    def forward(self, x: torch.Tensor):
        return self.proj_output_linear(x)