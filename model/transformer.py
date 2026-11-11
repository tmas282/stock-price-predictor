import torch
from torch import nn


class SelfSupervisedTransformer(nn.Module):
    """
    Input: (B, L, feats)
    L = 180 Daily + 540 4-Hour + 720 1-Hour = 1440 tokens
    feats = 26 (cyclical time, relative returns, technical indicators)
    Output: (B, out_feats=5) -> next candle [open, high, low, close, volume]
    """
    def __init__(
        self,
        feats: int = 26,
        out_feats: int = 5,
        d_model: int = 64,
        nhead: int = 8,
        num_encoder_layers: int = 4,
        num_decoder_layers: int = 4,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "gelu",
        layer_norm_eps: float = 1e-5,
        batch_first: bool = True,
        norm_first: bool = False,
    ):
        super().__init__()
        self.projection_input_layer = nn.Linear(feats, d_model)
        self.relu = nn.ReLU()
        self.projection_output_layer = nn.Linear(d_model, out_feats)
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            layer_norm_eps=layer_norm_eps,
            batch_first=batch_first,
            norm_first=norm_first
        )

    @torch.autocast(device_type="cuda", enabled=torch.cuda.is_available(), 
                    dtype=torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16)
    def forward(self, x):
        # src: full historical context (B, L, feats)
        # tgt: the last candle in the context window (B, 1, feats)

        x_proj = self.projection_input_layer(x)
        x_proj_relu = self.relu(x_proj)

        src = x_proj_relu
        tgt = x_proj_relu[:, -1:, :]

        out = self.transformer(src=src, tgt=tgt)  # (B, 1, d_model)
        out_normal = self.projection_output_layer(out)  # (B, 1, out_feats)
        return out_normal.squeeze(1)  # (B, out_feats)