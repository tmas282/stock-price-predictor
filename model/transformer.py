import torch
from torch import nn


class SelfSupervisedTransformer(nn.Module):
    """
    Input:
      x:    (B, L, feats) where L = 180 1D + 540 4H + 720 1H = 1440 tokens
      news: (B, 120, news_feats) optional 5-day hourly news grid
    Output: (B, pred_steps=7, out_feats=5) -> next 7 Daily [open, high, low, close, volume]
    """
    def __init__(
        self,
        feats: int = 26,
        news_feats: int = 12,
        out_feats: int = 5,
        pred_steps: int = 7,
        len_1d: int = 180,
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
        self.pred_steps = pred_steps
        self.len_1d = len_1d
        self.projection_input_layer = nn.Linear(feats, d_model)
        self.news_projection_input_layer = nn.Linear(news_feats, d_model)
        self.price_type_emb = nn.Parameter(torch.zeros(1, 1, d_model))
        self.news_type_emb = nn.Parameter(torch.zeros(1, 1, d_model))
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
    def forward(self, x: torch.Tensor, news: torch.Tensor | None = None):
        # src: full historical context (B, L, feats) [+ optional (B, 120, news_feats)]
        # tgt: the last `pred_steps` (7) Daily (1D) candles in `x[:, :len_1d, :]`

        x_proj_relu = self.relu(self.projection_input_layer(x)) + self.price_type_emb

        if news is not None:
            news_proj_relu = self.relu(self.news_projection_input_layer(news)) + self.news_type_emb
            src = torch.cat([x_proj_relu, news_proj_relu], dim=1)
        else:
            src = x_proj_relu

        tgt = x_proj_relu[:, self.len_1d - self.pred_steps : self.len_1d, :]

        out = self.transformer(src=src, tgt=tgt)  # (B, pred_steps, d_model)
        return self.projection_output_layer(out)  # (B, pred_steps, out_feats)
