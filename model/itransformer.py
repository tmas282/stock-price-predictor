import torch
from torch import nn

from model.encoder_block import EncoderBlock
from model.projections import OutputProjection, ProjectionInput


class ITransformer(nn.Module):
    """
    Input:
      x:    (B, N_total_length, candle_i_dim)
    Output: (B, S_out_len, candle_o_dim)
    """
    def __init__(
        self,
        input_dim: int = 28,
        out_dim: int = 5,
        T_in_len: int = 180,
        token_emb: int = 256,
        S_out_len: int = 7,
        ffnn_dim: int = 1024,
        nhead: int = 8,
        n_layers: int = 4,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.projection_dim = 64
        self.token_emb = token_emb
        self.acti_layer = nn.LeakyReLU()
        
        self.proj_input_candle = ProjectionInput(input_dim, self.projection_dim)
        self.emb_series_to_var_tokens = nn.Linear(T_in_len, token_emb)

        self.transformer_layers = nn.ModuleList([EncoderBlock(token_emb, ffnn_dim=ffnn_dim, num_heads=nhead, dropout=dropout) for _ in range(n_layers)])

        self.var_tokens_to_emb_series = nn.Linear(token_emb, S_out_len)
        self.proj_output_candle = OutputProjection(out_dim, self.projection_dim)

    @torch.autocast(device_type="cuda", enabled=torch.cuda.is_available(),
                    dtype=torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16)
    def forward(self, x: torch.Tensor):
        """
        Input:
            x:    (B, N_total_length, candle_i_dim)
        Output: (S_out_len, candle_o_dim)
        """
        init_dim = x.ndim
        if init_dim == 2:
            x = x.unsqueeze(0)
        x_proj = self.proj_input_candle(x)
        x_hat = torch.transpose(x_proj, 1, 2)

        h_i = self.emb_series_to_var_tokens(x_hat)
        h_i = self.acti_layer(h_i)

        for transformer_layer in self.transformer_layers:
            h_i = transformer_layer(h_i)

        y_hat = self.var_tokens_to_emb_series(h_i)
        y_hat = self.acti_layer(y_hat)

        y = torch.transpose(y_hat, 1, 2)

        y = self.proj_output_candle(y)
        if init_dim == 2:
            y = y.squeeze(0)
        return y
