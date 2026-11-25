import torch
from torch import nn

class EncoderBlock(nn.Module):
    def __init__(
        self,
        emb_dim: int,
        ffnn_dim = 256,
        num_heads: int = 8,
        dropout = 0.,
    ):
        super().__init__()
        self.ffnn_dim=ffnn_dim
        self.multihead_attention = nn.MultiheadAttention(emb_dim, num_heads, batch_first=True, dropout=dropout)
        self.layer_norm1 = nn.LayerNorm(normalized_shape=emb_dim)
        self.ffnn1 = nn.Linear(emb_dim, self.ffnn_dim)
        self.ffnn2 = nn.Linear(self.ffnn_dim, emb_dim)
        self.layer_norm2 = nn.LayerNorm(normalized_shape=emb_dim)
        self.activation = nn.LeakyReLU()
        self.dropout = nn.Dropout(p=dropout)
    
    @torch.autocast(device_type="cuda", enabled=torch.cuda.is_available(),
                    dtype=torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16)
    def forward(self, x: torch.Tensor):
        x_attn,_ = self.multihead_attention(x, x, x, need_weights=False)
        x_norm1 = self.layer_norm1(x + x_attn)
        x_ffnn_1 = self.ffnn1(x_norm1)
        x_ffnn_a = self.activation(x_ffnn_1)
        x_ffnn_2 = self.ffnn2(x_ffnn_a)
        x_ffnn_d = self.dropout(x_ffnn_2)
        x_norm2 = self.layer_norm2(x_norm1 + x_ffnn_d)
        return x_norm2
