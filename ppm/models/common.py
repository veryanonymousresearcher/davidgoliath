import torch
from torch import nn
from torch.nn import functional as F
import math



class TimePositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.0, base: float = 10000.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model
        # precompute frequency exponents like standard sinusoidal PE
        inv_freq = torch.exp(-math.log(base) * torch.arange(0, d_model, 2).float() / d_model)
        self.register_buffer("inv_freq", inv_freq)

        # optional learnable scale if you want to tune strength
        self.alpha = nn.Parameter(torch.tensor(1.0))

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D], t: [B, T] (continuous time, e.g., accumulated_time)
        # build sinusoidal features from t
        # shape handling to broadcast across feature dims
        theta = t.unsqueeze(-1) * self.inv_freq  # [B, T, D//2]
        pe = torch.zeros_like(x)
        pe[..., 0::2] = torch.sin(theta)
        pe[..., 1::2] = torch.cos(theta)
        x = x + self.alpha * pe
        return self.dropout(x)


class InLayer(nn.Module):
    def __init__(
        self,
        categorical_cols: list[str],
        categorical_sizes: dict[str, int],
        numerical_cols: list[str] = [],
        embedding_size: int = 768,
        strategy: str = "concat",
        padding_idx: int = 0,
    ):
        assert len(categorical_cols) == len(categorical_sizes)

        super(InLayer, self).__init__()

        self.embedding_size = embedding_size
        self.categorical_cols = categorical_cols
        self.categorical_sizes = categorical_sizes
        self.numerical_cols = numerical_cols
        self.padding_idx = padding_idx
        self.strategy = strategy

        self.total_features = len(categorical_cols) + len(numerical_cols)

        if strategy == "concat":
            in_embedding_size = embedding_size // 2
        elif strategy == "sum":
            in_embedding_size = embedding_size
        else:
            raise ValueError("Invalid strategy")

        # assert embedding size is divisible by the number of features
        # assert embedding_size % self.total_features == 0, "Embedding size must be divisible by the number of features"
        self.embedding_layers = nn.ModuleDict()
        for col in categorical_cols:
            self.embedding_layers[col] = nn.Embedding(
                categorical_sizes[col],
                in_embedding_size,
                padding_idx=padding_idx,
            )

        if len(numerical_cols) > 0:
            self.continuous_layer = nn.Linear(len(numerical_cols), in_embedding_size)

        self.layer_norm = nn.LayerNorm(embedding_size)
        self.init_params()

    def forward(self, cat_x, num_x=None):

        # cat features
        embedded_features = []
        for ix, name in enumerate(self.categorical_cols):
            # since we use OrderedDict, we can access the embedding layer by index
            embed = self.embedding_layers[name](cat_x[..., ix])
            embedded_features.append(embed)

        # num features
        if len(self.numerical_cols) > 0:
            projected_features = self.continuous_layer(num_x)

        # concatenate or sum
        if self.strategy == "concat":
            x = torch.cat(embedded_features, dim=-1)
            if len(self.numerical_cols) > 0:
                x = torch.cat(
                    [x, projected_features],
                    dim=-1,
                )
        elif self.strategy == "sum":
            x = sum(embedded_features)
            if len(self.numerical_cols) > 0:
                x += projected_features

        x = self.layer_norm(x)
        return x

    def init_params(self):
        for _, layer in self.embedding_layers.items():
            nn.init.xavier_uniform_(layer.weight)

        if len(self.numerical_cols) > 0:
            nn.init.xavier_uniform_(self.continuous_layer.weight)


class OutLayer(nn.Module):
    def __init__(self, input_size, output_size):
        super(OutLayer, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.layer_norm = nn.LayerNorm(input_size)
        self.linear = nn.Linear(input_size, output_size)
        self.init_params()

    def init_params(self):
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        x = self.layer_norm(x)
        return self.linear(x)


class PositionalEncoding(nn.Module):
    '''
    Hans: additional positional encoding for transformer, as used in the original transformer paper.
    Llama and Qwen do not use positional encoding, but this could be used to augment the embeddings nonetheless.
    Current implementation is just the positions for testing purposes, idea is to use timestamps later.
    '''
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000, numerical_cols: list[str] = []):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        x = x.transpose(0, 1)  # -> [seq_len, batch_size, embedding_dim]
        x = x + self.pe[: x.size(0)]
        x = self.dropout(x)
        return x.transpose(0, 1)  # -> [batch_size, seq_len, embedding_dim]