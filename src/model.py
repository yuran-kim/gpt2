import torch
import torch.nn as nn
import torch.nn.functional as F


class Head(nn.Module):
    """하나의 masked self-attention head."""

    def __init__(
        self,
        emb_dim: int,
        head_size: int,
        block_size: int,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.key = nn.Linear(emb_dim, head_size, bias=False)
        self.query = nn.Linear(emb_dim, head_size, bias=False)
        self.value = nn.Linear(emb_dim, head_size, bias=False)

        # 미래 위치를 가리기 위한 하삼각행렬
        self.register_buffer(
            "tril",
            torch.tril(torch.ones(block_size, block_size)),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, T, C)
        B, T, C = x.shape

        k = self.key(x)      # (B, T, head_size)
        q = self.query(x)    # (B, T, head_size)
        v = self.value(x)    # (B, T, head_size)

        # 각 토큰이 다른 토큰과 얼마나 관련 있는지 계산
        weights = q @ k.transpose(-2, -1)

        # 값이 지나치게 커지는 것을 막기 위한 scaling
        weights = weights * (k.size(-1) ** -0.5)

        # 미래 토큰을 볼 수 없도록 masking
        weights = weights.masked_fill(
            self.tril[:T, :T] == 0,
            float("-inf"),
        )

        # attention score를 확률처럼 변환
        weights = F.softmax(weights, dim=-1)
        weights = self.dropout(weights)

        # attention 가중치에 따라 value를 결합
        output = weights @ v

        return output


class MultiHeadAttention(nn.Module):
    """여러 attention head를 병렬로 실행한다."""

    def __init__(
        self,
        emb_dim: int,
        num_heads: int,
        block_size: int,
        dropout: float = 0.1,
    ):
        super().__init__()

        if emb_dim % num_heads != 0:
            raise ValueError("emb_dim은 num_heads로 나누어떨어져야 합니다.")

        head_size = emb_dim // num_heads

        self.heads = nn.ModuleList(
            [
                Head(
                    emb_dim=emb_dim,
                    head_size=head_size,
                    block_size=block_size,
                    dropout=dropout,
                )
                for _ in range(num_heads)
            ]
        )

        self.projection = nn.Linear(emb_dim, emb_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 각 head의 결과를 마지막 차원에서 연결
        output = torch.cat(
            [head(x) for head in self.heads],
            dim=-1,
        )

        output = self.projection(output)
        output = self.dropout(output)

        return output


class FeedForward(nn.Module):
    """각 토큰의 표현을 독립적으로 변환하는 신경망."""

    def __init__(
        self,
        emb_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(emb_dim, 4 * emb_dim),
            nn.ReLU(),
            nn.Linear(4 * emb_dim, emb_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class Block(nn.Module):
    """masked self-attention과 feed-forward로 구성된 Transformer block."""

    def __init__(
        self,
        emb_dim: int,
        num_heads: int,
        block_size: int,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.layer_norm1 = nn.LayerNorm(emb_dim)

        self.self_attention = MultiHeadAttention(
            emb_dim=emb_dim,
            num_heads=num_heads,
            block_size=block_size,
            dropout=dropout,
        )

        self.layer_norm2 = nn.LayerNorm(emb_dim)

        self.feed_forward = FeedForward(
            emb_dim=emb_dim,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # residual connection
        x = x + self.self_attention(self.layer_norm1(x))

        # residual connection
        x = x + self.feed_forward(self.layer_norm2(x))

        return x


class TinyGPT(nn.Module):
    """문자 단위 다음 토큰 예측 GPT 모델."""

    def __init__(
        self,
        vocab_size: int,
        block_size: int,
        emb_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.block_size = block_size

        self.token_embedding = nn.Embedding(
            vocab_size,
            emb_dim,
        )

        self.position_embedding = nn.Embedding(
            block_size,
            emb_dim,
        )

        self.blocks = nn.Sequential(
            *[
                Block(
                    emb_dim=emb_dim,
                    num_heads=num_heads,
                    block_size=block_size,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

        self.final_layer_norm = nn.LayerNorm(emb_dim)

        self.language_model_head = nn.Linear(
            emb_dim,
            vocab_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, T)
        B, T = x.shape

        if T > self.block_size:
            raise ValueError(
                f"입력 길이 {T}가 block_size {self.block_size}보다 큽니다."
            )

        positions = torch.arange(
            T,
            device=x.device,
        )

        # (B, T, C)
        token_embeddings = self.token_embedding(x)

        # (T, C)
        position_embeddings = self.position_embedding(positions)

        # broadcasting으로 (B, T, C) + (T, C)
        hidden = token_embeddings + position_embeddings

        hidden = self.blocks(hidden)
        hidden = self.final_layer_norm(hidden)

        # (B, T, vocab_size)
        logits = self.language_model_head(hidden)

        return logits