# -*- coding: utf-8 -*-
"""
Model definition module.

Contains only the architecture of the transformer used for MIDI token generation.
No training, data loading, or inference logic lives here – those are delegated to
`training.py` and `inference.py` respectively.
"""

import torch
import torch.nn as nn
from torch.nn import functional as F
import config


class Head(nn.Module):
    def __init__(self, head_size: int, n_embed: int, block_size: int, dropout: float):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        # Causal mask – same shape as original implementation
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        # Scaled dot‑product attention
        wei = q @ k.transpose(-2, -1) * C ** -0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        return wei @ v


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads: int, n_embed: int, block_size: int, dropout: float):
        super().__init__()
        head_size = config.N_EMBED // num_heads
        self.heads = nn.ModuleList(
            [Head(head_size, n_embed, block_size, dropout) for _ in range(num_heads)]
        )
        self.proj = nn.Linear(n_embed, n_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))


class FeedForward(nn.Module):
    def __init__(self, n_embed: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.ReLU(),
            nn.Linear(4 * n_embed, n_embed),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, num_heads: int, n_embed: int, block_size: int, dropout: float):
        super().__init__()
        self.sa = MultiHeadAttention(num_heads, n_embed, block_size, dropout)
        self.ffwd = FeedForward(n_embed, dropout)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class BigramLanguageModel(nn.Module):
    """Decoder‑only transformer used for MIDI token generation.

    The implementation mirrors the original `train_model.py` exactly – only the
    surrounding concerns have been stripped away.
    """

    def __init__(self, vocab_size: int, *, n_embed: int = config.N_EMBED,
                 block_size: int = config.BLOCK_SIZE, n_head: int = config.N_HEAD,
                 n_layer: int = config.N_LAYER, dropout: float = config.DROPOUT):
        super().__init__()
        self.block_size = block_size
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        self.blocks = nn.Sequential(
            *[TransformerBlock(n_head, n_embed, block_size, dropout) for _ in range(n_layer)]
        )
        self.ln_f = nn.LayerNorm(n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size, bias=False)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        """Autoregressive generation.

        Mirrors the original method – repeatedly feeds the last `block_size`
        tokens back into the model and samples the next token.
        """
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size :]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


def create_model(vocab_size: int, device: torch.device | str = config.DEVICE) -> BigramLanguageModel:
    """Factory helper used by training and inference modules.

    Returns a model instance placed on the requested device.
    """
    model = BigramLanguageModel(vocab_size).to(device)
    return model
