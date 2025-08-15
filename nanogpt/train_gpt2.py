from dataclasses import dataclass
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    block_size: int = 256
    vocab_size: int = 65
    n_layers: int = 4
    n_heads: int = 4
    n_embed: int = 128


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embed, 4 * config.n_embed)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embed, config.n_embed)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x
    

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embed % config.n_heads == 0, "n_embed must be divisible by n_heads"
        self.n_embed = config.n_embed
        self.n_heads = config.n_heads

        self.c_attn = nn.Linear(config.n_embed, config.n_embed * 3)  # query, key, value
        self.c_proj = nn.Linear(config.n_embed, config.n_embed)

        self.register_buffer('tril', torch.tril(torch.ones(config.block_size, config.block_size)
                                                .view(1, 1, config.block_size, config.block_size)))

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embed, dim=-1)
        # split into heads
        q = q.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)  # (B, n_heads, T, C // n_heads)
        k = k.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)  # (B, n_heads, T, C // n_heads)
        v = v.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)  # (B, n_heads, T, C // n_heads)
        # self-attention
        att = q @ k.transpose(-2, -1) * (1.0 / math.sqrt(k.size(-1)))  # (B, n_heads, T, T)
        att = att.masked_fill(self.tril[:, :, :T, :T] == 0, float('-inf')) # apply causal mask
        att = F.softmax(att, dim=-1) # softmax over last dim
        y = att @ v  # (B, n_heads, T, C // n_heads)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # (B, T, n_heads, C // n_heads) --> (B, T, C) # concatenate heads
        # output projection
        y = self.c_proj(y)
        return y

class Block(nn.Module):
    # Transformer Block
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embed)  # layer normalization
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embed)  # layer normalization
        self.mlp = MLP(config)  # feed-forward network

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))  # residual connection
        x = x + self.mlp(self.ln_2(x))  # residual connection
        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embed),  # token embeddings
            wpe = nn.Embedding(config.block_size, config.n_embed),  # positional embeddings
            h = nn.ModuleList([Block(config.n_embed, config.n_heads) for _ in range(config.n_layers)]),  # transformer blocks
            ln_f = nn.LayerNorm(config.n_embed),  # final layer normalization
        ))
        self.lm_head = nn.Linear(config.n_embed, config.vocab_size, bias=False)
