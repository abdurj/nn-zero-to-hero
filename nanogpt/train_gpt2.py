from dataclasses import dataclass
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x
    

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0, "n_embd must be divisible by n_head"
        self.c_attn = nn.Linear(config.n_embd, config.n_embd * 3)  # query, key, value
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_LIMIT = 1
        
        self.n_head = config.n_head
        self.n_embd = config.n_embd


        self.register_buffer('bias', torch.tril(torch.ones(config.block_size, config.block_size)
                                                .view(1, 1, config.block_size, config.block_size)))

    def forward(self, x):
        B, T, C = x.shape
        # get QKV projections
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=-1)
        
        # split into heads
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, n_head, T, C // n_head)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, n_head, T, C // n_head)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, n_head, T, C // n_head)
        
        # masked multi-head causal self-attention
        att = q @ k.transpose(-2, -1) * (1.0 / math.sqrt(k.size(-1)))  # (B, n_head, T, T)
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf')) # apply causal mask
        att = F.softmax(att, dim=-1) # softmax over last dim
        y = att @ v  # (B, n_head, T, C // n_head)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # (B, T, n_head, C // n_head) --> (B, T, C) # concatenate heads
        
        # output projection
        y = self.c_proj(y)
        
        return y

class Block(nn.Module):
    # Transformer Block
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)  # layer normalization
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)  # layer normalization
        self.mlp = MLP(config)  # feed-forward network

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))  # residual connection
        x = x + self.mlp(self.ln_2(x))  # residual connection
        return x


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),  # token embeddings
            wpe = nn.Embedding(config.block_size, config.n_embd),  # positional embeddings
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),  # transformer blocks
            ln_f = nn.LayerNorm(config.n_embd),  # final layer normalization
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # weight sharing 
        self.transformer.wte.weight = self.lm_head.weight  # share weights between token embeddings and output layer

        # init params
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            # initialize linear layers with normal distribution
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_LIMIT'):
                # control growth of residual stream based on number of layers
                std *= (2 * self.config.n_layer) ** -0.5
            
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            # initialize embeddings with normal distribution
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 model weights from huggingface"""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model
    
    
    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.config.block_size, "Cannot forward sequence of length %d, block size is only %d" % (T, self.config.block_size)

        # gpt forward
        # get token and position embeddings
        token_embeddings = self.transformer.wte(idx) # (T, n_embd)
        position_ids = torch.arange(T, dtype=torch.long, device=idx.device) # (T,)
        position_embeddings = self.transformer.wpe(position_ids) # (T, n_embd)
        x = token_embeddings + position_embeddings  # (B, T, n_embd)
        
        # forward transformer block
        for block in self.transformer.h:
            x = block(x) # (B, T, n_embd)
        # final layernorm
        x = self.transformer.ln_f(x) # (B, T, n_embd)
        
        # get logits through lm_head
        logits = self.lm_head(x) # (B, T, vocab_size)
        
        loss = None
        if targets is not None:
            # flatten logits and targets for cross-entropy loss
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss


class DataLoader:
    def __init__(self, B, T):
        self.B = B
        self.T = T
        
        with open('input.txt', 'r') as f:
            text = f.read()
        
        import tiktoken
        enc = tiktoken.get_encoding("gpt2")
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens)
        print(f"Total Tokens: {len(self.tokens)}")
        print(f"1 epoch = {len(self.tokens) // (B * T)} batches")

        self.current_pos = 0
        
    def next_batch(self):
        B, T = self.B, self.T
        step = B * T
        buf = self.tokens[self.current_pos:self.current_pos + step + 1] # +1 for the next token for targets
        x = buf[:-1].view(B, T)  # input data
        y = buf[1:].view(B, T)   # output data
        self.current_pos += step
        if self.current_pos + (step + 1) >= len(self.tokens):
            self.current_pos = 0 
        return x, y
            

# ----------------------------------------
import time

DEVICE = 'cpu'
if torch.cuda.is_available():
    DEVICE = 'cuda'
elif torch.backends.mps.is_available():
    DEVICE = 'mps'

torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)
print(f"Using device: {DEVICE}")

train_loader = DataLoader(B=16, T=1024)  # batch size 16, sequence length 1024

torch.set_float32_matmul_precision('high')

# model = GPT.from_pretrained('gpt2')
model = GPT(GPTConfig())
model = model.to(DEVICE)
model = torch.compile(model)


optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
for i in range(50):
    t0 = time.time()
    # sample a batch of data
    x, y = train_loader.next_batch()
    x, y = x.to(DEVICE), y.to(DEVICE)  # move to device
    
    # forward pass
    with torch.autocast(device_type=DEVICE, dtype=torch.bfloat16):
        logits, loss  = model(x, y)  # (B, T, vocab_size)
    
    # backward pass
    optimizer.zero_grad()
    loss.backward()
    
    # optimization step
    optimizer.step()
    
    torch.cuda.synchronize()
    t1 = time.time()
    dt = (t1 - t0) * 1000
    throughput = (train_loader.B * train_loader.T) / (t1 - t0)  # tokens per second
    print(f"Step {i}, Loss: {loss.item()}, Time: {dt:.2f} ms, Throughput: {throughput:.2f} tokens/s")

import sys
sys.exit(0)

# Sample Inference
num_return_sequences = 5
max_length = 30
model.eval()  # set to evaluation mode
model.to(DEVICE)  
import tiktoken
enc = tiktoken.get_encoding("gpt2")
tokens = enc.encode("Hello, I'm a language model,")
tokens = torch.tensor(tokens, dtype=torch.long, device=DEVICE)
tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1) # add batch dimension
x = tokens.to(DEVICE)

# generate
topk = 50
while x.size(1) < max_length:
    with torch.no_grad():
        logits = model(x)  # (B, T, vocab_size)
        logits = logits[:, -1, :] # discard everything but last time step
        probs = F.softmax(logits, dim=-1) # (B, vocab_size)
        topk_probs, topk_indices = torch.topk(probs, k=topk, dim=-1)
        # sample from top-k
        ix = torch.multinomial(topk_probs, num_samples=1)
        xcol = topk_indices.gather(-1, ix)
        x = torch.cat((x, xcol), dim=1)  # append to
        
        
for i in range(num_return_sequences):
    generated_tokens = x[i, :max_length].tolist()
    decoded = enc.decode(generated_tokens)
    print(f"> {decoded}")
