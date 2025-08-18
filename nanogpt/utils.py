import tiktoken
from tiktoken.load import load_tiktoken_bpe
import os

def load_offline_gpt2_encoding(tiktoken_file="gpt2_encoding.tiktoken"):
    """Load GPT-2 encoding from local tiktoken file"""
    if not os.path.exists(tiktoken_file):
        raise FileNotFoundError(f"Encoding file not found: {tiktoken_file}")
    
    # Pass the file path, not the contents
    mergeable_ranks = load_tiktoken_bpe(tiktoken_file)
    
    return tiktoken.Encoding(
        name="gpt2-offline",
        pat_str=r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""",
        mergeable_ranks=mergeable_ranks,
        special_tokens={"<|endoftext|>": 50256}
    )