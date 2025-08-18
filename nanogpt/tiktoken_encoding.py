import tiktoken
import os
import shutil

# Download the encoding first
encoding = tiktoken.get_encoding("gpt2")

# The cache is typically in your user directory
# On Linux/Mac: ~/.cache/tiktoken/
# On Windows: %LOCALAPPDATA%/tiktoken/
import platformdirs
cache_dir = platformdirs.user_cache_dir("tiktoken")
print(f"Cache directory: {cache_dir}")

# List and copy files
if os.path.exists(cache_dir):
    for file in os.listdir(cache_dir):
        print(f"Found: {file}")
        if file.endswith('.tiktoken'):
            shutil.copy(os.path.join(cache_dir, file), f"./{file}")