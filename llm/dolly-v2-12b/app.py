import os
import sys
import torch
from transformers import pipeline, set_seed

# Set the cache directory path
# cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model')
# cache_dir = os.path.join(sys.prefix, 'models/dolly-v2-7b')

local_model_dir = 'M:/dev/python-multi/models/llm/dolly-v2-7b'

# Initialize the pipeline with the cache directory
generate_text = pipeline(
    model=local_model_dir,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device=0,
    # device_map="auto",
    # cache_dir=cache_dir,
)

# Set a random seed for reproducibility (optional)
set_seed(42)

generated_text = generate_text("hi")
print(generated_text)

# wsl source \\wsl.localhost\Ubuntu\home\mlnima\myenv\bin\activate && python your_script.py
# \\wsl$
