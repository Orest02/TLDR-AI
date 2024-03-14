# Use a pipeline as a high-level helper
import torch
from transformers import pipeline

pipe = pipeline("text-generation", model="bardsai/jaskier-7b-dpo-v5.6", torch_dtype=torch.float16, device_map="auto", max_length=200)

text = "How do I exit vim?"

print(pipe(text))
