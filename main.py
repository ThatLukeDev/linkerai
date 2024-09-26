from transformers import pipeline, BitsAndBytesConfig
import torch

model = pipeline(
    "text-generation",
    model="models/Llama-3.2-3B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

messages = [
    {"role": "user", "content": "Hi"},
]
outputs = model(
    messages,
    max_new_tokens=65536
)
response = outputs[0]["generated_text"][-1]["content"]
print(response)