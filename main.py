from transformers import pipeline, MllamaForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
import torch
from PIL import Image

# model = pipeline(
#     "text-generation",
#     model="models/textmodel",
#     torch_dtype=torch.bfloat16,
#     device_map="auto",
# )
# 
# messages = [
#     {"role": "user", "content": "Hi"},
# ]
# outputs = model(
#     messages,
#     max_new_tokens=65536,
# )
# response = outputs[0]["generated_text"][-1]["content"]
# print(response)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = MllamaForConditionalGeneration.from_pretrained(
    "models/imagemodel",
    quantization_config=bnb_config,
)
processor = AutoProcessor.from_pretrained("models/imagemodel")

messages = [
    {"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": "Hi"}
    ]}
]

input_text = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
)
inputs = processor(Image.new("RGB", (100, 100)), input_text, return_tensors="pt").to(model.device)
output = model.generate(**inputs, max_new_tokens=65536)
print(processor.decode(output[0][inputs["input_ids"].shape[-1]:]))