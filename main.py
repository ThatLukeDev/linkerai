from transformers import pipeline, MllamaForConditionalGeneration, AutoModelForCausalLM, AutoProcessor, BitsAndBytesConfig
import torch
from PIL import Image

MODELPATH = "models/imagemodel"

model = AutoModelForCausalLM.from_pretrained(
    MODELPATH,
    device_map="cuda", 
    trust_remote_code=True, 
    torch_dtype="auto", 
    _attn_implementation='flash_attention_2'    
)
processor = AutoProcessor.from_pretrained(
    MODELPATH,
    trust_remote_code=True, 
    num_crops=16,
)

images = [Image.new("RGB", (100, 100))]
imgtxt = "<|image_1|>\n"

messages = [
    {"role": "user", "content": imgtxt + "Hi"},
]

prompt = processor.tokenizer.apply_chat_template(
    messages, 
    tokenize=False, 
    add_generation_prompt=True
)

inputs = processor(prompt, images, return_tensors="pt").to("cuda:0") 

generation_args = { 
    "max_new_tokens": 32767, 
    "temperature": 0.0, 
    "do_sample": False, 
} 

generate_ids = model.generate(
    **inputs, 
    eos_token_id=processor.tokenizer.eos_token_id, 
    **generation_args
)

# remove input tokens 
generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
response = processor.batch_decode(
    generate_ids, 
    skip_special_tokens=True, 
    clean_up_tokenization_spaces=False)[0] 

print(response)