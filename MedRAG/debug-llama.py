from transformers import pipeline

pipe = pipeline(
    "text-generation",
    model="meta-llama/Llama-3.1-8B-Instruct",
    torch_dtype="auto",  # or torch.bfloat16 if your GPU supports it
    device_map="auto"
)

print(pipe("What is a large language model?", max_new_tokens=50))


# import torch
# print("CUDA available:", torch.cuda.is_available())
# print("Device name:", torch.cuda.get_device_name(0))
