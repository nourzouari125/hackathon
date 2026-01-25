import torch
from transformers import CLIPModel, CLIPProcessor
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def txtVector(text):
    
    text_inputs = processor(text=[text], return_tensors="pt", padding=True)
    text_emb = model.get_text_features(**text_inputs)
    text_emb = text_emb / text_emb.norm(dim=1, keepdim=True)
    
    return (text_emb)