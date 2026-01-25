from transformers import CLIPModel, CLIPProcessor
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load once
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def embed(combined_text):
     
    inputs = processor(text=combined_text, return_tensors="pt", padding=True).to(device)
    
    with torch.no_grad():
        text_embedding = model.get_text_features(**inputs)
    
    # Normalize
    text_embedding = torch.nn.functional.normalize(text_embedding, p=2, dim=1)
    
    # Since we only have one text now, remove batch dimension
    return text_embedding[0]