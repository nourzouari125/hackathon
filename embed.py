
import torch
from pathlib import Path 
from transformers import CLIPModel, CLIPProcessor
from PIL import Image
import librosa
import whisper
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

whisper_model = whisper.load_model("base")


def audioVector(audio_path):
    
    result = whisper_model.transcribe(audio_path)
    text = result["text"]

    print("Transcription :", text)

    
    text_inputs = processor(text=[text], return_tensors="pt", padding=True)
    text_emb = model.get_text_features(**text_inputs)

    
    text_emb = text_emb / text_emb.norm(dim=1, keepdim=True)

    return text_emb






def txtVector(text):
    
    text_inputs = processor(text=[text], return_tensors="pt", padding=True)
    text_emb = model.get_text_features(**text_inputs)
    text_emb = text_emb / text_emb.norm(dim=1, keepdim=True)
    
    return (text_emb)

def imVector(image):
    #image = Image.open("product.jpg")
    image_inputs = processor(images=image, return_tensors="pt")
    image_emb = model.get_image_features(**image_inputs)
    image_emb = image_emb / image_emb.norm(dim=1, keepdim=True)
    return (image_emb)



if _name__=="__main_":
    x1=0.4
    x2=0.4
    x3=0.2
    v0 = torch.zeros(1, 512)
    v1 = v0.clone()
    v2 = v0.clone()
    v3 = v0.clone()

   
    
    
    
    image = Image.open("product.jpg")
    
    BASE_DIR = Path(_file_).parent
    file_path = BASE_DIR / "rv.txt"

    if file_path.exists():
        with file_path.open() as f:
            line = f.readline()
            print(line)
        v2=txtVector(line)
        
        
        
        
    img_path = Path(_file_).parent / "product.jpg"

    if img_path.is_file():
        image = Image.open("product.jpg")
        v1=imVector(image)
        

        
        
    audio_path = Path(_file_).parent / "record.mp3"
    if audio_path.exists():
        audio, sr = librosa.load("record.mp3", sr=None)
        
        
        v3 = audioVector(audio)

    
    if torch.all(v1==v0):
        x2=x2+x1/2
        x3=x3+x1/2
        x1=0
    if torch.all(v2==v0):
        x1=x1+x2/2
        x3=x3+x2/2
        x2=0
    if torch.all(v3==v0):
        x1=x1+x3
        x3=0

    
    
    q=v1*x1+v2*x2+v3*x3
    q=q/q.norm(dim=1, keepdim=True)
    print(q)

    #JUST AN  EMBEDDING IDEA NO INSERTION INTO QDRANT YET JUST INTO NORMAL TXT  FILES TO TEST