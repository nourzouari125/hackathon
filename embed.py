import numpy as np
from sentence_transformers import SentenceTransformer
from PIL import Image
import speech_recognition as sr
from pydub import AudioSegment
import os

# Modèle qui fonctionne SANS PyTorch lourd
model = SentenceTransformer('all-MiniLM-L6-v2')

def audioVector(audio_path):
    """Transcrit l'audio et retourne un embedding"""
    recognizer = sr.Recognizer()
    
    try:
        # Convertir MP3 en WAV si nécessaire
        if audio_path.endswith('.mp3'):
            audio = AudioSegment.from_mp3(audio_path)
            wav_path = audio_path.replace('.mp3', '.wav')
            audio.export(wav_path, format="wav")
            audio_path = wav_path
        
        # Reconnaissance vocale
        with sr.AudioFile(audio_path) as source:
            audio_data = recognizer.record(source)
            text = recognizer.recognize_google(audio_data, language='fr-FR')
        
        print("Transcription :", text)
        
        # Créer l'embedding
        embedding = model.encode(text, convert_to_numpy=True)
        embedding = embedding / np.linalg.norm(embedding)
        return embedding.reshape(1, -1)
        
    except Exception as e:
        print(f"Erreur audio: {e}")
        # Embedding par défaut
        embedding = model.encode("audio request", convert_to_numpy=True)
        embedding = embedding / np.linalg.norm(embedding)
        return embedding.reshape(1, -1)


def txtVector(text):
    """Retourne un embedding pour du texte"""
    embedding = model.encode(text, convert_to_numpy=True)
    embedding = embedding / np.linalg.norm(embedding)
    return embedding.reshape(1, -1)


def imVector(image):
    """Retourne un embedding pour une image (simplifié pour hackathon)"""
    # Pour l'instant, on utilise une description textuelle
    # Vous pouvez améliorer ça plus tard avec CLIP
    description = "laptop computer product image"
    embedding = model.encode(description, convert_to_numpy=True)
    embedding = embedding / np.linalg.norm(embedding)
    return embedding.reshape(1, -1)