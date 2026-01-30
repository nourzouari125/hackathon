import streamlit as st
import torch
from PIL import Image
from pathlib import Path

import embed as e
import qdrant_client
from qdrant_client.models import SearchParams

# -------------------------
# Qdrant config
# -------------------------
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "finmatch_products"

client = qdrant_client.QdrantClient(url=QDRANT_URL)

# -------------------------
# Page config
# -------------------------
st.set_page_config(
    page_title="FinMatch",
    page_icon="💳",
    layout="centered"
)

st.title("💳 FinMatch")
st.subheader("Context-Aware Financial Product Recommendation")

st.markdown(
    "Find the **best laptop** based on your **needs, budget, voice, or image**."
)

# -------------------------
# Inputs
# -------------------------
text_query = st.text_input("📝 Describe what you want")

image_file = st.file_uploader(
    "🖼️ Upload a laptop image (optional)",
    type=["png", "jpg", "jpeg"]
)

audio_file = st.file_uploader(
    "🎤 Upload voice request (optional)",
    type=["mp3", "wav"]
)

max_price = st.slider(
    "💰 Maximum budget",
    min_value=300,
    max_value=4000,
    step=100,
    value=1500
)

# -------------------------
# Search button
# -------------------------
if st.button("🔍 Find Best Match"):

    vectors = []
    weights = []

    # ---- TEXT ----
    if text_query.strip():
        v_text = e.txtVector(text_query)
        vectors.append(v_text)
        weights.append(0.4)

    # ---- IMAGE ----
    if image_file:
        image = Image.open(image_file)
        v_img = e.imVector(image)
        vectors.append(v_img)
        weights.append(0.4)

    # ---- AUDIO ----
    if audio_file:
        audio_path = Path("temp_audio.mp3")
        audio_path.write_bytes(audio_file.read())
        v_audio = e.audioVector(str(audio_path))
        vectors.append(v_audio)
        weights.append(0.2)

    if not vectors:
        st.warning("Please provide at least one input.")
        st.stop()

    # ---- FUSION ----
    query_vector = torch.zeros_like(vectors[0])
    for v, w in zip(vectors, weights):
        query_vector += v * w

    query_vector = query_vector / query_vector.norm(dim=1, keepdim=True)
    query_vector = query_vector.squeeze(0).tolist()

    # ---- SEARCH ----
    results = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vector,
        limit=5,
        with_payload=True,
        search_params=SearchParams(hnsw_ef=128)
    )

    # ---- DISPLAY RESULTS ----
    st.divider()
    st.subheader("📊 Recommended Products")

    if not results:
        st.info("No matching products found.")
    else:
        for r in results:
            payload = r.payload
            st.markdown(f"""
            **🖥️ {payload.get('name', 'Unknown')}**  
            Brand: {payload.get('brand', 'N/A')}  
            Price: 💲{payload.get('price', 'N/A')}  
            Similarity Score: `{round(r.score, 3)}`
            """)
            st.divider()