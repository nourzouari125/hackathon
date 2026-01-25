from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance

client = QdrantClient(":memory:")

client.create_collection(
    collection_name="products",
    vectors_config=VectorParams(
        size=1536,
        distance=Distance.COSINE
    )
)
