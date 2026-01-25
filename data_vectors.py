from products_database import products_database
from chunker import chunkie
import embedding 
from qdrant_db import client

point_id = 0

for product in products_database:
    text = f"""
    Name: {product['name']}
    Brand: {product['brand']}
    Category: {product['category']}
    Price: {product['price']}
    Description: {product['description']}
    """

    chunks = chunkie(text)

    for chunk in chunks:
        vector = embedding.txtVector(chunk)

        client.upsert(
            collection_name="products",
            points=[
                {
                    "id": point_id,
                    "vector": vector,
                    "payload": {
                        "product_name": product["name"],
                        "brand": product["brand"],
                        "price": product["price"],
                        "chunk_text": chunk
                    }
                }
            ]
        )

        point_id += 1

print(" Handshake Chonkie × Qdrant terminé avec succès.")
