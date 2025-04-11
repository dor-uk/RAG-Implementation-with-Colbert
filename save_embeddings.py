import numpy as np
import json
from qdrant_client import QdrantClient, models
from qdrant_client.models import Distance, VectorParams, PointStruct
from fastembed import LateInteractionTextEmbedding

embedding_model = LateInteractionTextEmbedding("jinaai/jina-colbert-v2")

json_path = "./chunked_documents.json"
with open(json_path, "r", encoding="utf-8") as file:
    chunked_data = json.load(file)


docs = [chunk["content"] for chunk in chunked_data]
document_metadatas = [
    {
        "doc_name": chunk["doc_name"],
        "chunk_id": chunk["chunk_id"],
        "chunk_size": chunk["chunk_size"]
    }
    for chunk in chunked_data
]

embeddings = list(embedding_model.embed(docs))

# Save embeddings to a file
embedding_file = "colbert_embeddings.npy"
metadata_file = "colbert_metadata.json"

# Convert multi-vector embeddings to lists and save
np.save(embedding_file, np.array(embeddings, dtype=object))  # Save embeddings
with open(metadata_file, "w", encoding="utf-8") as file:
    json.dump(document_metadatas, file, ensure_ascii=False, indent=4)

print(f"Embeddings saved to {embedding_file}")
print(f"Metadata saved to {metadata_file}")