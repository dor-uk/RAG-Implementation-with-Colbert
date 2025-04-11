import chainlit as cl
import json
from fastembed import LateInteractionTextEmbedding
from qdrant_client import QdrantClient, models
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import CrossEncoder
from langchain_openai import ChatOpenAI
import numpy as np
import asyncio
import torch

embedding_model = LateInteractionTextEmbedding("jinaai/jina-colbert-v2")

json_path = "./chunked_documents.json"
with open(json_path, "r", encoding="utf-8") as file:
    chunked_data = json.load(file)


docs = [chunk["content"] for chunk in chunked_data]


embedding_file = "colbert_embeddings.npy"
metadata_file = "colbert_metadata.json"
loaded_embeddings = np.load(embedding_file, allow_pickle=True)
with open(metadata_file, "r", encoding="utf-8") as file:
    loaded_metadatas = json.load(file)

#print(f"Loaded {len(loaded_embeddings)} embeddings from {embedding_file}")

#print(loaded_embeddings[0].shape)

qdrant_client = QdrantClient(":memory:")

collection_name = "colbert_index"
vector_size = loaded_embeddings[0].shape[1]

# Create collection
if not qdrant_client.collection_exists(collection_name):
    qdrant_client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=vector_size,
            distance=Distance.COSINE,
            multivector_config=models.MultiVectorConfig(
                comparator=models.MultiVectorComparator.MAX_SIM
            ),
        ),
    )

# Load Reranker (MiniLM)
device = "cuda" if torch.cuda.is_available() else "cpu"
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device=device, max_length= 352)

# Load LLM for Final Generation
MODEL_BASE_URL = "https://9jb4dhts2y4rur-8000.proxy.runpod.net"
MODEL_NAME = "c4ai-command-r-plus-GPTQ"
llm = ChatOpenAI(
    openai_api_base=f"{MODEL_BASE_URL}/v1",
    model_name=MODEL_NAME,
    openai_api_key="sk-no-key",
    streaming=True,  
    temperature=0
)


points = [
    PointStruct(
        id=i,
        vector=[vec.tolist() for vec in loaded_embeddings[i]],  # Use precomputed embeddings
        payload=loaded_metadatas[i]  # Use precomputed metadata
    )
    for i in range(len(loaded_embeddings))
]

qdrant_client.upsert(collection_name=collection_name, points=points)

# **Retrieve Documents from Qdrant**
def search_query(query, output_path="./retrieved_chunks_colbert.json", top_k=20):
    
    query_embedding = list(embedding_model.embed([query]))[0]

    
    search_results = qdrant_client.query_points(
        collection_name=collection_name,
        query=query_embedding.tolist(),
        limit=top_k,
        with_payload=False
    ).points

    retrieved_chunks = []
    for i, result in enumerate(search_results):  
        retrieved_chunks.append({
            "rank": i + 1,
            "content": docs[result.id],  
            "score": result.score,
            #"metadata": result.payload  
        })

    
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(retrieved_chunks, file, ensure_ascii=False, indent=4)

    return retrieved_chunks


# **Rerank Retrieved Documents using MiniLM**
def rerank_results(query, retrieved_chunks):
    reranking_pairs = [(query, chunk["content"]) for chunk in retrieved_chunks]
    scores = reranker.predict(reranking_pairs,batch_size=8)  


    for i, chunk in enumerate(retrieved_chunks):
        chunk["rerank_score"] = float(scores[i])

    # Sort by highest rerank score
    reranked_chunks = sorted(retrieved_chunks, key=lambda x: x["rerank_score"], reverse=True)
    
    return reranked_chunks[:10]  



def generate_answer(query, retrieved_chunks):
    context = "\n\n".join([f"- {chunk['content']}" for chunk in retrieved_chunks])

    prompt = f"""
        You are an advanced AI assistant with expert-level knowledge Turkish Education System. 
        Given the following retrieved documents, answer the user's question accurately and concisely.
        You need to answer in only Turkish. You are not allowed to make up information. Just use the
        retrieved documents to create a sensible answer.

        ### **Retrieved Documents:**
        {context}

        ### **Question:**
        {query}

        ### **Answer:**
        """

    response = llm.invoke(prompt)
    return response.content


### Create Chainlit UI** ###
@cl.on_message
async def main(user_message: cl.Message):
    query = user_message.content

    await cl.Message(content="**Retrieving relevant documents...**").send()
    
    retrieved_chunks = search_query(query)
    

    await cl.Message(content="**Reranking results...**").send()
    reranked_chunks = rerank_results(query, retrieved_chunks)
    
    

    await cl.Message(content="**Generating final answer...**").send()
    final_answer = generate_answer(query, reranked_chunks)
    
    await cl.Message(content=f"**Final Answer:**\n\n{final_answer}").send()
