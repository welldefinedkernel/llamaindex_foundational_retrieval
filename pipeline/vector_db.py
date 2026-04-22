import os

from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.core.schema import BaseNode
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.milvus import MilvusVectorStore


def create_vector_store(db_name: str, collection_name: str, embedding_dim: int) -> MilvusVectorStore:
    db_path = os.path.abspath(db_name)
    os.makedirs(os.path.dirname(db_path), exist_ok=True)

    vector_store = MilvusVectorStore(
        uri=db_path, 
        collection_name=collection_name,
        dim=embedding_dim,               # Must match your HF model dimension (e.g., BGE-small is 384)
        overwrite=True,
        similarity_metric="COSINE",
    )
    return vector_store

def create_index_from_embedded_chunks(
        vector_store: MilvusVectorStore,
        embedded_chunks: list[BaseNode],
        embed_model: str,
    ) -> VectorStoreIndex:
    assert len(embedded_chunks) > 0, "No embedded chunks provided to create the index."
    assert all(hasattr(chunk, "embedding") and chunk.embedding is not None for chunk in embedded_chunks), "All chunks must have embeddings before creating the index."
    embedding = embedded_chunks[0].embedding
    assert embedding is not None and len(embedding) == vector_store.dim, f"Embedding dimension of chunks ({len(embedding) if embedding else 0}) does not match vector store dimension ({vector_store.dim})."
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex(
        nodes=embedded_chunks, 
        storage_context=storage_context,
        show_progress=True,
        embed_model=HuggingFaceEmbedding(model_name=embed_model),
    )
    
    return index