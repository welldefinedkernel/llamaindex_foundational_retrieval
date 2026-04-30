import os

from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.schema import BaseNode
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.milvus import MilvusVectorStore
from models.embedder import Embedder


def create_vector_store(
    db_path: str,
    collection_name: str,
    embedding_dim: int,
    overwrite: bool = False,
) -> MilvusVectorStore:
    if not db_path.startswith("http://") and not db_path.startswith("https://"):
        db_path = os.path.abspath(db_path)
        os.makedirs(os.path.dirname(db_path), exist_ok=True)

    vector_store = MilvusVectorStore(
        uri=db_path, 
        collection_name=collection_name,
        dim=embedding_dim,               # Must match your HF model dimension (e.g., BGE-small is 384)
        overwrite=overwrite,
        similarity_metric="COSINE",
    )
    return vector_store

def load_vector_store(db_path: str, collection_name: str) -> MilvusVectorStore:
    if not db_path.startswith("http://") and not db_path.startswith("https://"):
        db_path = os.path.abspath(db_path)
        
    vector_store = MilvusVectorStore(
        uri=db_path, 
        collection_name=collection_name,
        dim=None,  # Dimension is not needed for loading an existing store
        overwrite=False,
        similarity_metric="COSINE",
    )
    return vector_store

def create_index_from_embedded_chunks(
        vector_store: MilvusVectorStore,
        embedded_chunks: list[BaseNode],
        embed_model: Embedder,
    ) -> VectorStoreIndex:
    assert len(embedded_chunks) > 0, "No embedded chunks provided to create the index."
    assert all(hasattr(chunk, "embedding") and chunk.embedding is not None for chunk in embedded_chunks), "All chunks must have embeddings before creating the index."
    embedding = embedded_chunks[0].embedding
    assert embedding is not None and len(embedding) == vector_store.dim, f"Embedding dimension of chunks ({len(embedding) if embedding else 0}) does not match vector store dimension ({vector_store.dim})."
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex(
        nodes=embedded_chunks, 
        storage_context=storage_context,
        show_progress=False,
        embed_model=HuggingFaceEmbedding(model_name=embed_model.model_id),
    )
    
    return index