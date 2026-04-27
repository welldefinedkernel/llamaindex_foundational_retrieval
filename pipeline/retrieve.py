from llama_index.core.vector_stores.types import VectorStoreQuery, VectorStoreQueryResult
from llama_index.vector_stores.milvus import MilvusVectorStore
from models.embedder import Embedder
from sentence_transformers import SentenceTransformer

def embed_query(query: str, embed_model: Embedder) -> list[float]:
    return embed_model.model.encode(query, prompt_name="web_search_query").tolist()
    
def retrieve_relevant_chunks(
        query: str, 
        vector_store: MilvusVectorStore,
        embed_model: Embedder,
        top_k: int = 5, 
    ) -> VectorStoreQueryResult:
    query_embedding = embed_query(query, embed_model)
    result = vector_store.query(
        VectorStoreQuery(query_embedding=query_embedding, similarity_top_k=top_k)
    )
    
    return result