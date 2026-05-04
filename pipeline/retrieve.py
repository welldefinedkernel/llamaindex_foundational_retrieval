from llama_index.core.vector_stores.types import VectorStoreQuery
from llama_index.vector_stores.milvus import MilvusVectorStore
from models.embedder import Embedder

class Retriever:
    def __init__(self, embed_model: Embedder, vector_store: MilvusVectorStore) -> None:
        self.embed_model = embed_model
        self.vector_store = vector_store

    def _embed_query(self, query: str) -> list[float]:
        return self.embed_model.model.encode(query, prompt_name="web_search_query").tolist()

    def retrieve_relevant_chunks(self, query: str, top_k: int = 5) -> list[str]:
        query_embedding = self._embed_query(query)
        result = self.vector_store.query(
            VectorStoreQuery(query_embedding=query_embedding, similarity_top_k=top_k)
        )

        nodes = result.nodes or []
        chunks: list[str] = []
        for node in nodes:
            content = node.get_content()
            if isinstance(content, str):
                chunks.append(content)

        return chunks