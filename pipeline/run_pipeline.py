from models.embedder import Embedder
from pipeline.retrieve import Retriever
from pipeline.vector_db import load_vector_store

embedder = Embedder(embed_model="microsoft/harrier-oss-v1-0.6b")
vector_store = load_vector_store(
    db_path="http://localhost:19530", 
    collection_name="HID_docs"
)  
retriever = Retriever(
    embed_model=embedder, 
    vector_store=vector_store
)


def run_pipeline(query: str, top_k: int = 5):
    retrieval_results = retriever.retrieve_relevant_chunks(query=query, top_k=top_k)
    return retrieval_results
