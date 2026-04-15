from llama_index.core.node_parser import SentenceSplitter
from sentence_transformers import SentenceTransformer

def chunk_documents(documents, chunk_size=512, chunk_overlap=0):
    splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    nodes = splitter.get_nodes_from_documents(documents)
    return nodes

def embed_chunks(chunks, embed_model):
    model = SentenceTransformer(embed_model, model_kwargs={"dtype": "auto"})
    for chunk in chunks:
        chunk.embedding = model.encode(chunk.get_content(), prompt_name="embedding")
    return chunks