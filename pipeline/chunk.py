from models.embedder import Embedder
from llama_index.core.node_parser import SentenceSplitter
from tqdm import tqdm

def chunk_documents(documents, embed_model: Embedder, chunk_size=512, chunk_overlap=0):
    splitter = SentenceSplitter(
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap,
        tokenizer=embed_model.tokenizer.tokenize,
    )
    nodes = splitter.get_nodes_from_documents(documents)
    return nodes

def embed_chunks(chunks: list, embed_model: Embedder):
    for chunk in tqdm(chunks):
        chunk.embedding = embed_model.model.encode(chunk.get_content(), prompt_name="document")
    return chunks