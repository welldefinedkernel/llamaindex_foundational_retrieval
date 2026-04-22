from llama_index.core.node_parser import SentenceSplitter
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
from tqdm import tqdm

def chunk_documents(documents, embed_model: str, chunk_size=512, chunk_overlap=0):
    hf_tok = AutoTokenizer.from_pretrained(embed_model)
    splitter = SentenceSplitter(
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap,
        tokenizer=hf_tok.tokenize,
    )
    nodes = splitter.get_nodes_from_documents(documents)
    return nodes

def embed_chunks(chunks: list, embed_model: str):
    model = SentenceTransformer(embed_model, model_kwargs={"dtype": "auto"})
    for chunk in tqdm(chunks):
        chunk.embedding = model.encode(chunk.get_content(), prompt_name="document")
    return chunks