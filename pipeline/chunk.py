from pathlib import Path
from models.embedder import Embedder
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import Document
from docling.document_converter import DocumentConverter
from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from tqdm import tqdm

def chunk_documents(documents, embed_model: Embedder, chunk_size=512, chunk_overlap=0):
    splitter = SentenceSplitter(
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap,
        tokenizer=embed_model.tokenizer.tokenize,
    )
    nodes = splitter.get_nodes_from_documents(documents)
    seen = set()
    unique_nodes = []
    for node in nodes:
        content = node.get_content()
        if content not in seen:
            seen.add(content)
            unique_nodes.append(node)
    return unique_nodes


def chunk_documents_docling(input_dir: str, embed_model: Embedder, chunk_size=512):
    tokenizer = HuggingFaceTokenizer(tokenizer=embed_model.tokenizer, max_tokens=chunk_size)
    converter = DocumentConverter()
    chunker = HybridChunker(tokenizer=tokenizer)

    file_paths = [p for p in Path(input_dir).rglob("*") if p.is_file()]
    documents = []
    for result in tqdm(converter.convert_all(file_paths), desc="Docling converting & chunking", total=len(file_paths)):
        for chunk in chunker.chunk(result.document):
            text = chunker.contextualize(chunk)
            if text:
                documents.append(Document(text=text))

    seen = set()
    unique_documents = []
    for doc in documents:
        if doc.text not in seen:
            seen.add(doc.text)
            unique_documents.append(doc)
    return unique_documents


def embed_chunks(chunks: list, embed_model: Embedder):
    for chunk in tqdm(chunks):
        chunk.embedding = embed_model.model.encode(
            chunk.get_content(), 
            prompt_name="document", 
            show_progress_bar=False,
            normalize_embeddings=True
        )
    return chunks
