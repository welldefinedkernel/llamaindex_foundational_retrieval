from llama_index.core.node_parser import SentenceSplitter

def chunk_documents(documents, chunk_size=512, chunk_overlap=0):
    splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    nodes = splitter.get_nodes_from_documents(documents)
    return nodes