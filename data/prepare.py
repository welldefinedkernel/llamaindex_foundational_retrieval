from datasets import load_dataset
from llama_index.core import Document
from typing import Any, cast

def create_dataset_from_hf(dataset_name: str, subset_name: str, split: str):
    dataset = load_dataset(dataset_name, subset_name, split=split)
    documents = []
    for row in dataset:
        row_dict = cast(dict[str, Any], row)
        for doc_text in row_dict['documents']:
            doc = Document(text=doc_text)
            documents.append(doc)
    return documents
