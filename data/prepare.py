from datasets import load_dataset
from llama_index.core import Document
from typing import Any, cast

def create_dataset_from_hf(dataset_name: str, subset_name: str, split: str):
    dataset = load_dataset(dataset_name, subset_name, split=split)
    seen = set()
    documents = []
    for row in dataset:
        row_dict = cast(dict[str, Any], row)
        for doc_text in row_dict['documents']:
            if doc_text not in seen:
                seen.add(doc_text)
                documents.append(Document(text=doc_text))
    return documents
