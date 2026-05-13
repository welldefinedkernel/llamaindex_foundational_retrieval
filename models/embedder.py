from dataclasses import dataclass
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

class Embedder:
    def __init__(self, embed_model: str):
        self.model_id = embed_model
        self.model = SentenceTransformer(embed_model, model_kwargs={"dtype": "auto"})
        self.tokenizer = AutoTokenizer.from_pretrained(embed_model)
