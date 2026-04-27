from dataclasses import dataclass
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

@dataclass
class Embedder:
    def __init__(self, embed_model: str):
        self.model_id = embed_model
        self.model = SentenceTransformer(embed_model, model_kwargs={"dtype": "auto"})
        self.tokenizer = AutoTokenizer.from_pretrained(embed_model)

    
def load_embedder(model_name: str) -> Embedder:
    return Embedder(
        embed_model=model_name
    )