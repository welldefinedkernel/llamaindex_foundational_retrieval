from datasets import load_dataset
from llama_index.core import VectorStoreIndex
from deepeval.models.base_model import DeepEvalBaseLLM

def main():
    ragbench = load_dataset("rungalileo/ragbench", "techqa", split="train")

if __name__ == "__main__":
    main()
