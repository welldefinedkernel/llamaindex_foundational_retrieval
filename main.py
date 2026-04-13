import os
from data.prepare import create_dataset_from_hf
from pipeline.chunk import chunk_documents

os.environ["HF_HUB_DISABLE_XET"] = "1"

def main():
    docs = create_dataset_from_hf("galileo-ai/ragbench", "techqa", "train")
    bigger_chunks = chunk_documents(docs, chunk_size=512, chunk_overlap=20)
    smaller_chunks = chunk_documents(docs, chunk_size=128, chunk_overlap=20)
    print(f"Number of documents: {len(docs)}")
    print(f"First document text: {docs[0].text}")
    print(f"Number of bigger chunks: {len(bigger_chunks)}")
    print(f"Number of smaller chunks: {len(smaller_chunks)}")

if __name__ == "__main__":
    main()
