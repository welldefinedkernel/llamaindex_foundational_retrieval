import os
from data.prepare import create_dataset_from_hf

os.environ["HF_HUB_DISABLE_XET"] = "1"

def main():
    docs = create_dataset_from_hf("galileo-ai/ragbench", "techqa", "train")
    
if __name__ == "__main__":
    main()
