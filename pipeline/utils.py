import json

from huggingface_hub import hf_hub_download


def get_available_prompts(model_name: str) -> dict:
    """Read prompt config from the Hub without downloading model weights."""
    config_path = hf_hub_download(
        repo_id=model_name,
        filename="config_sentence_transformers.json",
    )
    with open(config_path, "r") as f:
        cfg = json.load(f)

    prompts = cfg.get("prompts", {}) or {}
    print(prompts)
    print(list(prompts.keys()))
    return prompts