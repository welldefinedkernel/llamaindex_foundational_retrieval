from sentence_transformers import SentenceTransformer

def get_available_prompts(model_name):
    model = SentenceTransformer(model_name, model_kwargs={"dtype": "auto"})

    print(model.prompts)               
    print(list(model.prompts.keys()))  
    print(model.default_prompt_name)  