from dotenv import load_dotenv
from deepeval.synthesizer import Synthesizer
from deepeval.synthesizer.config import ContextConstructionConfig
from pathlib import Path

load_dotenv()

def create_synthetic_dataset(
        docs_dir: str, 
        output_dir: str, 
        max_contexts_per_document:int = 3,
        max_goldens_per_context: int = 2
    ) -> None:
    
    docs_path = Path(docs_dir).resolve()
    output_path = Path(output_dir).resolve()

    document_paths = [str(p) for p in docs_path.glob("*.docx")]
    if not document_paths:
        raise FileNotFoundError(f"No .docx files found in {docs_dir}")

    synthesizer = Synthesizer(cost_tracking=True)
    context_config = ContextConstructionConfig(
        max_contexts_per_document=max_contexts_per_document,
        chunk_size=1024,
        chunk_overlap=50
    )

    synthesizer.generate_goldens_from_docs(
        document_paths=document_paths,
        max_goldens_per_context=max_goldens_per_context,
        context_construction_config=context_config
    )
    synthesizer.save_as(
        file_type="json",
        directory=str(output_path),
        file_name="applets_4_0_synthetic",
    )

    print(f"DeepEval synthesis cost: ${synthesizer.synthesis_cost:.6f}")
