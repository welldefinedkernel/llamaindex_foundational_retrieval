from deepeval.dataset import EvaluationDataset

def create_evaluation_dataset(
        data_path: str
    ) -> EvaluationDataset:
    
    dataset = EvaluationDataset()
    dataset.add_goldens_from_json_file(
        file_path=data_path, 
        input_key_name="input", 
        actual_output_key_name="actual_output",
        expected_output_key_name="expected_output"
    )

    return dataset