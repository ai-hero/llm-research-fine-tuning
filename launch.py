"""run script for fine-tuning a model."""
from aihero.research.config.schema import BatchInferenceJob, TrainingJob
from aihero.research.finetuning.infer import BatchInferenceJobRunner
from aihero.research.finetuning.train import TrainingJobRunner
from fire import Fire


def train(training_config_file: str) -> None:
    """Run Training."""
    training_config = TrainingJob.load(training_config_file)
    TrainingJobRunner(training_config).run()


def infer(batch_inference_config_file: str) -> None:
    """Run Batch Inference."""
    batch_inference_config = BatchInferenceJob.load(batch_inference_config_file)
    BatchInferenceJobRunner(batch_inference_config).run()


if __name__ == "__main__":
    Fire({"train": train, "infer": infer})
