"""run script for fine-tuning a model."""
import yaml  # type: ignore
from aihero.research.config.schema import BatchInferenceJob, TrainingJob
from aihero.research.finetuning.infer import BatchInferenceJobRunner
from aihero.research.finetuning.train import TrainingJobRunner
from fire import Fire


def train(training_config_file: str = "/mnt/config/training/config.yaml", distributed_config_file: str = "") -> None:
    """Run Training."""
    training_config = TrainingJob.load(training_config_file)
    if distributed_config_file:
        with open(distributed_config_file) as f:
            distributed_training_config = yaml.safe_load(f)
    else:
        distributed_training_config = ""
    TrainingJobRunner(training_config, distributed_config=distributed_training_config).run()


def infer(batch_inference_config_file: str = "/mnt/config/batch_inference/config.yaml") -> None:
    """Run Batch Inference."""
    batch_inference_config = BatchInferenceJob.load(batch_inference_config_file)
    BatchInferenceJobRunner(batch_inference_config).run()


if __name__ == "__main__":
    Fire({"train": train, "infer": infer})
