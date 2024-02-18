"""Custom callback for sampling from a LLM and reporting custom eval to WANDB."""
import random
from typing import Any

from datasets import Dataset
from transformers.integrations import WandbCallback
from trl import SFTTrainer

from aihero.research.finetuning.infer import BatchInferenceWithEval

MAX_NEW_TOKENS = 512


class LLMSampleCB(WandbCallback):  # type: ignore
    """Callback for sampling from a LLM and reporting custom eval to WANDB."""

    def __init__(
        self: "LLMSampleCB",
        trainer: SFTTrainer,
        task: str,
        test_split: Dataset,
        num_samples: int = 100,
        max_new_tokens: int = MAX_NEW_TOKENS,
        log_model: str = "checkpoint",
        run_tests_str: str = "",
        run_metrics_str: str = "",
    ):
        """Initialize the callback by extracting a few rows from the test split."""
        super().__init__()
        assert task == "completion", "Only completion task supported for now"
        self._log_model = log_model
        self.batch_inference = BatchInferenceWithEval(
            model=trainer.model,
            tokenizer=trainer.tokenizer,
            task=task,
            run_tests_str=run_tests_str,
            run_metrics_str=run_metrics_str,
            max_new_tokens=max_new_tokens,
        )

        # Sample a few rows from the test split to generate a table of predictions
        # for visual inspection a.k.a. spot checking
        # Randomly select indices for the samples
        selected_indices = random.sample(range(test_split.num_rows), num_samples)
        # Retrieve the selected samples from the dataset
        test_split_list = list(test_split)
        self.sample_split = []
        for i in selected_indices:
            self.sample_split.append(test_split_list[i])
        self.sample_split = Dataset.from_list(self.sample_split)

    def initialize(self: "LLMSampleCB") -> None:
        """Generate initial predictions for the sample split and log them to WANDB."""
        self._wandb.init()

        _, (records_table, metrics) = self.batch_inference.run_initial_predictions(self.sample_split)

        # Log the table of sample predictions to W&B
        self._wandb.log({"sample_predictions": records_table})

        # Log the calculated metrics to W&B
        self._wandb.log(metrics)
        print("LLMSampleCB initialized")

    def on_evaluate(self, args: Any, state: Any, control: Any, **kwargs: dict[str, Any]) -> None:
        """Log the sample predictions and metrics to WANDB on eval callback."""
        super().on_evaluate(args, state, control, **kwargs)

        # Generate the table of sample predictions
        _, (records_table, metrics) = self.batch_inference.infer(self.sample_split)

        # Log the table of sample predictions to W&B
        self._wandb.log({"sample_predictions": records_table})

        # Log the calculated metrics to W&B
        self._wandb.log(metrics)
