"""Module to run batch inference jobs."""
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Tuple

import torch
from datasets import Dataset, DatasetDict, DatasetInfo
from huggingface_hub import login
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, GenerationConfig
from wandb import Table, finish

from aihero.research.config.schema import BatchInferenceJob
from aihero.research.finetuning.utils import DatasetMover, dataset_generator

CHECKPOINT_DIR = "/mnt/checkpoint"
DATASET_DIR = "/mnt/dataset"
MAX_NEW_TOKENS = 512

if os.environ.get("HF_TOKEN", None):
    login(token=os.environ["HF_TOKEN"])


class BatchInferenceJobRunner:
    """Class to run a batch inferenc job."""

    def __init__(self, batch_inference_job: BatchInferenceJob):
        """Initialize the training job runner."""
        self.batch_inference_job = batch_inference_job
        print("Loading model")
        self.model, self.tokenizer = self.load_model()
        print("Loading dataset")
        self.dataset_dict = self.fetch_dataset()

        # Prep for eval
        if self.batch_inference_job.eval:
            run_tests_str = self.batch_inference_job.tests or ""
            run_metrics_str = self.batch_inference_job.metrics or ""
            size = self.batch_inference_job.size or 100
            randomize = self.batch_inference_job.randomize or False
        else:
            run_tests_str = self.batch_inference_job.tests or ""
            run_metrics_str = self.batch_inference_job.metrics or ""
            size = 100
            randomize = False

        self.batch_inference_split = self.dataset_dict["batch_inference"]
        if size:
            if randomize:
                self.batch_inference_split = self.batch_inference_split.shuffle()
            self.batch_inference_split = self.batch_inference_split.select(range(size))

        self.batch_inference_with_eval = BatchInferenceWithEval(
            model=self.model,
            tokenizer=self.tokenizer,
            task=self.batch_inference_job.task,
            run_tests_str=run_tests_str,
            run_metrics_str=run_metrics_str,
            max_new_tokens=self.batch_inference_job.generator.max_seq_length or MAX_NEW_TOKENS,
        )

    def load_model(self) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        """Load the model from HuggingFace Hub or S3."""
        use_4bit = self.batch_inference_job.quantized or False
        if use_4bit:
            # Compute dtype for 4-bit base models
            bnb_4bit_compute_dtype = "float16"
            # Quantization type (fp4 or nf4)
            bnb_4bit_quant_type = "nf4"
            # Activate nested quantization for 4-bit base models (double quantization)
            use_nested_quant = False

            # Load tokenizer and model with QLoRA configuration
            compute_dtype = getattr(torch, bnb_4bit_compute_dtype)

            bnb_config = BitsAndBytesConfig(
                load_in_4bit=use_4bit,
                bnb_4bit_quant_type=bnb_4bit_quant_type,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=use_nested_quant,
            )

            # Check GPU compatibility with bfloat16
            if compute_dtype == torch.float16 and use_4bit:
                major, _ = torch.cuda.get_device_capability()
                if major >= 8:
                    print("=" * 80)
                    print("Your GPU supports bfloat16: accelerate training with bf16=True")
                    print("=" * 80)

        if self.batch_inference_job.model.type == "huggingface":
            device_map = {"": 0}

            if use_4bit:
                # Load base model
                model = AutoModelForCausalLM.from_pretrained(
                    self.batch_inference_job.model.name,
                    quantization_config=bnb_config,
                    device_map=device_map,
                    trust_remote_code=True,
                )
                model.config.use_cache = False
                model.config.pretraining_tp = 1
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    self.batch_inference_job.model.name,
                    torch_dtype=torch.bfloat16,
                    use_cache=False,
                    trust_remote_code=True,
                    device_map=device_map,
                )
            tokenizer = AutoTokenizer.from_pretrained(
                self.batch_inference_job.model.name,
                trust_remote_code=True,
                add_eos_token=False,
                add_bos_token=False,
            )
            # May need to have some custom padding logic here
            special_tokens = {"pad_token": "[PAD]"}
            tokenizer.add_special_tokens(special_tokens)
            if self.batch_inference_job.tokenizer and self.batch_inference_job.tokenizer.additional_tokens:
                tokenizer.add_tokens(self.batch_inference_job.tokenizer.additional_tokens)
            tokenizer.padding_side = "right"
            model.config.pad_token_id = tokenizer.pad_token_id
            model.resize_token_embeddings(len(tokenizer))
        elif self.batch_inference_job.model.type == "s3":
            # TODO : Add s3 support
            raise NotImplementedError("S3 support not implemented yet")
        else:
            raise ValueError(f"Unknown base_model_type: {self.batch_inference_job.model.type}")
        return model, tokenizer

    def fetch_dataset(self) -> DatasetDict:
        """Fetch the dataset from HuggingFace Hub or S3."""
        splits = {}
        bos_token = self.tokenizer.bos_token
        eos_token = self.tokenizer.eos_token
        if self.batch_inference_job.dataset.type == "huggingface":
            splits["batch_inference"] = Dataset.from_generator(
                dataset_generator,
                gen_kwargs={
                    "dataset": self.batch_inference_job.dataset.name,
                    "split": "batch_inference",
                    "task": self.batch_inference_job.task,
                    "bos_token": bos_token,
                    "eos_token": eos_token,
                },
            )
        elif self.batch_inference_job.dataset.type == "s3":
            os.makedirs(DATASET_DIR)
            dataset_mover = DatasetMover()
            # If the dataset is s3, download it to the local directory
            # The path would look like bucket_name/path/to/dataset_name.tar.gz
            # local_name would then be = path/to/dataset_name.tar.gz
            local_name = self.batch_inference_job.dataset.name[self.batch_inference_job.dataset.name.find("/") + 1 :]
            dataset_mover.download(
                bucket_name=self.batch_inference_job.dataset.name.split("/")[0],
                object_name=f"{local_name}.tar.gz",
                output_folder_path=DATASET_DIR,
            )
            print(os.listdir(DATASET_DIR))
            print(os.listdir(f"{DATASET_DIR}/{local_name}"))
            splits["batch_inference"] = Dataset.from_generator(
                dataset_generator,
                gen_kwargs={
                    "dataset": f"{DATASET_DIR}/{local_name}",
                    "split": "batch_inference",
                    "from_disk": True,
                    "task": self.batch_inference_job.task,
                    "bos_token": bos_token,
                    "eos_token": eos_token,
                },
            )
        else:
            raise ValueError(f"Unknown dataset_type: {self.batch_inference_job.dataset.type}")

        return DatasetDict(splits)

    def infer_on_dataset(self) -> None:
        """Generate batch predictions."""
        # Batch inference config
        predicted_rows, (_, _) = self.batch_inference_with_eval.infer(self.batch_inference_split)

        with TemporaryDirectory() as temp_dir:
            temp_dir_path = Path(temp_dir)
            # If you're creating a new dataset from scratch:
            dataset_dict = DatasetDict(
                {"predictions": Dataset.from_list(predicted_rows)}  # Assign the new dataset as the train split
            )

            output_short_name = self.batch_inference_job.dataset.name.split("/")[-1] + "-output"
            print(f"Converting {output_short_name} to dataset dict")

            dataset_info = DatasetInfo(
                description=f"Contains output for {output_short_name} from batch inference",
                version="1.0.0",
            )
            for split, dataset in dataset_dict.items():
                dataset.dataset_info = dataset_info
            dataset_path = (temp_dir_path / output_short_name).as_posix()
            dataset_dict.save_to_disk(dataset_path)

            # Compress the folder
            print(f"Compressing the folder {dataset_path}")
            folder_to_compress = dataset_path
            output_tar_file = f"{output_short_name}.tar.gz"
            bucket_name = "fine-tuning-research"
            print(f"Uploading {output_tar_file} to {bucket_name}")
            dataset_mover = DatasetMover()
            dataset_mover.upload(folder_to_compress, output_tar_file, bucket_name)

    def run(self) -> None:
        """Execute the main inference loop."""
        print("Starting batch inference")
        self.infer_on_dataset()
        print("Save and Uploading model..")
        finish()


class BatchInferenceWithEval:
    """Batch inference class for generating predictions and running custom tests and metrics."""

    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        task: str,
        run_tests_str: str = "",
        run_metrics_str: str = "",
        max_new_tokens: int = MAX_NEW_TOKENS,
        batch_size: int = 8,
    ):
        """Initialize the batch inference class."""
        self.gen_config = GenerationConfig.from_pretrained(model.name_or_path, max_new_tokens=max_new_tokens)
        self.model = model
        self.tokenizer = tokenizer
        self.task = task
        self.run_tests_str = run_tests_str
        if run_tests_str and os.environ.get("ALLOW_CUSTOM_TESTS", "false").lower() == "true":
            exec(self.run_tests_str, globals())
        else:
            self.run_tests_str = ""
        self.run_metrics_str = run_metrics_str
        if run_metrics_str and os.environ.get("ALLOW_CUSTOM_METRICS", "false").lower() == "true":
            exec(self.run_metrics_str, globals())
        else:
            self.run_metrics_str = ""
        self.initial_predictions: list[str] = []

    def generate(self, prompt: str) -> Any:
        """Generate a completion from a prompt."""
        tokenized_prompt = self.tokenizer(prompt, return_tensors="pt", padding=True)["input_ids"].cuda()
        with torch.inference_mode():
            output = self.model.generate(inputs=tokenized_prompt, generation_config=self.gen_config)
        return self.tokenizer.decode(output[0][len(tokenized_prompt[0]) :], skip_special_tokens=True)

    def run_initial_predictions(self, rows: Dataset) -> Tuple[list[dict[str, Any]], Tuple[Table, dict[str, Any]]]:
        """Generate initial predictions for the sample split."""
        # Test the provided code if present:
        print("Testing custom code, on ground truth if provided")
        test_rows = []
        for example in tqdm(rows, leave=False):
            prompt = example["prompt"]
            actual = example["completion"]
            test_rows.append({"prompt": prompt, "actual": actual, "predicted": actual, "initial": actual})
        self.execute_custom_code(test_rows)

        print("Generating initial predictions for sample split")
        predicted_rows = []
        for example in tqdm(rows, leave=False):
            if self.task == "text":
                prompt = example["text"]
            else:
                prompt = example["prompt"]
            if not prompt.startswith(self.tokenizer.bos_token):
                prompt = f"{self.tokenizer.bos_token}{prompt}"
            predicted = self.generate(prompt=prompt)
            self.initial_predictions.append(predicted)
            actual = example["completion"]
            predicted_rows.append({"prompt": prompt, "actual": actual, "predicted": predicted, "initial": predicted})
        return predicted_rows, self.execute_custom_code(predicted_rows)

    def infer(self, rows: Dataset) -> Tuple[list[dict[str, Any]], Tuple[Table, dict[str, Any]]]:
        """Generate batch predictions."""
        print("Generating predictions for sample split")
        predicted_rows = []
        for i, example in tqdm(enumerate(rows), leave=False):
            if self.task == "text":
                prompt = example["text"]
            else:
                prompt = example["prompt"]
            actual = example["completion"]
            if not prompt.startswith(self.tokenizer.bos_token):
                prompt = f"{self.tokenizer.bos_token}{prompt}"
            predicted = self.generate(prompt=prompt)
            row_obj = {"prompt": prompt, "actual": actual, "predicted": predicted}
            if self.initial_predictions:
                row_obj["initial"] = self.initial_predictions[i]
            predicted_rows.append(row_obj)
        return predicted_rows, self.execute_custom_code(predicted_rows)

    def execute_custom_code(self, rows: list[dict[str, Any]]) -> Tuple[Table, dict[str, Any]]:
        """Execute custom code for tests and metrics."""
        records_table = Table(columns=["prompt", "predicted", "actual", "initial", "test_result", "errors"])

        # Assuming run_tests_str and run_metrics_str contain your testing and metrics code respectively

        print("Updating records_table with predictions, test results, and errors")
        if self.run_tests_str and os.environ.get("ALLOW_CUSTOM_TESTS", "false").lower() == "true":
            # Execute dynamic code for tests
            print("Running custom tests")
            tests, errors = run_tests([row["prompt"] for row in rows], [row["predicted"] for row in rows])  # type: ignore  # noqa: F821
        else:
            print("Skipping custom tests")
            tests, errors = [False] * len(rows), [""] * len(rows)

        if self.run_metrics_str and os.environ.get("ALLOW_CUSTOM_METRICS", "false").lower() == "true":
            # Execute dynamic code for metrics
            print("Running custom metrics")
            pts = [row["prompt"] for row in rows]
            acts = [row["actual"] for row in rows]
            prds = [row["predicted"] for row in rows]
            metrics = run_metrics(pts, acts, prds)  # type: ignore  # noqa: F821
        else:
            print("Skipping custom metrics")
            metrics = {}

        print("Building table")
        index = 0
        passed = 0
        for row in tqdm(rows, leave=False):
            test_result = "PASS" if tests[index] else "FAIL"
            passed += 1 if test_result == "PASS" else 0
            error_message = errors[index] if index < len(errors) else ""
            records_table.add_data(
                row["prompt"],
                row["predicted"],
                row["actual"],
                row.get("initial", "N/A"),
                test_result,
                error_message,
            )
            index += 1

        metrics["passed"] = passed * 100 / len(rows)
        print("Metrics:", metrics)

        return records_table, metrics
