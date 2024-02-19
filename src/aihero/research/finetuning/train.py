"""Launch the training job inside a container."""
import os
from typing import Any, Optional, Tuple

import torch
from datasets import Dataset, DatasetDict
from huggingface_hub import login
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from trl import SFTTrainer
from wandb import finish

from aihero.research.config.schema import TrainingJob
from aihero.research.finetuning.callback import LLMSampleCB
from aihero.research.finetuning.utils import DatasetMover, dataset_generator, peft_module_casting_to_bf16

CHECKPOINT_DIR = "/mnt/checkpoint"
DATASET_DIR = "/mnt/dataset"

if os.environ.get("HF_TOKEN", None):
    login(token=os.environ["HF_TOKEN"])


class TrainingJobRunner:
    """Class to run a training job."""

    def __init__(self, training_job: TrainingJob, distributed_config: Optional[dict[str, Any]] = None):
        """Initialize the training job runner."""
        self.training_job = training_job
        if distributed_config:
            self.distributed_config = distributed_config
            print("Training LOCAL RANK: {} ...".format(os.getenv("LOCAL_RANK", "Unknown")))
            print("Training RANK: {} ...".format(os.getenv("RANK", "Unknown")))
            print("Training LOCAL WORLD SIZE: {} ...".format(os.getenv("LOCAL_WORLD_SIZE", "Unknown")))

        print("Loading model")
        self.model, self.tokenizer = self.load_model()
        print("Loading dataset")
        self.dataset_dict = self.fetch_dataset()

    def load_model(self) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        """Load the model from HuggingFace Hub or S3."""
        use_4bit = self.training_job.quantized or False
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

        if self.training_job.base.type == "huggingface":
            device_map = {"": 0}

            if use_4bit:
                # Load base model
                model = AutoModelForCausalLM.from_pretrained(
                    self.training_job.base.name,
                    quantization_config=bnb_config,
                    device_map=device_map,
                    trust_remote_code=True,
                )
                model.config.use_cache = False
                model.config.pretraining_tp = 1
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    self.training_job.base.name,
                    torch_dtype=torch.bfloat16,
                    use_cache=False,
                    trust_remote_code=True,
                    device_map=device_map,
                )
            tokenizer = AutoTokenizer.from_pretrained(
                self.training_job.base.name,
                trust_remote_code=True,
                add_eos_token=False,
                add_bos_token=False,
            )
            # May need to have some custom padding logic here
            special_tokens = {"pad_token": "[PAD]"}
            tokenizer.add_special_tokens(special_tokens)
            if self.training_job.tokenizer and self.training_job.tokenizer.additional_tokens:
                tokenizer.add_tokens(self.training_job.tokenizer.additional_tokens)
            tokenizer.padding_side = "right"
            model.config.pad_token_id = tokenizer.pad_token_id
            model.resize_token_embeddings(len(tokenizer))

        elif self.training_job.base.type == "s3":
            # TODO : Add s3 support
            raise NotImplementedError("S3 support not implemented yet")
        else:
            raise ValueError(f"Unknown base_model_type: {self.training_job.base.type}")
        return model, tokenizer

    def fetch_dataset(self) -> DatasetDict:
        """Fetch the dataset from HuggingFace Hub or S3."""
        splits = {}
        bos_token = self.tokenizer.bos_token
        eos_token = self.tokenizer.eos_token
        if self.training_job.dataset.type == "huggingface":
            splits["train"] = Dataset.from_generator(
                dataset_generator,
                gen_kwargs={
                    "dataset": self.training_job.dataset.name,
                    "split": "train",
                    "task": self.training_job.task,
                    "bos_token": bos_token,
                    "eos_token": eos_token,
                },
            )
            try:
                splits["val"] = Dataset.from_generator(
                    dataset_generator,
                    gen_kwargs={
                        "dataset": self.training_job.dataset.name,
                        "split": "val",
                        "task": self.training_job.task,
                        "bos_token": bos_token,
                        "eos_token": eos_token,
                    },
                )
            except:  # pylint: disable=bare-except  # noqa: E722
                print("Unable to create val dataset")
            try:
                splits["test"] = Dataset.from_generator(
                    dataset_generator,
                    gen_kwargs={
                        "dataset": self.training_job.dataset.name,
                        "split": "test",
                        "task": self.training_job.task,
                        "bos_token": bos_token,
                        "eos_token": eos_token,
                    },
                )
            except:  # pylint: disable=bare-except  # noqa: E722
                print("Unable to create test dataset")
        elif self.training_job.dataset.type == "s3":
            os.makedirs(DATASET_DIR)
            dataset_mover = DatasetMover()
            # If the dataset is s3, download it to the local directory
            # The path would look like bucket_name/path/to/dataset_name.tar.gz
            # local_name would then be = path/to/dataset_name.tar.gz
            local_name = self.training_job.dataset.name[self.training_job.dataset.name.find("/") + 1 :]
            dataset_mover.download(
                bucket_name=self.training_job.dataset.name.split("/")[0],
                object_name=f"{local_name}.tar.gz",
                output_folder_path=DATASET_DIR,
            )
            print(os.listdir(DATASET_DIR))
            print(os.listdir(f"{DATASET_DIR}/{local_name}"))
            splits["train"] = Dataset.from_generator(
                dataset_generator,
                gen_kwargs={
                    "dataset": f"{DATASET_DIR}/{local_name}",
                    "split": "train",
                    "from_disk": True,
                    "task": self.training_job.task,
                    "bos_token": bos_token,
                    "eos_token": eos_token,
                },
            )
            try:
                splits["val"] = Dataset.from_generator(
                    dataset_generator,
                    gen_kwargs={
                        "dataset": f"{DATASET_DIR}/{local_name}",
                        "split": "val",
                        "from_disk": True,
                        "task": self.training_job.task,
                        "bos_token": bos_token,
                        "eos_token": eos_token,
                    },
                )
            except:  # pylint: disable=bare-except  # noqa: E722
                print("Unable to create val dataset")
            try:
                splits["test"] = Dataset.from_generator(
                    dataset_generator,
                    gen_kwargs={
                        "dataset": f"{DATASET_DIR}/{local_name}",
                        "split": "test",
                        "from_disk": True,
                        "task": self.training_job.task,
                        "bos_token": bos_token,
                        "eos_token": eos_token,
                    },
                )
            except:  # pylint: disable=bare-except  # noqa: E722
                print("Unable to create test dataset")
        elif self.training_job.dataset.type == "local":
            print("Loading dataset locally: ", os.listdir(self.training_job.dataset.path))
            splits["train"] = Dataset.from_generator(
                dataset_generator,
                gen_kwargs={
                    "dataset": self.training_job.dataset.path,
                    "split": "train",
                    "from_disk": True,
                    "task": self.training_job.task,
                    "bos_token": bos_token,
                    "eos_token": eos_token,
                },
            )
            try:
                splits["val"] = Dataset.from_generator(
                    dataset_generator,
                    gen_kwargs={
                        "dataset": self.training_job.dataset.path,
                        "split": "val",
                        "from_disk": True,
                        "task": self.training_job.task,
                        "bos_token": bos_token,
                        "eos_token": eos_token,
                    },
                )
            except:  # pylint: disable=bare-except  # noqa: E722
                print("Unable to create val dataset")
            try:
                splits["test"] = Dataset.from_generator(
                    dataset_generator,
                    gen_kwargs={
                        "dataset": self.training_job.dataset.path,
                        "split": "test",
                        "from_disk": True,
                        "task": self.training_job.task,
                        "bos_token": bos_token,
                        "eos_token": eos_token,
                    },
                )
            except:  # pylint: disable=bare-except  # noqa: E722
                print("Unable to create test dataset")
        else:
            raise ValueError(f"Unknown dataset_type: {self.training_job.dataset.type}")

        return DatasetDict(splits)

    def freeze(self) -> None:
        """Freeze the model layers for SFT without PEFT."""
        if self.training_job.freeze:
            n_freeze = self.training_job.freeze.n_freeze or 24

            if n_freeze > 0:
                module_name: str = "layers"

                def _find_mod(model: AutoModelForCausalLM, module_name: str) -> Any:
                    for name, mod in model.named_modules():
                        if name.endswith(module_name):
                            return mod

                # freeze layers (disable gradients)
                for param in self.model.parameters():
                    param.requires_grad = False

                # never freeze the head
                for param in self.model.lm_head.parameters():
                    param.requires_grad = True

                layers = _find_mod(self.model, module_name)
                for param in layers[n_freeze:].parameters():
                    param.requires_grad = True

            # Freeze embeddings for small memory decrease
            if self.training_job.freeze.freeze_embed:
                embed_tokens = _find_mod(self.model, "embed_tokens")
                embed_tokens.weight.requires_grad_(False)

    def save_model(self) -> None:
        """Save the model to a local directory."""
        """Upload the model to HuggingFace Hub or S3."""
        if os.getenv("RANK", "0") != "0":
            return
        if not self.training_job.output:
            return
        print("Saving model and tokenizer")
        local_name = self.training_job.output.name.split("/")[-1]
        self.model.save_pretrained(local_name)
        self.tokenizer.save_pretrained(local_name)
        print("Model saved at: ", os.listdir(local_name))
        if self.training_job.output.type == "huggingface":
            print("Saving model and tokenizer to hf")
            self.model.push_to_hub(local_name)
            self.tokenizer.push_to_hub(local_name)
        elif self.training_job.output.type == "s3":
            # TODO : Add s3 support
            raise NotImplementedError("S3 support not implemented yet")

    def train(self) -> None:
        """Start training the model as defined by the config."""
        # Assumes model is a causal language model
        self.model.config.use_cache = False
        train_split = self.dataset_dict["train"]
        val_split = self.dataset_dict.get("val", None)
        test_split = self.dataset_dict.get("test", None)

        # SFT training config
        training_arguments_dict = self.training_job.sft.model_dump()
        training_arguments_dict["save_total_limit"] = 2
        training_arguments_dict["save_strategy"] = "steps"
        training_arguments_dict["load_best_model_at_end"] = True
        training_arguments_dict["output_dir"] = CHECKPOINT_DIR
        training_arguments = TrainingArguments(**training_arguments_dict)
        # PEFT training config
        if self.training_job.peft:
            lora_config = LoraConfig(**self.training_job.peft.model_dump())
            model = get_peft_model(self.model, lora_config)
            if self.training_job.peft.bf16:
                peft_module_casting_to_bf16(model, self.training_job.peft.model_dump())
            model.print_trainable_parameters()
            self.training_job.sft.peft_config = lora_config
            self.training_job.sft.n_freeze = "all"
        elif self.training_job.freeze:
            self.freeze()
            peft_config = None

        trainer = SFTTrainer(
            tokenizer=self.tokenizer,
            model=self.model,
            train_dataset=train_split,
            eval_dataset=val_split,
            peft_config=peft_config,
            dataset_text_field="text",
            max_seq_length=self.training_job.trainer.max_seq_length,
            packing=self.training_job.trainer.packing,  # Should you combine multiple examples into one sequence?
            args=training_arguments,
        )

        task = self.training_job.dataset.task
        if self.training_job.eval and self.training_job.eval.tests:
            run_tests_str = self.training_job.eval.tests
        else:
            run_tests_str = ""
        if self.training_job.eval and self.training_job.eval.metrics:
            run_metrics_str = self.training_job.eval.metrics
        else:
            run_metrics_str = ""
        if test_split and test_split.num_rows > 0 and task == "completion":
            # we instantiate the W&B callback with the trainer object and the dataset we want to sample from
            wandb_callback = LLMSampleCB(
                trainer,
                task,
                test_split,
                num_samples=100,
                max_new_tokens=self.training_job.trainer.max_seq_length,
                run_tests_str=run_tests_str,
                run_metrics_str=run_metrics_str,
            )
            wandb_callback.initialize()
            trainer.add_callback(wandb_callback)

        # distributed training config
        if trainer.is_fsdp_enabled:
            trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")

        print("Starting training")
        trainer.train()

        # if test_split and test_split.num_rows > 0:
        #     trainer.predict(test_split)

    def run(self) -> None:
        """Execute the main training loop."""
        print("Starting training")
        self.train()
        print("Save and Uploading model..")
        self.save_model()
        finish()
