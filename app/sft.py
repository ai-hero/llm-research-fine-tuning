import os

import torch
import tqdm
import yaml
from datasets import Dataset, load_dataset, load_from_disk
from fire import Fire
from huggingface_hub import HfApi, login
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    GenerationConfig,
    TrainingArguments,
)
from trl import SFTTrainer
from utils import DatasetMover
from transformers.integrations import WandbCallback
from wandb import Table

DEFAULT_STATIC_CONFIG_PATH = "./default_config.yaml"
MOUNTED_CONFIG_PATH = "/mnt/config/training/config.yaml"
CHECKPOINT_DIR = "/mnt/checkpoint"
DATASET_DIR = "/mnt/dataset"


def training_generator(dataset, split="train", from_disk=False, format="text"):
    if from_disk:
        ds = load_from_disk(dataset)
        ds = ds[split]
    else:
        ds = load_dataset(dataset, streaming=True, split=split)
    print(f"{ds.num_rows} rows in {split} split")
    for row in iter(ds):
        if format == "text":
            text = row["text"]
        elif format == "completion":
            text = row["prompt"] + "\n" + row["completion"]
        yield {"text": text}


def fetch_dataset(config):
    if config["dataset"]["type"] == "hf":
        if os.environ.get("HF_TOKEN", None):
            login(token=os.environ["HF_TOKEN"])
        train_dataset = Dataset.from_generator(
            training_generator,
            gen_kwargs={
                "dataset": config["dataset"]["name"],
                "split": "train",
                "format": config["dataset"].get("format", "text"),
            },
        )
        try:
            val_dataset = Dataset.from_generator(
                training_generator,
                gen_kwargs={
                    "dataset": config["dataset"]["name"],
                    "split": "val",
                    "format": config["dataset"].get("format", "text"),
                },
            )
        except:
            print("Unable to create val dataset")
            val_dataset = None
        try:
            test_dataset = Dataset.from_generator(
                training_generator,
                gen_kwargs={
                    "dataset": config["dataset"]["name"],
                    "split": "test",
                    "format": config["dataset"].get("format", "text"),
                },
            )
        except:
            print("Unable to create test dataset")
            test_dataset = None
    elif config["dataset"]["type"] == "s3":
        os.makedirs(DATASET_DIR)
        dataset_mover = DatasetMover()
        local_name = config["dataset"]["name"][
            config["dataset"]["name"].find("/") + 1 :
        ]
        dataset_mover.download(
            bucket_name=config["dataset"]["name"].split("/")[0],
            object_name=f"{local_name}.tar.gz",
            output_folder_path=DATASET_DIR,
        )
        print(os.listdir(DATASET_DIR))
        print(os.listdir(f"{DATASET_DIR}/{local_name}"))
        train_dataset = Dataset.from_generator(
            training_generator,
            gen_kwargs={
                "dataset": f"{DATASET_DIR}/{local_name}",
                "split": "train",
                "from_disk": True,
                "format": config["dataset"].get("format", "text"),
            },
        )
        try:
            val_dataset = Dataset.from_generator(
                training_generator,
                gen_kwargs={
                    "dataset": f"{DATASET_DIR}/{local_name}",
                    "split": "val",
                    "from_disk": True,
                    "format": config["dataset"].get("format", "text"),
                },
            )
        except:
            print("Unable to create val dataset")
            val_dataset = None
        try:
            test_dataset = Dataset.from_generator(
                training_generator,
                gen_kwargs={
                    "dataset": f"{DATASET_DIR}/{local_name}",
                    "split": "test",
                    "from_disk": True,
                    "format": config["dataset"].get("format", "text"),
                },
            )
        except:
            print("Unable to create test dataset")
            test_dataset = None
    else:
        raise ValueError(f"Unknown dataset_type: {config['dataset']['type']}")
    return train_dataset, val_dataset, test_dataset


def load_model(config):
    if config["model"]["base"]["type"] == "hf":
        if os.environ.get("HF_TOKEN", None):
            login(token=os.environ["HF_TOKEN"])

        model = AutoModelForCausalLM.from_pretrained(
            config["model"]["base"]["name"],
            torch_dtype=torch.bfloat16,
            use_cache=False,
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            config["model"]["base"]["name"], trust_remote_code=True
        )
    elif config["model"]["base"]["type"] == "s3":
        # TODO : Add s3 support
        raise NotImplementedError("S3 support not implemented yet")
    else:
        raise ValueError(f"Unknown base_model_type: {config['model']['base']['type']}")
    return model, tokenizer


def freeze(model, n_freeze, freeze_embed, module_name="layers"):
    if n_freeze > 0:

        def _find_mod(model, module_name):
            for name, mod in model.named_modules():
                if name.endswith(module_name):
                    return mod

        # freeze layers (disable gradients)
        for param in model.parameters():
            param.requires_grad = False

        # never freeze the head
        for param in model.lm_head.parameters():
            param.requires_grad = True

        layers = _find_mod(model, module_name)
        for param in layers[n_freeze:].parameters():
            param.requires_grad = True

    # Freeze embeddings for small memory decrease
    if freeze_embed:
        embed_tokens = _find_mod(model, "embed_tokens")
        embed_tokens.weight.requires_grad_(False)


def train(train_split, val_split, test_split, model, tokenizer, config):
    # Assumes model is a causal language model
    model.config.use_cache = False

    # May need to have some custom padding logic here
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Calculate max steps from num epochs
    if "num_train_epochs" in config["training"]["sft"]:
        num_train_epochs = config["training"]["sft"].pop("num_train_epochs")
        max_steps = (
            num_train_epochs
            * train_split.num_rows
            // config["training"]["sft"]["per_device_train_batch_size"]
        )  # TODO: Add num gpus when we do FSDP
        config["training"]["sft"]["max_steps"] = max_steps
        save_steps = max_steps // 8
        config["training"]["sft"]["save_steps"] = save_steps
        config["training"]["sft"]["save_strategy"] = "steps"

    # SFT training config
    sft_config = TrainingArguments(
        output_dir=CHECKPOINT_DIR, **config["training"]["sft"]
    )
    # PEFT training config
    if "peft" in config["training"]:
        peft_config = LoraConfig(**config["training"]["peft"])
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
        sft_config.peft_config = peft_config
        sft_config.n_freeze = "all"
    else:
        n_freeze = config["training"]["other"].get("n_freeze", 24)
        freeze_embed = config["training"]["other"].get("freeze_embed", True)
        freeze(model, n_freeze, freeze_embed)
        peft_config = None

    trainer = SFTTrainer(
        tokenizer=tokenizer,
        model=model,
        train_dataset=train_split,
        eval_dataset=val_split,
        peft_config=peft_config,
        dataset_text_field="text",
        max_seq_length=config["training"]["trainer"]["max_seq_length"],
        packing=config["training"]["trainer"][
            "packing"
        ],  # Should you combine multiple examples into one sequence?
        args=sft_config,
    )

    trainer.train()

    # if test_split:
    #     trainer.evaluate(test_split)


def upload_model(config):
    if "output" not in config["model"]:
        return
    if config["model"]["output"]["type"] == "hf":
        if os.environ.get("HF_TOKEN", None):
            login(token=os.environ["HF_TOKEN"])
        api = HfApi()
        api.upload_folder(
            folder_path=CHECKPOINT_DIR,
            repo_id=config["model"]["output"]["name"],
            repo_type="model",
        )
    elif config["model"]["output"]["type"] == "s3":
        # TODO : Add s3 support
        raise NotImplementedError("S3 support not implemented yet")


def main():
    if os.path.exists(MOUNTED_CONFIG_PATH):
        config_file = MOUNTED_CONFIG_PATH
        print("Loading mounted config")
    else:
        config_file = DEFAULT_STATIC_CONFIG_PATH
        print("Loading default config")
    with open(file=config_file, mode="r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    print("Loading dataset")
    train_split, val_split, test_split = fetch_dataset(config)
    print("Loading model")
    model, tokenizer = load_model(config)
    print("Starting training")
    train(
        train_split=train_split,
        val_split=val_split,
        test_split=test_split,
        model=model,
        tokenizer=tokenizer,
        config=config,
    )
    print("Uploading model..")
    upload_model(config)


if __name__ == "__main__":
    Fire(main)
