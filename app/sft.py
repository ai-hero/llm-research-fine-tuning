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
from transformers.integrations import WandbCallback
from trl import SFTTrainer
from utils import DatasetMover
from wandb import Table

DEFAULT_STATIC_CONFIG_PATH = "./config.yaml"
MOUNTED_CONFIG_PATH = "/mnt/config/training/config.yaml"
CHECKPOINT_DIR = "/mnt/checkpoint"
DATASET_DIR = "/mnt/dataset"


def training_generator(dataset, split="train", from_disk=False, format="text"):
    if from_disk:
        ds = load_from_disk(dataset)
        ds = ds[split]
    else:
        ds = load_dataset(dataset, streaming=True, split=split)
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
        return Dataset.from_generator(
            training_generator,
            gen_kwargs={
                "dataset": config["dataset"]["name"],
                "split": "train",
                "format": config["dataset"].get("format", "text"),
            },
        ), Dataset.from_generator(
            training_generator,
            gen_kwargs={
                "dataset": config["dataset"]["name"],
                "split": "val",
                "format": config["dataset"].get("format", "text"),
            },
        )
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
        return Dataset.from_generator(
            training_generator,
            gen_kwargs={
                "dataset": f"{DATASET_DIR}/{local_name}",
                "split": "train",
                "from_disk": True,
                "format": config["dataset"].get("format", "text"),
            },
        ), Dataset.from_generator(
            training_generator,
            gen_kwargs={
                "dataset": f"{DATASET_DIR}/{local_name}",
                "split": "val",
                "from_disk": True,
                "format": config["dataset"].get("format", "text"),
            },
        )
    else:
        raise ValueError(f"Unknown dataset_type: {config['dataset']['type']}")


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


def train(train_dataset, val_dataset, model, tokenizer, config):
    # Note that default configurations are intended for falcoln 7B model
    train_dataset = train_dataset.shuffle()

    # Assumes model is a causal language model
    model.config.use_cache = False

    # May need to have some custom padding logic here
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

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
        freeze(model, sft_config.n_freeze, sft_config.freeze_embed)

    trainer = SFTTrainer(
        tokenizer=tokenizer,
        model=model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        peft_config=peft_config,
        dataset_text_field="text",
        max_seq_length=config["training"]["trainer"]["max_seq_length"],
        packing=config["training"]["trainer"]["packing"],
        args=sft_config,
    )

    trainer.train()


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
    train_dataset, val_dataset = fetch_dataset(config)
    print("Loading model")
    model, tokenizer = load_model(config)
    print("Starting training")
    train(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        model=model,
        tokenizer=tokenizer,
        config=config,
    )
    print("Uploading model..")
    upload_model(config)


if __name__ == "__main__":
    Fire(main)
