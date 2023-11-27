import os
import yaml
from train import load_train_config, do_train, fetch_dataset, load_model, upload_model
from fire import Fire

STATIC_CONFIG_PATH = "./config.yaml"


def main(
    base_model_type: str = "hf",
    base_model_name: str = "",
    dataset_type: str = "hf",
    dataset_name: str = "",
    output_model_type: str = "hf",
    output_model_name: str = "",
):
    print("Loading config")
    with open(file=STATIC_CONFIG_PATH, mode="r", encoding="utf-8") as f:
        bootstrap_config = yaml.safe_load(f)

    # Base Model
    bootstrap_config["base_model_type"] = (
        base_model_type if base_model_type else bootstrap_config["base_model_type"]
    )
    bootstrap_config["base_model_name"] = (
        base_model_name if base_model_name else bootstrap_config["base_model_name"]
    )

    # Dataset
    bootstrap_config["dataset_type"] = (
        dataset_type if dataset_type else bootstrap_config["dataset_type"]
    )
    bootstrap_config["dataset_name"] = (
        dataset_name if dataset_name else bootstrap_config["dataset_name"]
    )

    # Output Model
    bootstrap_config["output_model_type"] = (
        output_model_type
        if output_model_type
        else bootstrap_config["output_model_type"]
    )
    bootstrap_config["output_model_name"] = (
        output_model_name
        if output_model_name
        else bootstrap_config["output_model_name"]
    )

    print("Loading train config")
    train_run_config = load_train_config(bootstrap_config)
    print("Loading dataset")
    dataset = fetch_dataset(bootstrap_config)
    print("Loading model")
    model, tokenizer = load_model(bootstrap_config, train_run_config)
    print("Starting training")
    do_train(
        dataset=dataset,
        train_column_name="text",
        model=model,
        tokenizer=tokenizer,
        train_run_config=train_run_config,
    )
    upload_model(bootstrap_config)


if __name__ == "__main__":
    Fire(main)
