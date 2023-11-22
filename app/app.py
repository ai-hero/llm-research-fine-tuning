import os
import yaml
from train import load_train_config, do_train, fetch_dataset, load_model, upload_model

STATIC_CONFIG_PATH="./config.yaml"

def bootstrap_config():
    with open(file=STATIC_CONFIG_PATH, mode='r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    for key, _ in config.items():
        if key in os.environ:
            config[key] = os.environ[key]

    return config

def do_run(bootstrap_config):
    print("Loading train config")
    train_run_config = load_train_config(bootstrap_config)
    print("Loading dataset")
    dataset, col_name = fetch_dataset(bootstrap_config)
    print("Loading model")
    model, tokenizer = load_model(bootstrap_config, train_run_config)
    print("Starting training")
    do_train(dataset=dataset, train_column_name=col_name, model=model, tokenizer=tokenizer, train_run_config=train_run_config)
    upload_model(bootstrap_config)


if __name__ == "__main__":
    print("Loading config")
    config = bootstrap_config()
    print("Starting run")
    do_run(config)


