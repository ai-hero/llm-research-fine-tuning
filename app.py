import os
from time import sleep
import yaml

from train import load_train_config, do_train, fetch_dataset, load_model

STATIC_CONFIG_PATH="./config.yaml"

def bootstrap_config():
    with open(file=STATIC_CONFIG_PATH, mode='r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    for key, _ in config.items():
        if key in os.environ:
            config[key] = os.environ[key]

    return config

def stall(bootstrap_config):
   sleep(3600)

def do_run(bootstrap_config):
    train_run_config = load_train_config(bootstrap_config)
    dataset, col_name = fetch_dataset(bootstrap_config)
    model, tokenizer = load_model(bootstrap_config, train_run_config)
    do_train(dataset=dataset, train_column_name=col_name, model=model, tokenizer=tokenizer, train_run_config=train_run_config)


def map_env(bootstrap_config):
    if bootstrap_config["env"] == "test":
        return stall
    else:
        return do_run

if __name__ == " __main__":
    config = bootstrap_config()
    executor_func = map_env(config)
    executor_func(config)


