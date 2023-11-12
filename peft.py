import yaml
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, AutoTokenizer, TrainingArguments
from datasets import load_dataset
from peft import LoraConfig
from trl import SFTTrainer

CONSTANT_MOUNT_PATH="/mnt/k8s.yaml"

if __name__ == "__main__":

    # Note that default configurations are intended for falcoln 7B model
    k8s_yaml_path = "./k8s.yaml"
    if os.path.isfile(CONSTANT_MOUNT_PATH):
        k8s_yaml_path = CONSTANT_MOUNT_PATH
    
    with open(k8s_yaml_path, 'r') as f:
        path = yaml.safe_load(f)

    # Assumes dataset in csv files
    dataset = load_dataset("csv", data_files=os.path.join(path["dataset_mount_path"], path["dataset_file"]), split="train")
    dataset = dataset.shuffle()

    #os.environ["WANDB_DISABLED"] = "true"


    if path["bnb_config_path"] and os.path.isfile(os.path.join(path["bnb_config_path"], path["config_suffix"])):
        bnb_config_args = yaml.safe_load(os.path.join(path["bnb_config_path"], path["config_suffix"]))
        bnb_config = BitsAndBytesConfig(**bnb_config_args)
    else:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )

    # Assumes model is a causal language model
    model = AutoModelForCausalLM.from_pretrained(path["model_mount_path"], quantization_config=bnb_config)
    model.config.use_cache = False
    tokenizer = AutoTokenizer.from_pretrained(path["model_mount_path"])

    #May need to have some custom padding logic here
    tokenizer.pad_token = tokenizer.eos_token

    if path["peft_config_path"] and os.path.isfile(os.path.join(path["peft_config_path"], path["config_suffix"])):
        peft_config_args = yaml.safe_load(os.path.join(path["peft_config_path"], path["config_suffix"]))
        peft_config = LoraConfig(**peft_config_args)
    else:
        peft_config = LoraConfig(
            lora_alpha=16,
            lora_dropout=0.1,
            r=64,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=[
                "query_key_value",
                "dense",
                "dense_h_to_4h",
                "dense_4h_to_h",
            ]
        )
    
    checkpoint_path = path["checkpoint_path"]

    if path["training_config_path"] and os.path.isfile(os.path.join(path["training_config_path"], path["config_suffix"])):
        training_config_args = yaml.safe_load(os.path.join(path["training_config_path"], path["config_suffix"]))
        training_config = TrainingArguments(output_dir = checkpoint_path, **training_config_args)
    else:
        training_config = TrainingArguments(
            output_dir=checkpoint_path,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,
            optim="paged_adamw_32bit",
            save_steps=10,
            logging_steps=10,
            learning_rate=2e-4,
            fp16=True,
            max_grad_norm=0.3,
            max_steps=500,
            warmup_ratio=0.03,
            group_by_length=True,
            lr_scheduler_type="constant",
            gradient_checkpointing=True,
        )


    # TODO : need to parametrize this
    max_seq_length = 512

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        dataset_text_field=path["dataset_training_column"],
        max_seq_length=max_seq_length,
        tokenizer=tokenizer,
        args=training_config
    )

    ## TODO : Remove, falcoln 7B specific training code
    for name, module in trainer.model.named_modules():
        if "norm" in name:
            module = module.to(torch.float32)

    trainer.train()






