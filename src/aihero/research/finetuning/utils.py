"""Utility functions for the app. e.g. upload and download files from S3."""
import os
import tarfile
from typing import Any, Generator

import torch
from datasets import load_dataset, load_from_disk
from minio import Minio, S3Error
from peft.tuners.lora import LoraLayer
from transformers import AutoModelForCausalLM


class DatasetMover:
    """Utility class for uploading and downloading files from S3."""

    def _compress_folder(self, folder_path: str, output_filename: str) -> None:
        """Compress a folder into a tar.gz file."""
        with tarfile.open(output_filename, "w:gz") as tar:
            tar.add(folder_path, arcname=os.path.basename(folder_path))

    def _upload_to_s3(self, file_name: str, bucket_name: str, object_name: str) -> None:
        """Upload a file to S3."""
        try:
            # Initialize MinIO client
            minio_client = Minio(
                os.environ["S3_ENDPOINT"],
                access_key=os.environ["S3_ACCESS_KEY_ID"],
                secret_key=os.environ["S3_SECRET_ACCESS_KEY"],
                region=os.environ["S3_REGION"],
                secure=os.environ.get("S3_SECURE", "True").lower() == "true",
            )  # Use secure=False if not using https
            minio_client.fput_object(bucket_name, object_name, file_name)
            print(f"'{file_name}' is successfully uploaded as '{object_name}' to bucket '{bucket_name}'.")
        except S3Error as e:
            print("Error occurred: ", e)

    def upload(self, folder_path: str, output_filename: str, bucket_name: str) -> None:
        """Compress the folder and upload it to S3."""
        self._compress_folder(folder_path, output_filename)
        self._upload_to_s3(output_filename, bucket_name, output_filename)

    def _download_from_s3(self, bucket_name: str, object_name: str, file_name: str) -> None:
        """Download a file from S3."""
        try:
            # Initialize MinIO client
            minio_client = Minio(
                os.environ["S3_ENDPOINT"],
                access_key=os.environ["S3_ACCESS_KEY_ID"],
                secret_key=os.environ["S3_SECRET_ACCESS_KEY"],
                region=os.environ["S3_REGION"],
                secure=os.environ.get("S3_SECURE", "True").lower() == "true",
            )
            minio_client.fget_object(bucket_name, object_name, file_name)
            print(f"'{object_name}' from bucket '{bucket_name}' is successfully downloaded as '{file_name}'.")
        except S3Error as e:
            print("Error occurred: ", e)

    def _decompress_folder(self, input_filename: str, output_folder_path: str) -> None:
        """Decompress a tar.gz file into a folder."""
        try:
            with tarfile.open(input_filename, "r:gz") as tar:
                tar.extractall(path=output_folder_path)
            print(f"'{input_filename}' is successfully decompressed to '{output_folder_path}'.")
        except Exception as e:
            print("Error occurred: ", e)

    def download(self, bucket_name: str, object_name: str, output_folder_path: str) -> None:
        """Download a tar.gz file from S3 and decompress it into a folder."""
        temp_filename = "temp.tar.gz"
        self._download_from_s3(bucket_name, object_name, temp_filename)
        self._decompress_folder(temp_filename, output_folder_path)
        os.remove(temp_filename)  # Clean up the temporary compressed file


def dataset_generator(
    dataset: str,
    split: str = "train",
    from_disk: bool = False,
    task: str = "text",
    bos_token: str = "<s>",
    eos_token: str = "</s>",
) -> Generator[dict[str, Any], dict[str, Any], None]:
    """Generate training data by yielding each row in the dataset split."""
    # We assume that the dataset is a HuggingFace dataset, and a DatasetDict
    # such that the dict has train, val, and test splits.
    if from_disk:
        ds = load_from_disk(dataset)
        ds = ds[split]
        # Iterate through the dataset and yield each row
        print(f"{ds.num_rows} rows in {split} split")
    else:
        ds = load_dataset(dataset, streaming=True, split=split)

    for row in iter(ds):
        if task == "text":
            text = f"{row['text']}"
            if not text.startswith(bos_token):
                text = f"{bos_token}{text}{eos_token}"
            yield {"text": text}
        elif task == "completion":
            # If the dataset is a 'completion' task dataset, we need to concatenate the prompt and completion
            text = f"{row['prompt']}{row['completion']}"
            if not text.startswith(bos_token):
                text = f"{bos_token}{text}{eos_token}"
            yield {
                "text": text,
                "prompt": row["prompt"],
                "completion": row["completion"],
            }
        else:
            raise Exception(f"Unknown task: {task}")


def peft_module_casting_to_bf16(model: AutoModelForCausalLM, args: dict[str, str]) -> None:
    """Cast the PEFT model to bf16."""
    for name, module in model.named_modules():
        if isinstance(module, LoraLayer):
            if args.get("bf16", "false").lower() == "true":
                module = module.to(torch.bfloat16)
        if "norm" in name:
            module = module.to(torch.float32)
        if any(x in name for x in ["lm_head", "embed_tokens", "wte", "wpe"]):
            if hasattr(module, "weight"):
                if args["bf16"] and module.weight.dtype == torch.float32:
                    module = module.to(torch.bfloat16)
