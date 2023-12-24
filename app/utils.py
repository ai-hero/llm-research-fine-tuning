"""Utility functions for the app. e.g. upload and download files from S3."""
import os
import tarfile

import torch
import yaml
from minio import Minio, S3Error
from peft.tuners.lora import LoraLayer
from transformers import AutoModelForCausalLM

DEFAULT_STATIC_CONFIG_PATH = "./default_config.yaml"
MOUNTED_CONFIG_PATH = "/mnt/config/training/config.yaml"


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


def load_config() -> dict[str, str]:
    """Load the application configuration."""
    config: dict[str, str] = {}
    if os.path.exists(MOUNTED_CONFIG_PATH):
        config_file = MOUNTED_CONFIG_PATH
        print("Loading mounted config")
    else:
        config_file = DEFAULT_STATIC_CONFIG_PATH
        print("Loading default config")
    with open(file=config_file, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def peft_module_casting_to_bf16(model: AutoModelForCausalLM, args: dict[str, str]) -> None:
    """Cast the PEFT model to bf16."""
    for name, module in model.named_modules():
        if isinstance(module, LoraLayer):
            if args["bf16"]:
                module = module.to(torch.bfloat16)
        if "norm" in name:
            module = module.to(torch.float32)
        if any(x in name for x in ["lm_head", "embed_tokens", "wte", "wpe"]):
            if hasattr(module, "weight"):
                if args["bf16"] and module.weight.dtype == torch.float32:
                    module = module.to(torch.bfloat16)


def dump_envs() -> None:
    """Dump the application relevant environment variables."""
    print("Training LOCAL RANK: {} ...".format(os.getenv("LOCAL_RANK", "Unknown")))
    print("Training RANK: {} ...".format(os.getenv("RANK", "Unknown")))
    print("Training LOCAL WORLD SIZE: {} ...".format(os.getenv("LOCAL_WORLD_SIZE", "Unknown")))
