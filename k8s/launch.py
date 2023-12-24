"""Script used to launch a Kubernetes job for training a model."""
import base64
import glob
import os
import subprocess
import time

import yaml  # type: ignore
from codenamize import codenamize
from dotenv import load_dotenv
from fire import Fire
from jinja2 import Environment, FileSystemLoader

# Load environment variables
load_dotenv()


# Function to base64 encode
def b64encode_filter(s: str) -> str:
    """Base64 encode a string. Used in Jinja2 template."""
    if s is not None:
        return base64.b64encode(s.encode()).decode()
    return None


def train(container_image: str, config_file: str = "guanaco_peft.yaml", distributed_config_file: str = "") -> None:
    """Launch a Kubernetes job for training a model."""
    job_name = codenamize(f"{config_file}-{time.time()}")
    print(f"Job name: {job_name}")
    num_gpu = ""

    # Get variables
    assert container_image, "You need to provide container_image"
    hf_token = os.getenv("HF_TOKEN", "")
    s3_endpoint = os.getenv("S3_ENDPOINT", "")
    s3_access_key_id = os.getenv("S3_ACCESS_KEY_ID", "")
    s3_secret_access_key = os.getenv("S3_SECRET_ACCESS_KEY", "")
    s3_region = os.getenv("S3_REGION", "")
    s3_secure = os.getenv("S3_SECURE", "")
    wandb_api_key = os.getenv("WANDB_API_KEY", "")
    wandb_username = os.getenv("WANDB_USERNAME", "")

    assert container_image, "You need to set CONTAINER_IMAGE env var"
    assert wandb_api_key, "You need to set WANDB_API_KEY env var"
    assert wandb_username, "You need to set WANDB_USERNAME env var"

    # Setup Jinja2 environment
    env = Environment(loader=FileSystemLoader("."))
    env.filters["b64encode"] = b64encode_filter

    # Directory containing the YAML files
    yaml_dir = os.path.join(os.path.dirname(__file__), "yamls")

    # Load training config file and extract dataset name
    with open(os.path.join(os.path.dirname(__file__), "configs", config_file)) as f:
        training_config = yaml.safe_load(f)
    if distributed_config_file:
        with open(os.path.join(os.path.dirname(__file__), "distributed", distributed_config_file)) as f:
            distributed_training_config = yaml.safe_load(f)
            num_gpu = distributed_training_config["num_processes"]

    dataset_name = training_config["dataset"]["name"]
    project_name = training_config["project"]["name"]
    wandb_tags = f"{os.getenv('USER',os.getenv('USERNAME'))},{job_name},{dataset_name}"

    # Iterate through all yaml files in the 'yamls' directory
    for yaml_file in glob.glob(os.path.join(yaml_dir, "*.yaml")):
        # Load the template
        template = env.get_template(os.path.relpath(yaml_file))
        # Render the template with environment variables
        rendered_template = template.render(
            project_name=project_name,
            job_name=job_name,
            container_image=container_image,
            s3_endpoint=s3_endpoint,
            s3_access_key_id=s3_access_key_id,
            s3_secret_access_key=s3_secret_access_key,
            s3_region=s3_region,
            s3_secure=f"{s3_secure}",
            hf_token=hf_token,
            wandb_api_key=wandb_api_key,
            wandb_username=wandb_username,
            wandb_job_name=job_name,
            wandb_tags=wandb_tags,
            config_file=config_file,
            num_gpu=num_gpu,
        )
        if "config_template.yaml" == yaml_file.split("/")[-1]:
            # Set the training config as the string value for config map
            config = yaml.safe_load(rendered_template)
            config["data"]["config.yaml"] = yaml.dump(training_config)
            if num_gpu:
                config["data"]["distributed.yaml"] = yaml.dump(distributed_training_config)
            rendered_template = yaml.dump(config)

        # Use subprocess.Popen with communicate to apply the Kubernetes configuration
        with subprocess.Popen(["kubectl", "apply", "-f", "-"], stdin=subprocess.PIPE, text=True) as proc:
            proc.communicate(rendered_template)

    print(f"Applied Kubernetes configuration from {yaml_file}")

    print(f"Launched job name: {job_name}")
    print(f"1. To see status, run: kubectl describe job/{job_name}")
    print(f"2. To see logs, run: kubectl logs job/{job_name} -f")
    print(f"3. To delete job and other artifacts, run: python launch.py delete {job_name}")


def delete(job_name: str) -> None:
    """Delete a Kubernetes job and other artifacts."""
    assert job_name, "You need to provide job_name"
    # Use subprocess.Popen with communicate to delete the Kubernetes job
    with subprocess.Popen(["kubectl", "delete", "job", job_name], stdin=subprocess.PIPE, text=True) as proc:
        proc.communicate()
    print(f"Deleted Kubernetes job {job_name}")

    # Use subprocess.Popen with communicate to delete the Kubernetes job
    with subprocess.Popen(["kubectl", "delete", "service", job_name], stdin=subprocess.PIPE, text=True) as proc:
        proc.communicate()
    print(f"Deleted Kubernetes service {job_name}")

    # Use subprocess.Popen with communicate to delete the Kubernetes job
    with subprocess.Popen(["kubectl", "delete", "configmap", job_name], stdin=subprocess.PIPE, text=True) as proc:
        proc.communicate()
    print(f"Deleted Kubernetes configmap {job_name}")


if __name__ == "__main__":
    Fire(
        {
            "train": train,
            "delete": delete,
        }
    )
