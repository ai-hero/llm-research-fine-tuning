import os
from fire import Fire
import glob
import base64
import subprocess
from jinja2 import Environment, FileSystemLoader
from dotenv import load_dotenv
import yaml
import time
from codenamize import codenamize

# Load environment variables
load_dotenv()


# Function to base64 encode
def b64encode_filter(s):
    if s is not None:
        return base64.b64encode(s.encode()).decode()
    return None


def train(container_image: str, config_file: str = "guanaco_peft.yaml"):
    job_name = codenamize(f"{config_file}-{time.time()}")
    print(f"Job name: {job_name}")

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
    wandb_project = os.getenv("WANDB_PROJECT", "")

    assert container_image, "You need to set CONTAINER_IMAGE env var"
    assert wandb_api_key, "You need to set WANDB_API_KEY env var"
    assert wandb_username, "You need to set WANDB_USERNAME env var"
    assert wandb_project, "You need to set WANDB_PROJECT env var"

    # Setup Jinja2 environment
    env = Environment(loader=FileSystemLoader("."))
    env.filters["b64encode"] = b64encode_filter

    # Directory containing the YAML files
    yaml_dir = os.path.join(os.path.dirname(__file__), "yamls")

    # Iterate through all yaml files in the 'yamls' directory
    for yaml_file in glob.glob(os.path.join(yaml_dir, "*.yaml")):
        # Load training config file and extract dataset name
        with open(
            os.path.join(os.path.dirname(__file__), "configs", config_file), "r"
        ) as f:
            training_config = yaml.safe_load(f)
        dataset_name = training_config["dataset"]["name"]
        wandb_tags = (
            f"{os.getenv('USER',os.getenv('USERNAME'))},{job_name},{dataset_name}"
        )

        # Load the template
        template = env.get_template(os.path.relpath(yaml_file))
        # Render the template with environment variables
        rendered_template = template.render(
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
            wandb_project=wandb_project,
            wandb_job_name=job_name,
            wandb_tags=wandb_tags,
            config_file=config_file,
        )
        if "config.yaml" == yaml_file.split("/")[-1]:
            # Set the training config as the string value for config map
            config = yaml.safe_load(rendered_template)
            config["data"]["config.yaml"] = yaml.dump(training_config)
            rendered_template = yaml.dump(config)

        # Use subprocess.Popen with communicate to apply the Kubernetes configuration
        with subprocess.Popen(
            ["kubectl", "apply", "-f", "-"], stdin=subprocess.PIPE, text=True
        ) as proc:
            proc.communicate(rendered_template)

    print(f"Applied Kubernetes configuration from {yaml_file}")

    print(f"Launched job name: {job_name}")
    print(f"1. To see status, run: kubectl describe job/{job_name}")
    print(f"2. To see logs, run: kubectl logs job/{job_name} -f")
    print(
        f"3. To delete job, run: python launch.py delete {job_name} (to delete all artifacts)"
    )


def delete(job_name: str):
    assert job_name, "You need to provide job_name"
    # Use subprocess.Popen with communicate to delete the Kubernetes job
    with subprocess.Popen(
        ["kubectl", "delete", "job", job_name], stdin=subprocess.PIPE, text=True
    ) as proc:
        proc.communicate()
    print(f"Deleted Kubernetes job {job_name}")

    # Use subprocess.Popen with communicate to delete the Kubernetes job
    with subprocess.Popen(
        ["kubectl", "delete", "service", job_name], stdin=subprocess.PIPE, text=True
    ) as proc:
        proc.communicate()
    print(f"Deleted Kubernetes service {job_name}")

    # Use subprocess.Popen with communicate to delete the Kubernetes job
    with subprocess.Popen(
        ["kubectl", "delete", "configmap", job_name], stdin=subprocess.PIPE, text=True
    ) as proc:
        proc.communicate()
    print(f"Deleted Kubernetes configmap {job_name}")


if __name__ == "__main__":
    Fire(
        {
            "train": train,
            "delete": delete,
        }
    )
