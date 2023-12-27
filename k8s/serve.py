"""Script used to launch a Kubernetes job for training a model."""
import base64
import glob
import os
import subprocess

import yaml  # type: ignore
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


def launch(config_file: str = "mmlu_peft.yaml") -> None:
    """Launch a Kubernetes service for serving the model."""
    hf_token = os.getenv("HF_TOKEN", "")
    s3_endpoint = os.getenv("S3_ENDPOINT", "")
    s3_access_key_id = os.getenv("S3_ACCESS_KEY_ID", "")
    s3_secret_access_key = os.getenv("S3_SECRET_ACCESS_KEY", "")
    s3_region = os.getenv("S3_REGION", "")
    s3_secure = os.getenv("S3_SECURE", "")
    wandb_api_key = os.getenv("WANDB_API_KEY", "")
    wandb_username = os.getenv("WANDB_USERNAME", "")

    assert wandb_api_key, "You need to set WANDB_API_KEY env var"
    assert wandb_username, "You need to set WANDB_USERNAME env var"

    # Setup Jinja2 environment
    env = Environment(loader=FileSystemLoader("."))
    env.filters["b64encode"] = b64encode_filter

    # Directory containing the YAML files
    yaml_dir = os.path.join(os.path.dirname(__file__), "yamls", "serving")

    # Load training config file and extract dataset name
    with open(os.path.join(os.path.dirname(__file__), "configs", config_file)) as f:
        training_config = yaml.safe_load(f)

    assert training_config["model"]["output"]["type"] == "hf", "Only hf models are supported for serving"
    model_name = training_config["model"]["output"]["name"]
    app_name = model_name.split("/")[-1].lower()
    project_name = training_config["project"]["name"]

    # Iterate through all yaml files in the 'yamls' directory
    for yaml_file in glob.glob(os.path.join(yaml_dir, "*.yaml")):
        # Load the template
        template = env.get_template(os.path.relpath(yaml_file))
        # Render the template with environment variables
        rendered_template = template.render(
            project_name=project_name,
            app_name=app_name,
            model_name=model_name,
            s3_endpoint=s3_endpoint,
            s3_access_key_id=s3_access_key_id,
            s3_secret_access_key=s3_secret_access_key,
            s3_region=s3_region,
            s3_secure=f"{s3_secure}",
            hf_token=hf_token,
            config_file=config_file,
        )
        if "config_template.yaml" == yaml_file.split("/")[-1]:
            # Set the training config as the string value for config map
            config = yaml.safe_load(rendered_template)
            config["data"]["config.yaml"] = yaml.dump(training_config)
            rendered_template = yaml.dump(config)

        # Use subprocess.Popen with communicate to apply the Kubernetes configuration
        with subprocess.Popen(["kubectl", "apply", "-f", "-"], stdin=subprocess.PIPE, text=True) as proc:
            proc.communicate(rendered_template)

    print(f"Applied Kubernetes configuration from {yaml_file}")

    print(f"Launched service name: {app_name}")
    print(f"1. To see status, run: kubectl describe deployment/{app_name}")
    print(f"2. To see logs, run: kubectl logs deployment/{app_name} -f")
    print(f"3. To port forward, run: kubectl port-forward deployment/{app_name} 8080:80")
    print(f"4. To delete job and other artifacts, run: python serve.py delete {app_name}")


def delete(app_name: str) -> None:
    """Delete a Kubernetes job and other artifacts."""
    assert app_name, "You need to provide app_name"
    # Use subprocess.Popen with communicate to delete the Kubernetes job
    with subprocess.Popen(["kubectl", "delete", "deployment", app_name], stdin=subprocess.PIPE, text=True) as proc:
        proc.communicate()
    print(f"Deleted Kubernetes deployment {app_name}")
    with subprocess.Popen(["kubectl", "delete", "service", app_name], stdin=subprocess.PIPE, text=True) as proc:
        proc.communicate()
    print(f"Deleted Kubernetes service {app_name}")

    # Use subprocess.Popen with communicate to delete the Kubernetes job
    with subprocess.Popen(["kubectl", "delete", "secret", app_name], stdin=subprocess.PIPE, text=True) as proc:
        proc.communicate()
    print(f"Deleted Kubernetes secret {app_name}")


if __name__ == "__main__":
    Fire(
        {
            "launch": launch,
            "delete": delete,
        }
    )
