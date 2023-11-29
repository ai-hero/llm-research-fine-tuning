import os
from fire import Fire
import glob
import base64
import subprocess
from jinja2 import Environment, FileSystemLoader
from dotenv import load_dotenv
import yaml

# Load environment variables
load_dotenv()


# Function to base64 encode
def b64encode_filter(s):
    if s is not None:
        return base64.b64encode(s.encode()).decode()
    return None


def main(container_image: str, config_file: str = "guanaco_peft.yaml"):
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
        if "config.yaml" == yaml_file.split("/")[-1]:
            with open(
                os.path.relpath(yaml_file),
                "r",
            ) as f:
                config = yaml.safe_load(f)
            with open(
                os.path.join(os.path.dirname(__file__), "configs", config_file), "r"
            ) as f:
                training_config = yaml.safe_load(f)
            config["data"]["config.yaml"] = yaml.dump(training_config)
            rendered_template = yaml.dump(config)
        else:
            # Load the template
            template = env.get_template(os.path.relpath(yaml_file))
            # Render the template with environment variables
            rendered_template = template.render(
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
                config_file=config_file,
            )

        # Use subprocess.Popen with communicate to apply the Kubernetes configuration
        with subprocess.Popen(
            ["kubectl", "apply", "-f", "-"], stdin=subprocess.PIPE, text=True
        ) as proc:
            proc.communicate(rendered_template)

        print(f"Applied Kubernetes configuration from {yaml_file}")


if __name__ == "__main__":
    Fire(main)
