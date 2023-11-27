import os
from fire import Fire
import glob
import base64
import subprocess
from jinja2 import Environment, FileSystemLoader
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


# Function to base64 encode
def b64encode_filter(s):
    if s is not None:
        return base64.b64encode(s.encode()).decode()
    return None


def main(
    container_image: str,
    base_model_type: str,
    base_model_name: str,
    dataset_type: str,
    dataset_name: str,
    output_model_type: str,
    output_model_name: str,
):
    if not base_model_type:
        base_model_type = "hf"
    if not dataset_type:
        dataset_type = "hf"
    if not output_model_type:
        output_model_type = "hf"

    # Get variables
    assert container_image, "You need to provide container_image"
    hf_token = os.getenv("HF_TOKEN", "").encode("utf-8").decode("utf-8")
    s3_endpoint = os.getenv("S3_ENDPOINT", "").encode("utf-8").decode("utf-8")
    s3_access_key_id = os.getenv("S3_ACCESS_KEY_ID", "").encode("utf-8").decode("utf-8")
    s3_secret_access_key = (
        os.getenv("S3_SECRET_ACCESS_KEY", "").encode("utf-8").decode("utf-8")
    )
    s3_region = os.getenv("S3_REGION", "").encode("utf-8").decode("utf-8")
    s3_secure = os.getenv("S3_SECURE", "").encode("utf-8").decode("utf-8")
    wandb_api_key = os.getenv("WANDB_API_KEY", "").encode("utf-8").decode("utf-8")
    wandb_username = os.getenv("WANDB_USERNAME", "").encode("utf-8").decode("utf-8")
    wandb_project = os.getenv("WANDB_PROJECT", "").encode("utf-8").decode("utf-8")

    assert container_image, "You need to set CONTAINER_IMAGE env var"
    assert wandb_api_key, "You need to set WANDB_API_KEY env var"
    assert wandb_username, "You need to set WANDB_USERNAME env var"
    assert wandb_project, "You need to set WANDB_PROJECT env var"

    assert (
        base_model_type == "hf"
    ), "Only huggingface models are supported as base_model_type"

    if dataset_type == "s3":
        # We'll download the data from S3
        assert s3_endpoint, "You need to set S3_ENDPOINT env var"
        assert s3_access_key_id, "You need to set S3_ACCESS_KEY_ID env var"
        assert s3_secret_access_key, "You need to set S3_SECRET_ACCESS_KEY env var"
        assert s3_region, "You need to set S3_REGION env var"
    elif dataset_type == "hf":
        # We'll use either a public or private dataset from huggingface hub
        assert hf_token, "You need to set HF_TOKEN env var"
    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type}")

    if output_model_type == "s3":
        # We'll upload the model to S3
        assert s3_endpoint, "You need to set S3_ENDPOINT env var"
        assert s3_access_key_id, "You need to set S3_ACCESS_KEY_ID env var"
        assert s3_secret_access_key, "You need to set S3_SECRET_ACCESS_KEY env var"
        assert s3_region, "You need to set S3_REGION env var"
    elif output_model_type == "hf":
        # We'll upload the model to huggingface hub
        assert hf_token, "You need to set HF_TOKEN env var"
    else:
        raise ValueError(f"Unknown output_model_type: {dataset_type}")

    # Setup Jinja2 environment
    env = Environment(loader=FileSystemLoader("."))
    env.filters["b64encode"] = b64encode_filter

    # Directory containing the YAML files
    yaml_dir = os.path.join(os.path.dirname(__file__), "yamls")

    # Iterate through all yaml files in the 'yamls' directory
    for yaml_file in glob.glob(os.path.join(yaml_dir, "*.yaml")):
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
            base_model_type=base_model_type,
            base_model_name=base_model_name,
            dataset_type=dataset_type,
            dataset_name=dataset_name,
            output_model_type=output_model_type,
            output_model_name=output_model_name,
        )

        # Use subprocess.Popen with communicate to apply the Kubernetes configuration
        with subprocess.Popen(
            ["kubectl", "apply", "-f", "-"], stdin=subprocess.PIPE, text=True
        ) as proc:
            proc.communicate(rendered_template)

        print(f"Applied Kubernetes configuration from {yaml_file}")


if __name__ == "__main__":
    Fire(main)
