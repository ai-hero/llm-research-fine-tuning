import os
import io
import glob
import base64
import subprocess
from jinja2 import Environment, FileSystemLoader
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get variables
wandb_api_key = os.getenv('WANDB_API_KEY')
wandb_username = os.getenv('WANDB_USERNAME')
wandb_project = os.getenv('WANDB_PROJECT')

assert wandb_api_key, "You need to set WANDB_API_KEY env var"
assert wandb_username, "You need to set WANDB_USERNAME env var"
assert wandb_project, "You need to set WANDB_PROJECT env var"


# Function to base64 encode
def b64encode_filter(s):
    if s is not None:
        return base64.b64encode(s.encode()).decode()
    return None

# Setup Jinja2 environment
env = Environment(loader=FileSystemLoader('.'))
env.filters['b64encode'] = b64encode_filter

# Directory containing the YAML files
yaml_dir = os.path.join(os.path.dirname(__file__), 'yamls')

# Iterate through all YAML files in the 'yamls' directory
for yaml_file in glob.glob(os.path.join(yaml_dir, '*.yaml')):
    # Load the template
    template = env.get_template(os.path.relpath(yaml_file))

    # Render the template with environment variables
    rendered_template = template.render(
        wandb_api_key=wandb_api_key.encode('utf-8').decode('utf-8'),
        wandb_username=wandb_username.encode('utf-8').decode('utf-8'),
        wandb_project=wandb_project.encode('utf-8').decode('utf-8')
    )

    # Use subprocess.Popen with communicate to apply the Kubernetes configuration
    with subprocess.Popen(["kubectl", "apply", "-f", "-"], stdin=subprocess.PIPE, text=True) as proc:
        proc.communicate(rendered_template)

    print(f"Applied Kubernetes configuration from {yaml_file}")