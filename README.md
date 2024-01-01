# Fine-Tuning for LLM Research by AI Hero

Container Code used for the fine-tuning on top of Kubernetes. This repo contains the code that will be run inside the container. The container is built and pushed to the repo using Github actions (see below). You can launch the fine tuning job using the `llm-research-fine-tuning` project with the data created with the `llm-research-data` project.

## Setup
```sh
pip install -r requirements.txt
```

## Building Docker using Github Actions
Change the Github actions in the `.github` folder and set the right environment variables in Github to auto build the container and push to the right repo.
