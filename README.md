# Fine-Tuning for LLM Research by AI Hero

This repo contains the code that will be run inside the container. Alternatively, this code can also be run natively. The container is built and pushed to the repo using Github actions (see below). You can launch the fine tuning job using the examples in the `https://github.com/ai-hero/llm-research-examples` project with the data created with the `https://github.com/ai-hero/llm-research-data` project.

## Container
Our latest container we use for training is `rparundekar/fine_tune_research:{SHORT_SHA_ON_MAIN}`. You can launch jobs using this tag with the `llm-research-examples` project.

## For Contributors
Installing this library locally.

```sh
pip install .
```

### Building Docker using Github Actions
Change the Github actions in the `.github` folder and set the right environment variables in Github to auto build the container and push to the right repo.

### Building a Docker Image Manually
Use a tag versioning by date / user as needed. For example,
```sh
docker build . -t rparundekar/fine_tune_research:20230110_01
docker push rparundekar/fine_tune_research:20230110_01
```
