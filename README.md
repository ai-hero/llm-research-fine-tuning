# fine_tune
Code meant to be deployed as a container image on K8s to fine-tune a model

## Building container
```sh
docker build . -t rparundekar/fine_tune_research:20231129_08 \
    && docker push rparundekar/fine_tune_research:20231129_08
```
## Config for the k8s job
Update [k8s/yamls/config.yaml](k8s/yamls/config.yaml).

## Launching k8s job
```sh
cd k8s/
python launch.py train rparundekar/fine_tune_research:20231129_08 alpaca_peft.yaml
```
You'll see the name of the job. 

## Deleting a job
```sh
python launch.py delete <job-name>
```