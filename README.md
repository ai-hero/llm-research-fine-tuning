# fine_tune
Code meant to be deployed as a container image on K8s to fine-tune a model

## Building container
```sh
docker build . -t rparundekar/fine_tune_research:20231129_01 \
    && docker push rparundekar/fine_tune_research:20231129_01
```
## Config for the k8s job
Update [k8s/yamls/config.yaml](k8s/yamls/config.yaml).

## Launching k8s job
```sh
cd k8s/
python launch.py rparundekar/fine_tune_research:20231129_01
```

## Deleting a job
```sh
kubectl delete job training-job
```