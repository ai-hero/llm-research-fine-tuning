# fine_tune
Code meant to be deployed as a container image on K8s to fine-tune a model

## Building container
```sh
docker build . -t rparundekar/fine_tune_research:20231209_01 \
    && docker push rparundekar/fine_tune_research:20231209_01
```
## Config for the k8s job
Update [k8s/yamls/config.yaml](k8s/yamls/config.yaml).

## Launching k8s job
```sh
cd k8s/
python train.py launch rparundekar/fine_tune_research:20231209_01 alpaca_peft.yaml
```
You'll see the name of the job. If launching with a distributed config
```sh
cd k8s/
python train.py launch rparundekar/fine_tune_research:20231209_01 distributed_default.yaml -d fsdp_single_worker.yaml
```


## Deleting a job
```sh
python train.py delete <job-name>
```

## Code QA
```sh
python3.9 -m venv venv
source venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
pre-commit install
pre-commit autoupdate
pre-commit run --all-files
```
