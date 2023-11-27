# fine_tune
Code meant to be deployed as a container image on K8s to fine-tune a model

## Building container
```sh
docker build .  -t rparundekar/fine_tune_research:202311221140
docker push rparundekar/fine_tune_research:202311221140
```

## Launching k8s job
```sh
cd k8s/
python launch.py --container_image=rparundekar/fine_tune_research:202311221140 --base_model_type=hf --base_model_name="tiiuae/falcon-7b" --dataset_type=s3 --dataset_name="fine-tuning-research/mmlu_dataset" --output_model_type=hf --output_model_name="sadmoseby/falcoln-7b-peft-train-intermediate" 
```

OR
```sh
python launch.py --container_image=rparundekar/fine_tune_research:202311221140 --base_model_type=hf --base_model_name="tiiuae/falcon-7b" --dataset_type=hf --dataset_name="heliosbrahma/mental_health_chatbot_dataset" --dataset_training_column="text" --output_model_type=hf --output_model_name="sadmoseby/falcoln-7b-peft-train-intermediate" 
```

## Deleting a job
```sh
kubectl delete job peft-job
```