# Praxi-Pipeline

## Overwiew
In [NERC](https://nerc-project.github.io/nerc-docs/), new container images are being generated every day. 
However as a research testbed, many images might be using some outdated or vulnerable packages.
This issue can be hard to identify as many the researchers are focusing on the efficient or quick-delivery of their studies.
To this end, we consider developping an automation tool to periodically introspect the dependencies in the recently built images and alarm those can potentially damage the cluster as a community.

The core design is motivated by `Praxi`, a software discovery tool previously developped by the `AI-4-Cloud-Ops` group. https://github.com/peaclab/praxi
We collect data (package installation fingerprints generation), tokenize data and generate predictions.
Furthermore, we build a decomposible ML system for efficiently incorperating new packages for discovery and minimize incremental training cost.
The ML system is deployed and evaluated using Openshift Data Science Pipelines (Kubeflow Pipelines) in NERC.

### Project Progress

https://docs.google.com/presentation/d/127wQdDaU1EWnZZRln63-D4_kqwltRWXAjWwCevHPhnk/edit?usp=sharing


### Inference Pipeline
![Inference Diagram](https://github.com/Zongshun96/Praxi-Pipeline/blob/main/Figures/Praxi-Kubeflow-Image-layers-inference-without-installation-DB.drawi.drawio.png?raw=true)

For this inference pipeline, the processing begins from left to right.

First we design a component to pull images from the Docker Hub based on cluster observability, i.e., Advance Cluster Monitoring (Prometheus&Grafana).
We envision Docker Hub is one popular registry for the moment.
And when `RHODS Image Registry` delivered, adapting to other Registries should be mainly changing the API used in this component.

Second, we generate the installation fingerprint by tokenize pathnames of file changes in each image layer.

Third, we generate predictions by feeding fingerprints to our pretrained Mixture of Expert motivated model

The detailed steps are shown in `Praxi-Pipeline/Praxi-Pipeline-xgb.py`.

## Running

Installing dependencies
```
pip install -r Praxi-Pipeline/requirements.txt
```

### Openshift AI Deployment



In `Praxi-Pipeline/Praxi-Pipeline-xgb.py`,

Configure `kubeflow_endpoint` and `bearer_token` to access the Openshift AI endpoint.

Configure `aws_access_key_id` and `aws_secret_access_key` to save predictions.

Run 
```
python3 Praxi-Pipeline/Praxi-Pipeline-xgb.py
```

### Model Training and Testing Scripts

In `Praxi-Pipeline/prediction_XGBoost_openshift_image/function`, model training and testing scripts are categorized by different package routing methods, i.e., random assignment (`nover`), cosin similarity based assignment (`clustering`) and package version clustering (`verpak`).

The packages used for model training as in the paper is listed in 
```
Praxi-Pipeline/data/data4/index_label_mapping
```
We also have the trained models with `1` or `1000` submodels in NERC object storge.

Examples:

To train package version clustering based model,
```
python Praxi-Pipeline/prediction_XGBoost_openshift_image/function/verpak/tagsets_XGBoost_pickCVbatch.py
```

For package version clustering based model, to test with expert selection,
```
python Praxi-Pipeline/prediction_XGBoost_openshift_image/function/verpak/tagsets_XGBoost_pickCVbatch_on_demand_expert_selector.py
```

For package version clustering based model, to test with a hybrid FaaS and IaaS Flask setup with expert selection,
```
python Praxi-Pipeline/prediction_XGBoost_openshift_image/function/verpak/tagsets_XGBoost_pickCVbatch_on_demand_expert_selector_flask_client.py
```

To calculate the cosine similarity of packages in submodels,
```
python Praxi-Pipeline/prediction_XGBoost_openshift_image/function/verpak/tagsets_XGBoost_pickCVbatch_model_share_token_verpak.py
```


Some test kfp examples
https://github.com/rh-datascience-and-edge-practice/kubeflow-examples/blob/main/pipelines/1_test_connection.py

