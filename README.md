# Praxi-Pipeline

## Overwiew
In NERC, new container images are being generated every day. 
However as a research testbed, many images might be using some outdated or vulnerable packages.
This issue can be hard to identify as many the researchers are focusing on the efficient or quick-delivery of their studies.
To this end, we consider developping an automation tool to periodically introspect the dependencies in the recently built images and alarm those can potentially damage the cluster as a community.

The core design is based on `Praxi`, a software discovery tool previously developped by the `AI-4-Cloud-Ops` group. https://github.com/peaclab/praxi
There are three steps for `Praxi` as a ML pipeline, i.e., collecting data (changeset/package installation fingerprints generation), data processing (tokenization) and prediction (feed tokens into `Vowpal Wabbit` model).
We consider extending and deploying the `Praxi Pipeline` using the recently introduced `RHODS pipeline` in NERC test cluster.
And if time permited, we plan to investigate and improve the performance of model inferencing with deployent plan optmization and LLM.

### Project Progress

https://docs.google.com/presentation/d/127wQdDaU1EWnZZRln63-D4_kqwltRWXAjWwCevHPhnk/edit?usp=sharing

<!-- We are working on modularize and containerize Praxi into a pipeline, specicially Kubeflow Pipeline. The goal is to adapt Praxi into RHODS in NERC and this repo shows how to adapt a monolithic application into a pipeline. And then the next step is to adapt this Kubeflow Pipeline into RHODS Pipeline. -->

## Design
We are developping two pipelines for such system to work, i.e., the training and inference pipeline. 
At the moment, we are focusing on Kubeflow Pipeline implementations and at the same time migrating our code to use RHODS Pipeline.
We expect the migration to be smooth as `KFP` is one layer of abstraction in `RHODS Pipeline`. 
https://docs.google.com/presentation/d/1XLuWAawt8w05rYHj8dNvcwK4Ywj0oSi3h_-QsnNM438/edit#slide=id.g1f1529e8b96_0_713

https://github.com/adrien-legros/rhods-mnist/blob/main/docs/lab-instructions.md#add-the-kubeflow-runtime

https://github.com/OCP-on-NERC/operations/issues/156

### Inference Pipeline
![Inference Diagram](https://github.com/Zongshun96/Praxi-Pipeline/blob/main/Figures/Praxi-Kubeflow-Image-layers-inference.drawi.drawio.png?raw=true)

For this inference pipeline, the processing begins from left to right.

First we design a component to pull images from the Docker Hub based on cluster observability, i.e., Advance Cluster Monitoring (Prometheus&Grafana).
We envision Docker Hub is one popular registry for the moment.
And when `RHODS Image Registry` delivered, adapting to other Registries should be mainly changing the API used in this component.

Second, we consider two sub-steps in generating the tagsets/fingerprint of an package installations.
First we read the installation DB from the image layer content (from the image tar ball).
For this step, we only consider Debian-based distros at the moment with `dpkg-query`.
And in the second sub-step we retrieve the pathnames of files changed in each layer as one changeset/installation fingerprint.

Thrid and forth steps are mostly similar to the original `Praxi`.
However, the inference component has to adapt according to the image tar design.
On one hand, since each image layer can have multiple package installations, so the model has to perform a multi-labels predictions (multiple outputs).
On the other hand, presumbly due to tagset data distribution (tagset size and distinguishable tags) naive confidence threshold based method didn't provide a high f1-scores (Our reproduced results is at about 0.7).
In `Praxi`, they consider clustering the modification timestamps of each files and infer the amount of labels to emit.
This method helped to improve the f1-score to be around 0.8 and not hyper-parameter to tune.
But since tar files only keeps 1 second granularity of file modification timestamps.
It becomes hard to identify clusterings for installations, as many individual package installations won't take over multiple seconds.
This lead us back to the threholding method but we are also developing a different emiting strategy to compete with the original Praxi design.
We apply density based clustering algorithm to model's output confidence and we emit labels with higher confidence than the highest confidence in the biggest clusters.
This is motivated by the two observations.
One, the absolute confidence of each label is different among test tagsets, which means a hard coded threhold won't give us good results from time to time.
Instead, the difference in confidences between true and false labels is more stable.
Two, the difference in confidences within false labels are similar while the true labels are more diverse.
By doing so, we were able to get around 0.8 f1-scores.
And sometimes simply threholding the difference from worst confidence to the the threhold also gives us about 0.8 f1-scores.

https://docs.google.com/presentation/d/1InksmODphuhfZm3hjezXCkQiLWdlsklHmpmNMjWttp8/edit?usp=sharing

https://docs.google.com/presentation/d/1qg1_n8iY0Jmk4pz7u-X0WI6kAqsivDHC492v0ii-fDI/edit?usp=sharing

### Train Pipeline
![Train Diagram](https://github.com/Zongshun96/Praxi-Pipeline/blob/main/Figures/Praxi-Kubeflow-train.drawi.drawio.png?raw=true)

Training step is more original to `Praxi`.
We still have the three components (changeset generator, tagset generator, VW model) in Kubeflow Pipeline.
And we want to pair a VM to select package installations based on the observability in the cluster.


<!-- There are two components are under development, i.e., `taggen` and `prediction`. 
`taggen` will mount a volume (currently a manual persistent volume in k8s) and load the difference from it (later we need a method to load differences from layers in snapshot) to generate `tags` with `columbus`.
`prediction` will take the `tags` generated in the `taggen` step and make prediction based on a trained `vw` model and write its output into a persistent volume (how to reply directly to user is unknown).

We have to define the container images for `taggen` and `prediction` components, as there are serveral dependancies cannot be installed with pip, i.e., `columbus` and `vw`. The correponding files are in the two directories, i.e., `taggen_base_image` and `prediciton_base_image`.

![Design Diagram](https://github.com/Zongshun96/Praxi-Pipeline/blob/main/Figures/Praxi-Kubeflow-agentless-PoC.drawio.png?raw=true) -->


## Running (Updating)
- Install kubeflow and sdk.

Some helpful tutorials.

https://docs.google.com/document/d/1LHMrcQJQWiq1pLyEHZPJScovWNdLJduE9e6BU-F_J_Y/edit

https://www.kubeflow.org/docs/components/pipelines/v1/sdk/install-sdk/

- Create a persistent volume.

```
kubectl create -f 'fake-snapshot-volume/fake-snapshot-pv.yaml'
```
Now you can see there is a new pv with `kubectl get pv`. We will use the name of this new pv later.

- Copy the changesets into the persistent volume (the volume should contain the changes in snapshot layers after we figure that out).

```
kubectl create -f 'fake-snapshot-volume/fake-deployment.yaml'

kubectl cp data/changesets root@{PODNAME}:/fake-snapshot/

kubectl delete -f 'fake-snapshot-volume/fake-deployment.yaml'

kubectl patch pv {PVNAME} -p '{"spec":{"claimRef": null}}' # see https://stackoverflow.com/questions/50667437/what-to-do-with-released-persistent-volume
```

References:

https://gist.githubusercontent.com/salayatana66/ceef347ec8f082bf08c1328e7a880407/raw/f02a7bd6e6662297605bab3d697c120c0e666be4/vw-29-apr-minikube6.yaml

https://towardsdatascience.com/serving-vowpal-wabbit-models-on-kubernetes-de0ed968f3b0

https://stackoverflow.com/questions/50667437/what-to-do-with-released-persistent-volume


- Build the pipeline component images.

There is a bash script `build.sh` in each base-image dir. E.g., for `prediction-base-image`,
```
cd prediction-base-image
bash build.sh
```

References

https://www.kubeflow.org/docs/components/pipelines/v1/sdk/python-function-components/#packages

https://www.kubeflow.org/docs/components/pipelines/v1/sdk/component-development/


- Package the pipeline and load it into Kubeflow Pipeline.

Run `Praxi-Pipeline.py`. It defines the function to run inside each base image,  the dataflow between components (base image + the Praxi function) and runs the pipeline in kubeflow. Please refer to [design section](#design) for the dataflow design.

References

https://github.com/kubeflow/pipelines/blob/sdk/release-1.8/samples/tutorials/Data%20passing%20in%20python%20components.ipynb

https://github.com/kubeflow/pipelines/issues/7667


- Check your kubeflow pipeline dashboard.

A new run will show up in your pipeline dashboard.

<!-- ![screenshot for a new run in pipeline dashboard](https://github.com/[username]/[reponame]/blob/[branch]/image.jpg?raw=true) -->




## Notes
Some test changesets are provided in `data/`.

The persistent volume yaml is provided in `fake-snapshot-volume/`.

Automatically delete PVC generated by pipeline when a run is deleted.
https://github.com/kubeflow/pipelines/issues/6649

