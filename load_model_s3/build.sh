#!/bin/bash -e
image_name=zongshun96/load_model_s3
image_tag=0.01
full_image_name=${image_name}:${image_tag}

# cd "$(dirname "$0")" 
docker build -t "registry-route-ai4cloudops-11855c.apps.shift.nerc.mghpcc.org/${full_image_name}" .
docker push "registry-route-ai4cloudops-11855c.apps.shift.nerc.mghpcc.org/$full_image_name"

# Output the strict image name, which contains the sha256 image digest
sudo docker inspect --format="{{index .RepoDigests 0}}" "${full_image_name}"