#!/bin/bash -e
image_name=zongshun96/prediction_xgb_openshift
image_tag=1.0
full_image_name=${image_name}:${image_tag}

# cd "$(dirname "$0")" 
docker build -t "registry-route-ai4cloudops-11855c.apps.shift.nerc.mghpcc.org/${full_image_name}" .
docker push "registry-route-ai4cloudops-11855c.apps.shift.nerc.mghpcc.org/$full_image_name"

# Output the strict image name, which contains the sha256 image digest
#docker inspect --format="{{index .RepoDigests 0}}" "${full_image_name}"