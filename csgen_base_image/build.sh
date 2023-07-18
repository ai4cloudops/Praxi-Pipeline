#!/bin/bash -e
image_name=lkorver/praxi-cs-base
image_tag=0.01
full_image_name=${image_name}:${image_tag}

# cd "$(dirname "$0")" 
docker build -t "${full_image_name}" .
docker push "$full_image_name"

# Output the strict image name, which contains the sha256 image digest
docker inspect --format="{{index .RepoDigests 0}}" "${full_image_name}"