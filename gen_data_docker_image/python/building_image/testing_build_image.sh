#!/bin/bash

# Directory containing the Dockerfiles
dockerfiles_dir="/home/cc/Praxi-study/Praxi-Pipeline/gen_data_docker_image/python/dockerfiles"

# Loop through all Dockerfiles in the specified directory
for dockerfile in $dockerfiles_dir/Dockerfile.*; do
    # Extract a name to use as the image tag from the Dockerfile's name
    image_name=$(basename "$dockerfile" | sed 's/Dockerfile\.//')
    
    # Build the Docker image
    sudo docker build -f "$dockerfile" -t "$image_name" .
    
    echo "Built image $image_name"
done
