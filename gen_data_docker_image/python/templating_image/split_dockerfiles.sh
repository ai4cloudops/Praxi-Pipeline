#!/bin/bash

# Directory containing the files to be split
SOURCE_DIR="/home/ubuntu/Praxi-study/Praxi-Pipeline/gen_data_docker_image/python/dockerfiles"

# Base name for subdirectories
SUBDIR_BASE="subdir"

# Number of subdirectories
NUM_DIRS=5

# Create subdirectories if they do not exist
for i in $(seq 1 $NUM_DIRS); do
  mkdir -p "$SOURCE_DIR/$SUBDIR_BASE$i"
done

# Get a list of files in the source directory
files=($SOURCE_DIR/*)

# Calculate the number of files to distribute evenly
num_files=${#files[@]}
files_per_dir=$(( (num_files + NUM_DIRS - 1) / NUM_DIRS ))

# Distribute files
for ((i=0; i<num_files; i++)); do
  target_dir="$SOURCE_DIR/$SUBDIR_BASE$((i % NUM_DIRS + 1))"
  mv "${files[$i]}" "$target_dir"
done
