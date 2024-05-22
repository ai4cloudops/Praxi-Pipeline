#!/bin/bash

# Define the directory containing the files. Use "." for the current directory
dir="/home/cc/Praxi-study/Praxi-Pipeline/data/data_4/tagsets_SL_test"

# Loop through all files in the directory
for file in "$dir"/*; do
  # Check if it's a regular file (not a directory)
  if [ -f "$file" ]; then
    # Construct the new filename by appending "_copy" before the file extension
    filename=$(basename -- "$file")
    extension="${filename##*.}"
    filename="${filename%.*}"
    newfile="${dir}/${filename}_copy.${extension}"

    # Copy the file to the new filename
    cp "$file" "$newfile"
  fi
done

echo "All files have been duplicated."
