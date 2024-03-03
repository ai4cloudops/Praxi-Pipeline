#!/bin/bash

# Directory where you want to search for files
SEARCH_DIR="/home/cc/Praxi-study/Praxi-Pipeline/gen_data_docker_image/python/templating_image/dockerfiles_failed"

# End time (files modified before this time will be considered)
# Format: YYYY-MM-DD HH:MM:SS
END_TIME="2024-02-25 22:03:00"

# Time interval in seconds (e.g., 10 seconds before the END_TIME)
INTERVAL=25

# Calculate start time
# START_TIME=$(date -d "$END_TIME - $INTERVAL seconds" +"%Y-%m-%d %H:%M:%S")
START_TIME="2024-02-25 22:01:00"
# echo $START_TIME $END_TIME

# Find and delete files modified within the specified time range
# find "$SEARCH_DIR" -type f -newermt "$START_TIME" ! -newermt "$END_TIME"
find "$SEARCH_DIR" -type f -newermt "$START_TIME" ! -newermt "$END_TIME" -exec rm {} \;
