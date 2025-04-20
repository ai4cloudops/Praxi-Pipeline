# Re-Tag images to local registry
docker tag zongshun96/python3_9_18-bookworm.poetry_v1_8_2_p_webcolors_v1_13 /zongshun96/python3_9_18-bookworm.poetry_v1_8_2_p_webcolors_v1_13
docker push /zongshun96/python3_9_18-bookworm.poetry_v1_8_2_p_webcolors_v1_13

# List Registry Images
curl -X GET -u 'myuser:!QAZ2wsx' https:///v2/_catalog
curl -X GET -u 'myuser:!QAZ2wsx' https:///v2/zongshun96/python3_9_18-bookworm.poetry_v1_8_2_p_webcolors_v1_13/tags/list


export REGISTRY_USER="myuser"
export REGISTRY_PASS="!QAZ2wsx"