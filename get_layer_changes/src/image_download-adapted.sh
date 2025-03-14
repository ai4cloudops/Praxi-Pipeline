#!/usr/bin/env bash
set -eo pipefail

# ========================================================================
# PRIVATE REGISTRY CONFIGURATION
# ========================================================================
REGISTRY_URL="https://registry-route-ai4cloudops-11855c.apps.shift.nerc.mghpcc.org"
# REGISTRY_USER="${REGISTRY_USER:-}"
# REGISTRY_PASS="${REGISTRY_PASS:-}"
REGISTRY_USER="myuser"
REGISTRY_PASS="!QAZ2wsx"

if [ -z "$REGISTRY_USER" ] || [ -z "$REGISTRY_PASS" ]; then
    echo >&2 "error: Set REGISTRY_USER and REGISTRY_PASS environment variables"
    exit 1
fi
# ========================================================================

for cmd in curl jq base64; do
    if ! command -v "$cmd" &> /dev/null; then
        echo >&2 "error: \"$cmd\" not found!"
        exit 1
    fi
done

usage() {
    echo "usage: $0 dir image[:tag][@digest] ..."
    echo "   e.g.: $0 /tmp/my-images hello-world:latest@sha256:8be990ef2aeb16dbcb9271ddfe2610fa6658d13f6dfb8cc..."
    echo "   or, if digest is omitted: $0 /tmp/my-images hello-world:latest"
    [ -z "$1" ] || exit "$1"
}

# ------------------------------------------------------------------------
# Helper function for Basic Auth (for local registry)
basic_auth() {
    echo -n "${REGISTRY_USER}:${REGISTRY_PASS}" | base64 -w 0
}

# ------------------------------------------------------------------------
# Download a blob from the registry using Basic auth.
fetch_blob() {
    local image="$1" digest="$2" targetFile="$3"
    echo "Fetching blob: ${image}@${digest}"
    curl -fSL \
         -H "Authorization: Basic $(basic_auth)" \
         "${REGISTRY_URL}/v2/${image}/blobs/${digest}" \
         -o "${targetFile}" \
         --progress-bar
}

# ------------------------------------------------------------------------
# handle_single_manifest_v2:
# Process a manifest (schemaVersion 2) and download config and layers.
handle_single_manifest_v2() {
    local manifestJson="$1"
    shift

    local configDigest imageId configFile
    configDigest=$(echo "$manifestJson" | jq -r '.config.digest')
    imageId="${configDigest#*:}"  # strip "sha256:" prefix
    configFile="${imageId}.json"

    # Download the image configuration
    fetch_blob "${image}" "${configDigest}" "${dir}/${configFile}"

    # Read the layers array
    local layers=()
    while IFS= read -r layer; do
        layers+=("$layer")
    done < <(echo "$manifestJson" | jq -r -c '.layers[]')

    echo "Downloading '$imageIdentifier' (${#layers[@]} layers)..."
    local layerId="" layerFiles=()

    for i in "${!layers[@]}"; do
        local layerMeta
        layerMeta="${layers[$i]}"
        local layerMediaType layerDigest
        layerMediaType=$(echo "$layerMeta" | jq -r '.mediaType')
        layerDigest=$(echo "$layerMeta" | jq -r '.digest')

        local parentId="$layerId"
        # Compute a fake layer ID by concatenating previous layer id and current layer digest, then hashing.
        layerId=$(echo -e "${parentId}\n${layerDigest}" | sha256sum | cut -d' ' -f1)

        mkdir -p "${dir}/${layerId}"
        echo '1.0' > "${dir}/${layerId}/VERSION"

        # Generate JSON metadata for the layer using a base structure.
        if [ ! -s "${dir}/${layerId}/json" ]; then
            local parentJson=""
            if [ -n "$parentId" ]; then
                parentJson=$(printf ', "parent": "%s"' "$parentId")
            fi
            local addJson
            addJson=$(printf '{ "id": "%s"%s }' "$layerId" "$parentJson")
            jq "$addJson + ." > "${dir}/${layerId}/json" <<-'EOJSON'
				{
					"created": "0001-01-01T00:00:00Z",
					"container_config": {
						"Hostname": "",
						"Domainname": "",
						"User": "",
						"AttachStdin": false,
						"AttachStdout": false,
						"AttachStderr": false,
						"Tty": false,
						"OpenStdin": false,
						"StdinOnce": false,
						"Env": null,
						"Cmd": null,
						"Image": "",
						"Volumes": null,
						"WorkingDir": "",
						"Entrypoint": null,
						"OnBuild": null,
						"Labels": null
					}
				}
EOJSON
        fi

        case "$layerMediaType" in
            application/vnd.docker.image.rootfs.diff.tar.gzip|application/vnd.oci.image.layer.v1.tar+gzip)
                local layerTar="${layerId}/layer.tar"
                layerFiles+=("$layerTar")
                if [ -f "${dir}/${layerTar}" ]; then
                    echo "skipping existing ${layerId:0:12}"
                    continue
                fi
                fetch_blob "${image}" "${layerDigest}" "${dir}/${layerTar}"
                ;;
            *)
                echo >&2 "error: unknown layer mediaType ($imageIdentifier, $layerDigest): '$layerMediaType'"
                exit 1
                ;;
        esac
    done

    # Adjust the top layer JSON by merging config and top layer JSON (for backward compatibility).
    jq -s '.[0] + {id: .[1].id, parent: (.[1].parent // "")}' \
        "${dir}/${configFile}" \
        <(jq '{id, parent}' "${dir}/${layerId}/json") \
        > "${dir}/${layerId}/json.tmp"
    mv "${dir}/${layerId}/json.tmp" "${dir}/${layerId}/json"

    # Build the manifest entry for this image.
    local manifestJsonEntry
    manifestJsonEntry=$(jq -n \
        --arg cfg "$configFile" \
        --arg id "$imageIdentifier" \
        --argjson lyr "$(printf '%s\n' "${layerFiles[@]}" | jq -R . | jq -s .)" \
        '{Config: $cfg, RepoTags: [$id], Layers: $lyr}')
    manifestJsonEntries+=("$manifestJsonEntry")
}

# ------------------------------------------------------------------------
# Main function: process arguments and generate image output.
main() {
    if [ $# -lt 2 ]; then
        usage 1
    fi

    dir="$1"
    shift
    mkdir -p "$dir"

    while [ $# -gt 0 ]; do
        imageTag="$1"
        shift
        image="${imageTag%%[:@]*}"          # Extract image name
        imageTagRest="${imageTag#*:}"         # Everything after first colon
        tag="${imageTagRest%%@*}"             # Tag (before '@')
        digest="${imageTagRest##*@}"          # Digest (after '@')
        # If no '@' is present, look up the digest.
        if [[ "$imageTag" != *"@"* ]]; then
            digest=$(curl -fsSLI \
                -H "Authorization: Basic $(basic_auth)" \
                -H "Accept: application/vnd.docker.distribution.manifest.v2+json" \
                "${REGISTRY_URL}/v2/${image}/manifests/${tag}" \
                | grep -i 'Docker-Content-Digest:' | awk '{print $2}' | tr -d '\r\n')
            if [ -z "$digest" ]; then
                echo >&2 "error: Unable to retrieve digest for ${image}:${tag}"
                exit 1
            fi
        fi

        imageIdentifier="${image}:${tag}@${digest}"
        echo "Processing ${imageIdentifier}"

        manifestJson=$(curl -fsSL \
            -H "Authorization: Basic $(basic_auth)" \
            -H "Accept: application/vnd.docker.distribution.manifest.v2+json" \
            -H "Accept: application/vnd.oci.image.manifest.v1+json" \
            "${REGISTRY_URL}/v2/${image}/manifests/${digest}")

        case $(echo "$manifestJson" | jq -r '.schemaVersion') in
            2)
                handle_single_manifest_v2 "$manifestJson"
                ;;
            1)
                echo >&2 "error: V1 manifests unsupported"
                exit 1
                ;;
            *)
                echo >&2 "error: Unknown manifest schemaVersion for ${imageIdentifier}"
                exit 1
                ;;
        esac

        imageFile="${image//\//_}"
        echo "\"${tag}\": \"${imageId}\"" >> "${dir}/tags-${imageFile}.tmp"
        images+=("$image")
    done

    # Generate the repositories file.
    echo '{' > "${dir}/repositories"
    first=true
    for tmpFile in "${dir}"/tags-*.tmp; do
        [ -e "$tmpFile" ] || continue
        imageName="${tmpFile##*/tags-}"
        imageName="${imageName%.tmp}"
        imageName="${imageName//_//}"
        if [ "$first" = true ]; then
            first=false
        else
            echo ',' >> "${dir}/repositories"
        fi
        echo -n "    \"${imageName}\": { $(<"$tmpFile") }" >> "${dir}/repositories"
        rm "$tmpFile"
    done
    echo -e "\n}" >> "${dir}/repositories"

    # Generate manifest.json by collecting all manifest entries into a JSON array.
    if [ "${#manifestJsonEntries[@]}" -gt 0 ]; then
        printf '%s\n' "${manifestJsonEntries[@]}" | jq -s '.' > "${dir}/manifest.json"
    else
        rm -f "${dir}/manifest.json"
    fi

    echo "Download complete. Load with: tar -cC '${dir}' . | docker load"
}

# Invoke the main function with all script arguments.
main "$@"
