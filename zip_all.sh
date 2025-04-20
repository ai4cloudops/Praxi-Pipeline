#!/usr/bin/env bash
set -euo pipefail

LOGFILE="upload.log"
CRED_ID="41853ef9524247859b846cd4c2dabbcd"
CRED_SECRET="OM12GKfB1gmldqPxoTtImuyB-Dg2LqzE8XXzTpvFTnOL2PTdKk2BfKFsuwUtRM5RCB7Pz2EWpt0Q5YvZARXNSA"
CONTAINER="2025-04-20-backup"

# Hard‑coded list of directories to process
declare -a DIRS=("DeathStarBench" "LIBRA" "ml-model-deployment" "NN-Inference-And-EE-With_FaaS" "Praxi-Pipeline" "praxium" "Prodigy" "pyyaml-anomalous")

for dname in "${DIRS[@]}"; do
  if [[ ! -d "$dname" ]]; then
    echo "$(date '+%F %T') SKIP: '$dname' is not a directory" >>"$LOGFILE"
    continue
  fi

  zipfile="${dname}.zip"

  {
    echo "===== $(date '+%F %T') – START: $dname ====="
    echo "Zipping ${dname} → ${zipfile}…"
    zip -r "$zipfile" "$dname"

    echo "Uploading ${zipfile} to Swift container '${CONTAINER}'…"
    swift --os-auth-type v3applicationcredential \
          --os-application-credential-id "$CRED_ID" \
          --os-application-credential-secret "$CRED_SECRET" \
          upload --changed --segment-size 4831838208 \
          "$CONTAINER" "$zipfile"

    echo "Removing local archive ${zipfile}…"
    rm -f "$zipfile"
    echo "===== $(date '+%F %T') – DONE: $dname ====="
    echo
  } >>"$LOGFILE" 2>&1
done
