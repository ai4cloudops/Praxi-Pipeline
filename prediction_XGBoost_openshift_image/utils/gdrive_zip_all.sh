set -e
# backup data

dirs=(archived_data_3_10-50_trees_False)
for j in ${!dirs[@]};
do
    echo $d "gdrive files upload '${dirs[$j]}'"
    # ls "${dirs[$j]}"
    echo "====================================="
    zip -r "${dirs[$j]}-praxibag.zip" "${dirs[$j]}"
    gdrive files upload "${dirs[$j]}-praxibag.zip" --parent 1cSgYLRJsrZlviG_JaelrzjxkOr6YQIpA
    rm "${dirs[$j]}-praxibag.zip"
done
# zip -r "Praxi-study.zip" "Praxi-study/"

# gdrive files upload './m4_nop_reconstruction_client_5_equal_work_dataset_base_500_[0, 7]_cifar10_equDiff.zip' --parent 1cSgYLRJsrZlviG_JaelrzjxkOr6YQIpA
