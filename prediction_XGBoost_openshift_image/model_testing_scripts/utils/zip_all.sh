set -e
# backup data

dirs=(cwd_ML_with_data_0*)
for j in ${!dirs[@]};
do
    echo $d "gdrive files upload '${dirs[$j]}'"
    # ls "${dirs[$j]}"
    echo "====================================="
    zip -r "./data0_results/${dirs[$j]}.zip" "${dirs[$j]}" >/dev/null
    # gdrive files upload "${dirs[$j]}.zip" --parent 1cSgYLRJsrZlviG_JaelrzjxkOr6YQIpA
    # rm "${dirs[$j]}-nerc-prod.zip"
done
# zip -r "Praxi-study.zip" "Praxi-study/"

# gdrive files upload './m4_nop_reconstruction_client_5_equal_work_dataset_base_500_[0, 7]_cifar10_equDiff.zip' --parent 1cSgYLRJsrZlviG_JaelrzjxkOr6YQIpA
