set -e
# backup data

dirs=(cwd_ML_with_data_0*2shuffleidx*)
for j in ${!dirs[@]};
do
    echo $d "mv '${dirs[$j]}' '/home/cc/Praxi-study/Praxi-Pipeline/prediction_XGBoost_openshift_image/model_testing_scripts/data0_results/'"
    # ls "${dirs[$j]}"
    echo "====================================="
    mv "${dirs[$j]}" "/home/cc/Praxi-study/Praxi-Pipeline/prediction_XGBoost_openshift_image/model_testing_scripts/data0_results/"
done
# zip -r "Praxi-study.zip" "Praxi-study/"

# gdrive files upload './m4_nop_reconstruction_client_5_equal_work_dataset_base_500_[0, 7]_cifar10_equDiff.zip' --parent 1cSgYLRJsrZlviG_JaelrzjxkOr6YQIpA
