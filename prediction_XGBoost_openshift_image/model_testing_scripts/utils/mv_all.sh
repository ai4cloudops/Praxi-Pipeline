set -e
# backup data

# mkdir /home/cc/Praxi-study/Praxi-Pipeline/prediction_XGBoost_openshift_image/model_testing_scripts/data0_results
dirs=(cwd_ML_with_data_0*_1njobs*)
for j in ${!dirs[@]};
do
    echo $d "mv '${dirs[$j]}' '/home/cc/Praxi-study/Praxi-Pipeline/prediction_XGBoost_openshift_image/model_testing_scripts/data0_results'"
    # ls "${dirs[$j]}"
    echo "====================================="
    cp -r "${dirs[$j]}" "/home/cc/Praxi-study/Praxi-Pipeline/prediction_XGBoost_openshift_image/model_testing_scripts/data0_results"
    rm -fr "${dirs[$j]}"
done
# zip -r "Praxi-study.zip" "Praxi-study/"

# gdrive files upload './m4_nop_reconstruction_client_5_equal_work_dataset_base_500_[0, 7]_cifar10_equDiff.zip' --parent 1cSgYLRJsrZlviG_JaelrzjxkOr6YQIpA
