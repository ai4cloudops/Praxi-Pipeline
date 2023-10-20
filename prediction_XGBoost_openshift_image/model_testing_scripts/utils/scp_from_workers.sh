set -e
# backup data

hosts=("129.114.108.92" "129.114.108.179" "129.114.108.213" "129.114.109.147")
for j in ${!hosts[@]};
do
    echo $d "scp -i /home/cc/expr-cred/praxi-chi.pem -r cc@${hosts[$j]}:/home/cc/Praxi-Pipeline/prediction_XGBoost_openshift_image/model_testing_scripts/data0_results /home/cc/Praxi-study/Praxi-Pipeline/prediction_XGBoost_openshift_image/model_testing_scripts/data0_results/"
    echo "====================================="
    scp -i "/home/cc/expr-cred/praxi-chi.pem" -r "cc@${hosts[$j]}:/home/cc/Praxi-Pipeline/prediction_XGBoost_openshift_image/model_testing_scripts/data0_results/" "/home/cc/Praxi-study/Praxi-Pipeline/prediction_XGBoost_openshift_image/model_testing_scripts/data0_results/"
done