from collections import defaultdict
from pathlib import Path
import matplotlib.pyplot as plt



input_data_pathnames_l = [
    "/home/cc/Praxi-study/Praxi-Pipeline/prediction_openshift_image/model_testing_scripts/incremental_batchbybatch/cwd_1000_verpak_0_csoaa3000_5timesdata_datareplay0_batchbybatch2_SL_conf/results/pred_output-iterative.txt",
    "/home/cc/Praxi-study/Praxi-Pipeline/prediction_openshift_image/model_testing_scripts/incremental_batchbybatch/cwd_1000_verpak_0_csoaa3000_5timesdata_datareplay0_batchbybatch4_SL_conf/results/pred_output-iterative.txt",
    "/home/cc/Praxi-study/Praxi-Pipeline/prediction_openshift_image/model_testing_scripts/incremental_batchbybatch/cwd_1000_verpak_0_csoaa3000_5timesdata_datareplay0_batchbybatch6_SL_conf/results/pred_output-iterative.txt",
    "/home/cc/Praxi-study/Praxi-Pipeline/prediction_openshift_image/model_testing_scripts/incremental_batchbybatch/cwd_1000_verpak_0_csoaa3000_5timesdata_datareplay0_batchbybatch8_SL_conf/results/pred_output-iterative.txt",
    "/home/cc/Praxi-study/Praxi-Pipeline/prediction_openshift_image/model_testing_scripts/incremental_batchbybatch/cwd_1000_verpak_0_csoaa3000_5timesdata_datareplay1_batchbybatch2_SL_conf/results/pred_output-iterative.txt",
    "/home/cc/Praxi-study/Praxi-Pipeline/prediction_openshift_image/model_testing_scripts/incremental_batchbybatch/cwd_1000_verpak_0_csoaa3000_5timesdata_datareplay1_batchbybatch4_SL_conf/results/pred_output-iterative.txt",
    "/home/cc/Praxi-study/Praxi-Pipeline/prediction_openshift_image/model_testing_scripts/incremental_batchbybatch/cwd_1000_verpak_0_csoaa3000_5timesdata_datareplay1_batchbybatch6_SL_conf/results/pred_output-iterative.txt",
    "/home/cc/Praxi-study/Praxi-Pipeline/prediction_openshift_image/model_testing_scripts/incremental_batchbybatch/cwd_1000_verpak_0_csoaa3000_5timesdata_datareplay1_batchbybatch8_SL_conf/results/pred_output-iterative.txt",
    "/home/cc/Praxi-study/Praxi-Pipeline/prediction_openshift_image/model_testing_scripts/incremental_batchbybatch/cwd_1000_verpak_0_csoaa3000_5timesdata_datareplay3_batchbybatch2_SL_conf/results/pred_output-iterative.txt",
    "/home/cc/Praxi-study/Praxi-Pipeline/prediction_openshift_image/model_testing_scripts/incremental_batchbybatch/cwd_1000_verpak_0_csoaa3000_5timesdata_datareplay3_batchbybatch4_SL_conf/results/pred_output-iterative.txt",
    "/home/cc/Praxi-study/Praxi-Pipeline/prediction_openshift_image/model_testing_scripts/incremental_batchbybatch/cwd_1000_verpak_0_csoaa3000_5timesdata_datareplay3_batchbybatch6_SL_conf/results/pred_output-iterative.txt",
    "/home/cc/Praxi-study/Praxi-Pipeline/prediction_openshift_image/model_testing_scripts/incremental_batchbybatch/cwd_1000_verpak_0_csoaa3000_5timesdata_datareplay3_batchbybatch8_SL_conf/results/pred_output-iterative.txt",
    "/home/cc/Praxi-study/Praxi-Pipeline/prediction_openshift_image/model_testing_scripts/incremental_batchbybatch/cwd_500_verpak_0_csoaa3000_5timesdata_datareplay0_batchbybatch1_SL_conf/results/pred_output-iterative.txt",
    "/home/cc/Praxi-study/Praxi-Pipeline/prediction_openshift_image/model_testing_scripts/incremental_batchbybatch/cwd_500_verpak_0_csoaa3000_5timesdata_datareplay0_batchbybatch2_SL_conf/results/pred_output-iterative.txt",
    "/home/cc/Praxi-study/Praxi-Pipeline/prediction_openshift_image/model_testing_scripts/incremental_batchbybatch/cwd_500_verpak_0_csoaa3000_5timesdata_datareplay0_batchbybatch3_SL_conf/results/pred_output-iterative.txt",
    "/home/cc/Praxi-study/Praxi-Pipeline/prediction_openshift_image/model_testing_scripts/incremental_batchbybatch/cwd_500_verpak_0_csoaa3000_5timesdata_datareplay0_batchbybatch4_SL_conf/results/pred_output-iterative.txt",
    "/home/cc/Praxi-study/Praxi-Pipeline/prediction_openshift_image/model_testing_scripts/incremental_batchbybatch/cwd_500_verpak_0_csoaa3000_5timesdata_datareplay3_batchbybatch1_SL_conf/results/pred_output-iterative.txt",
    "/home/cc/Praxi-study/Praxi-Pipeline/prediction_openshift_image/model_testing_scripts/incremental_batchbybatch/cwd_500_verpak_0_csoaa3000_5timesdata_datareplay3_batchbybatch2_SL_conf/results/pred_output-iterative.txt",
    "/home/cc/Praxi-study/Praxi-Pipeline/prediction_openshift_image/model_testing_scripts/incremental_batchbybatch/cwd_500_verpak_0_csoaa3000_5timesdata_datareplay3_batchbybatch3_SL_conf/results/pred_output-iterative.txt",
    "/home/cc/Praxi-study/Praxi-Pipeline/prediction_openshift_image/model_testing_scripts/incremental_batchbybatch/cwd_500_verpak_0_csoaa3000_5timesdata_datareplay3_batchbybatch4_SL_conf/results/pred_output-iterative.txt",

    "/home/cc/Praxi-study/Praxi-Pipeline/prediction_openshift_image/model_testing_scripts/incremental_batchbybatch/cwd_125_verpak_0_csoaa3000_5timesdata_datareplay0_batchbybatch1_SL_conf/results/pred_output-iterative.txt",
    "/home/cc/Praxi-study/Praxi-Pipeline/prediction_openshift_image/model_testing_scripts/incremental_batchbybatch/cwd_125_verpak_0_csoaa3000_5timesdata_datareplay3_batchbybatch1_SL_conf/results/pred_output-iterative.txt"
]

fig_path = '/home/cc/Praxi-study/Praxi-Pipeline/prediction_openshift_image/figs/'

for input_data_pathname in input_data_pathnames_l:
    if not Path(input_data_pathname).exists():
        print(f"Missing: {input_data_pathname}")
        continue
    filename = input_data_pathname.split("/")[-3]
    # Reduced dataset as provided
    with open(input_data_pathname, 'r') as file:
        # Parse the data
        label_costs = defaultdict(list)

        # Read each line and extract the data
        for line in file:
            items = line.strip().split()
            for item in items:
                label, cost = item.split(':')
                label = int(label)
                cost = float(cost)
                label_costs[label].append(cost)

        # Determine the ranges (min and max) for each label
        ranges = {label: (min(costs), max(costs)) for label, costs in label_costs.items()}

        # Prepare data for plotting
        labels = list(ranges.keys())
        min_values = [ranges[label][0] for label in labels]
        max_values = [ranges[label][1] for label in labels]

        # Create a plot
        plt.figure(figsize=(10, 5))
        plt.bar(labels, max_values, color='lightblue', label='Max Value')
        plt.bar(labels, min_values, color='salmon', label='Min Value')
        plt.yscale('log')
        plt.xlabel('Labels')
        plt.ylabel('Cost Values')
        plt.title('Range of Predicted Costs per Label')
        plt.legend()
        plt.xticks(labels)
        # plt.show()
        plt.savefig(fig_path+filename+'.pdf', bbox_inches='tight')
        plt.close()
