from collections import defaultdict
import os
from pathlib import Path
import matplotlib.pyplot as plt
import yaml



input_data_pathnames_l = [
    # "/home/cc/Praxi-study/Praxi-Pipeline/prediction_openshift_image/model_testing_scripts/incremental_batchbybatch/cwd_1000_verpak_0_csoaa3000_5timesdata_datareplay0_batchbybatch2_SL_conf/results/pred_output-iterative.txt",
    # "/home/cc/Praxi-study/Praxi-Pipeline/prediction_openshift_image/model_testing_scripts/incremental_batchbybatch/cwd_1000_verpak_0_csoaa3000_5timesdata_datareplay0_batchbybatch4_SL_conf/results/pred_output-iterative.txt",
    # "/home/cc/Praxi-study/Praxi-Pipeline/prediction_openshift_image/model_testing_scripts/incremental_batchbybatch/cwd_1000_verpak_0_csoaa3000_5timesdata_datareplay0_batchbybatch6_SL_conf/results/pred_output-iterative.txt",
    # "/home/cc/Praxi-study/Praxi-Pipeline/prediction_openshift_image/model_testing_scripts/incremental_batchbybatch/cwd_1000_verpak_0_csoaa3000_5timesdata_datareplay0_batchbybatch8_SL_conf/results/pred_output-iterative.txt",
    # "/home/cc/Praxi-study/Praxi-Pipeline/prediction_openshift_image/model_testing_scripts/incremental_batchbybatch/cwd_1000_verpak_0_csoaa3000_5timesdata_datareplay1_batchbybatch2_SL_conf/results/pred_output-iterative.txt",
    # "/home/cc/Praxi-study/Praxi-Pipeline/prediction_openshift_image/model_testing_scripts/incremental_batchbybatch/cwd_1000_verpak_0_csoaa3000_5timesdata_datareplay1_batchbybatch4_SL_conf/results/pred_output-iterative.txt",
    # "/home/cc/Praxi-study/Praxi-Pipeline/prediction_openshift_image/model_testing_scripts/incremental_batchbybatch/cwd_1000_verpak_0_csoaa3000_5timesdata_datareplay1_batchbybatch6_SL_conf/results/pred_output-iterative.txt",
    # "/home/cc/Praxi-study/Praxi-Pipeline/prediction_openshift_image/model_testing_scripts/incremental_batchbybatch/cwd_1000_verpak_0_csoaa3000_5timesdata_datareplay1_batchbybatch8_SL_conf/results/pred_output-iterative.txt",
    # "/home/cc/Praxi-study/Praxi-Pipeline/prediction_openshift_image/model_testing_scripts/incremental_batchbybatch/cwd_1000_verpak_0_csoaa3000_5timesdata_datareplay3_batchbybatch2_SL_conf/results/pred_output-iterative.txt",
    # "/home/cc/Praxi-study/Praxi-Pipeline/prediction_openshift_image/model_testing_scripts/incremental_batchbybatch/cwd_1000_verpak_0_csoaa3000_5timesdata_datareplay3_batchbybatch4_SL_conf/results/pred_output-iterative.txt",
    # "/home/cc/Praxi-study/Praxi-Pipeline/prediction_openshift_image/model_testing_scripts/incremental_batchbybatch/cwd_1000_verpak_0_csoaa3000_5timesdata_datareplay3_batchbybatch6_SL_conf/results/pred_output-iterative.txt",
    # "/home/cc/Praxi-study/Praxi-Pipeline/prediction_openshift_image/model_testing_scripts/incremental_batchbybatch/cwd_1000_verpak_0_csoaa3000_5timesdata_datareplay3_batchbybatch8_SL_conf/results/pred_output-iterative.txt",
    # "/home/cc/Praxi-study/Praxi-Pipeline/prediction_openshift_image/model_testing_scripts/incremental_batchbybatch/cwd_500_verpak_0_csoaa3000_5timesdata_datareplay0_batchbybatch1_SL_conf/results/pred_output-iterative.txt",
    # "/home/cc/Praxi-study/Praxi-Pipeline/prediction_openshift_image/model_testing_scripts/incremental_batchbybatch/cwd_500_verpak_0_csoaa3000_5timesdata_datareplay0_batchbybatch2_SL_conf/results/pred_output-iterative.txt",
    # "/home/cc/Praxi-study/Praxi-Pipeline/prediction_openshift_image/model_testing_scripts/incremental_batchbybatch/cwd_500_verpak_0_csoaa3000_5timesdata_datareplay0_batchbybatch3_SL_conf/results/pred_output-iterative.txt",
    # "/home/cc/Praxi-study/Praxi-Pipeline/prediction_openshift_image/model_testing_scripts/incremental_batchbybatch/cwd_500_verpak_0_csoaa3000_5timesdata_datareplay0_batchbybatch4_SL_conf/results/pred_output-iterative.txt",
    # "/home/cc/Praxi-study/Praxi-Pipeline/prediction_openshift_image/model_testing_scripts/incremental_batchbybatch/cwd_500_verpak_0_csoaa3000_5timesdata_datareplay3_batchbybatch1_SL_conf/results/pred_output-iterative.txt",
    # "/home/cc/Praxi-study/Praxi-Pipeline/prediction_openshift_image/model_testing_scripts/incremental_batchbybatch/cwd_500_verpak_0_csoaa3000_5timesdata_datareplay3_batchbybatch2_SL_conf/results/pred_output-iterative.txt",
    # "/home/cc/Praxi-study/Praxi-Pipeline/prediction_openshift_image/model_testing_scripts/incremental_batchbybatch/cwd_500_verpak_0_csoaa3000_5timesdata_datareplay3_batchbybatch3_SL_conf/results/pred_output-iterative.txt",
    # "/home/cc/Praxi-study/Praxi-Pipeline/prediction_openshift_image/model_testing_scripts/incremental_batchbybatch/cwd_500_verpak_0_csoaa3000_5timesdata_datareplay3_batchbybatch4_SL_conf/results/pred_output-iterative.txt",

    # "/home/cc/Praxi-study/Praxi-Pipeline/prediction_openshift_image/model_testing_scripts/incremental_batchbybatch/cwd_125_verpak_0_csoaa3000_5timesdata_datareplay0_batchbybatch1_SL_conf/results/pred_output-iterative.txt",
    # "/home/cc/Praxi-study/Praxi-Pipeline/prediction_openshift_image/model_testing_scripts/incremental_batchbybatch/cwd_125_verpak_0_csoaa3000_5timesdata_datareplay3_batchbybatch1_SL_conf/results/pred_output-iterative.txt"

    # "/home/cc/Praxi-study/Praxi-Pipeline/prediction_openshift_image/model_testing_scripts/incremental_batchbybatch/cwd_1_verpak_0_csoaa3000_5timesdata_datareplayFalse_batchbybatch5_SL/results/pred_output-iterative.txt"

    # "/home/cc/Praxi-study/Praxi-Pipeline/prediction_openshift_image/model_testing_scripts/incremental_batchbybatch/cwd_1000_verpak_0_csoaa3000_5timesdata_datareplay0_batchbybatch2_SL_conf_cost5/results/pred_output-iterative.txt",
    # "/home/cc/Praxi-study/Praxi-Pipeline/prediction_openshift_image/model_testing_scripts/incremental_batchbybatch/cwd_1000_verpak_0_csoaa3000_5timesdata_datareplay0_batchbybatch4_SL_conf_cost5/results/pred_output-iterative.txt",
    # "/home/cc/Praxi-study/Praxi-Pipeline/prediction_openshift_image/model_testing_scripts/incremental_batchbybatch/cwd_1000_verpak_0_csoaa3000_5timesdata_datareplay0_batchbybatch6_SL_conf_cost5/results/pred_output-iterative.txt",
    # "/home/cc/Praxi-study/Praxi-Pipeline/prediction_openshift_image/model_testing_scripts/incremental_batchbybatch/cwd_1000_verpak_0_csoaa3000_5timesdata_datareplay0_batchbybatch8_SL_conf_cost5/results/pred_output-iterative.txt",
    # "/home/cc/Praxi-study/Praxi-Pipeline/prediction_openshift_image/model_testing_scripts/incremental_batchbybatch/cwd_1000_verpak_0_csoaa3000_5timesdata_datareplay1_batchbybatch2_SL_conf_cost5/results/pred_output-iterative.txt",
    # "/home/cc/Praxi-study/Praxi-Pipeline/prediction_openshift_image/model_testing_scripts/incremental_batchbybatch/cwd_1000_verpak_0_csoaa3000_5timesdata_datareplay1_batchbybatch4_SL_conf_cost5/results/pred_output-iterative.txt",
    # "/home/cc/Praxi-study/Praxi-Pipeline/prediction_openshift_image/model_testing_scripts/incremental_batchbybatch/cwd_1000_verpak_0_csoaa3000_5timesdata_datareplay1_batchbybatch6_SL_conf_cost5/results/pred_output-iterative.txt",
    # "/home/cc/Praxi-study/Praxi-Pipeline/prediction_openshift_image/model_testing_scripts/incremental_batchbybatch/cwd_1000_verpak_0_csoaa3000_5timesdata_datareplay1_batchbybatch8_SL_conf_cost5/results/pred_output-iterative.txt",
    # "/home/cc/Praxi-study/Praxi-Pipeline/prediction_openshift_image/model_testing_scripts/incremental_batchbybatch/cwd_1000_verpak_0_csoaa3000_5timesdata_fullbatchdatareplay1_batchbybatch2_SL_conf_cost5/results/pred_output-iterative.txt",
    # "/home/cc/Praxi-study/Praxi-Pipeline/prediction_openshift_image/model_testing_scripts/incremental_batchbybatch/cwd_1000_verpak_0_csoaa3000_5timesdata_fullbatchdatareplay1_batchbybatch4_SL_conf_cost5/results/pred_output-iterative.txt",
    # "/home/cc/Praxi-study/Praxi-Pipeline/prediction_openshift_image/model_testing_scripts/incremental_batchbybatch/cwd_1000_verpak_0_csoaa3000_5timesdata_fullbatchdatareplay1_batchbybatch6_SL_conf_cost5/results/pred_output-iterative.txt",
    # "/home/cc/Praxi-study/Praxi-Pipeline/prediction_openshift_image/model_testing_scripts/incremental_batchbybatch/cwd_1000_verpak_0_csoaa3000_5timesdata_fullbatchdatareplay1_batchbybatch8_SL_conf_cost5/results/pred_output-iterative.txt",


    # "/home/cc/Praxi-study/Praxi-Pipeline/prediction_openshift_image/model_testing_scripts/incremental_batchbybatch/cwd_1000_verpak_1_csoaa3000_5timesdata_datareplay0_batchbybatch2_SL_conf/results/pred_output-iterative.txt",
    # "/home/cc/Praxi-study/Praxi-Pipeline/prediction_openshift_image/model_testing_scripts/incremental_batchbybatch/cwd_1000_verpak_1_csoaa3000_5timesdata_datareplay0_batchbybatch4_SL_conf/results/pred_output-iterative.txt",
    # "/home/cc/Praxi-study/Praxi-Pipeline/prediction_openshift_image/model_testing_scripts/incremental_batchbybatch/cwd_1000_verpak_1_csoaa3000_5timesdata_datareplay0_batchbybatch6_SL_conf/results/pred_output-iterative.txt",
    # "/home/cc/Praxi-study/Praxi-Pipeline/prediction_openshift_image/model_testing_scripts/incremental_batchbybatch/cwd_1000_verpak_1_csoaa3000_5timesdata_datareplay0_batchbybatch8_SL_conf/results/pred_output-iterative.txt",

    "/home/cc/Praxi-study/Praxi-Pipeline/prediction_openshift_image/model_testing_scripts/incremental_batchbybatch/cwd_1000_verpak_0_csoaa3000_7timesdata_fullbatchdatareplay10_batchbybatch6_SL_conf_cost5/results/pred_output-iterative.txt",

]
input_data_true_labels_pathnames_l = [
    # "/home/cc/Praxi-study/Praxi-Pipeline/prediction_openshift_image/model_testing_scripts/incremental_batchbybatch/cwd_1000_verpak_0_csoaa3000_5timesdata_datareplay0_batchbybatch2_SL_conf/results/pred_true.txt",
    # "/home/cc/Praxi-study/Praxi-Pipeline/prediction_openshift_image/model_testing_scripts/incremental_batchbybatch/cwd_1000_verpak_0_csoaa3000_5timesdata_datareplay0_batchbybatch4_SL_conf/results/pred_true.txt",
    # "/home/cc/Praxi-study/Praxi-Pipeline/prediction_openshift_image/model_testing_scripts/incremental_batchbybatch/cwd_1000_verpak_0_csoaa3000_5timesdata_datareplay0_batchbybatch6_SL_conf/results/pred_true.txt",
    # "/home/cc/Praxi-study/Praxi-Pipeline/prediction_openshift_image/model_testing_scripts/incremental_batchbybatch/cwd_1000_verpak_0_csoaa3000_5timesdata_datareplay0_batchbybatch8_SL_conf/results/pred_true.txt",
    # "/home/cc/Praxi-study/Praxi-Pipeline/prediction_openshift_image/model_testing_scripts/incremental_batchbybatch/cwd_1000_verpak_0_csoaa3000_5timesdata_datareplay1_batchbybatch2_SL_conf/results/pred_true.txt",
    # "/home/cc/Praxi-study/Praxi-Pipeline/prediction_openshift_image/model_testing_scripts/incremental_batchbybatch/cwd_1000_verpak_0_csoaa3000_5timesdata_datareplay1_batchbybatch4_SL_conf/results/pred_true.txt",
    # "/home/cc/Praxi-study/Praxi-Pipeline/prediction_openshift_image/model_testing_scripts/incremental_batchbybatch/cwd_1000_verpak_0_csoaa3000_5timesdata_datareplay1_batchbybatch6_SL_conf/results/pred_true.txt",
    # "/home/cc/Praxi-study/Praxi-Pipeline/prediction_openshift_image/model_testing_scripts/incremental_batchbybatch/cwd_1000_verpak_0_csoaa3000_5timesdata_datareplay1_batchbybatch8_SL_conf/results/pred_true.txt",
    # "/home/cc/Praxi-study/Praxi-Pipeline/prediction_openshift_image/model_testing_scripts/incremental_batchbybatch/cwd_1000_verpak_0_csoaa3000_5timesdata_datareplay3_batchbybatch2_SL_conf/results/pred_true.txt",
    # "/home/cc/Praxi-study/Praxi-Pipeline/prediction_openshift_image/model_testing_scripts/incremental_batchbybatch/cwd_1000_verpak_0_csoaa3000_5timesdata_datareplay3_batchbybatch4_SL_conf/results/pred_true.txt",
    # "/home/cc/Praxi-study/Praxi-Pipeline/prediction_openshift_image/model_testing_scripts/incremental_batchbybatch/cwd_1000_verpak_0_csoaa3000_5timesdata_datareplay3_batchbybatch6_SL_conf/results/pred_true.txt",
    # "/home/cc/Praxi-study/Praxi-Pipeline/prediction_openshift_image/model_testing_scripts/incremental_batchbybatch/cwd_1000_verpak_0_csoaa3000_5timesdata_datareplay3_batchbybatch8_SL_conf/results/pred_true.txt",
    # "/home/cc/Praxi-study/Praxi-Pipeline/prediction_openshift_image/model_testing_scripts/incremental_batchbybatch/cwd_500_verpak_0_csoaa3000_5timesdata_datareplay0_batchbybatch1_SL_conf/results/pred_true.txt",
    # "/home/cc/Praxi-study/Praxi-Pipeline/prediction_openshift_image/model_testing_scripts/incremental_batchbybatch/cwd_500_verpak_0_csoaa3000_5timesdata_datareplay0_batchbybatch2_SL_conf/results/pred_true.txt",
    # "/home/cc/Praxi-study/Praxi-Pipeline/prediction_openshift_image/model_testing_scripts/incremental_batchbybatch/cwd_500_verpak_0_csoaa3000_5timesdata_datareplay0_batchbybatch3_SL_conf/results/pred_true.txt",
    # "/home/cc/Praxi-study/Praxi-Pipeline/prediction_openshift_image/model_testing_scripts/incremental_batchbybatch/cwd_500_verpak_0_csoaa3000_5timesdata_datareplay0_batchbybatch4_SL_conf/results/pred_true.txt",
    # "/home/cc/Praxi-study/Praxi-Pipeline/prediction_openshift_image/model_testing_scripts/incremental_batchbybatch/cwd_500_verpak_0_csoaa3000_5timesdata_datareplay3_batchbybatch1_SL_conf/results/pred_true.txt",
    # "/home/cc/Praxi-study/Praxi-Pipeline/prediction_openshift_image/model_testing_scripts/incremental_batchbybatch/cwd_500_verpak_0_csoaa3000_5timesdata_datareplay3_batchbybatch2_SL_conf/results/pred_true.txt",
    # "/home/cc/Praxi-study/Praxi-Pipeline/prediction_openshift_image/model_testing_scripts/incremental_batchbybatch/cwd_500_verpak_0_csoaa3000_5timesdata_datareplay3_batchbybatch3_SL_conf/results/pred_true.txt",
    # "/home/cc/Praxi-study/Praxi-Pipeline/prediction_openshift_image/model_testing_scripts/incremental_batchbybatch/cwd_500_verpak_0_csoaa3000_5timesdata_datareplay3_batchbybatch4_SL_conf/results/pred_true.txt",

    # "/home/cc/Praxi-study/Praxi-Pipeline/prediction_openshift_image/model_testing_scripts/incremental_batchbybatch/cwd_125_verpak_0_csoaa3000_5timesdata_datareplay0_batchbybatch1_SL_conf/results/pred_true.txt",
    # "/home/cc/Praxi-study/Praxi-Pipeline/prediction_openshift_image/model_testing_scripts/incremental_batchbybatch/cwd_125_verpak_0_csoaa3000_5timesdata_datareplay3_batchbybatch1_SL_conf/results/pred_true.txt"

    # "/home/cc/Praxi-study/Praxi-Pipeline/prediction_openshift_image/model_testing_scripts/incremental_batchbybatch/cwd_1_verpak_0_csoaa3000_5timesdata_datareplayFalse_batchbybatch5_SL/results/pred_true.txt",

    # "/home/cc/Praxi-study/Praxi-Pipeline/prediction_openshift_image/model_testing_scripts/incremental_batchbybatch/cwd_1000_verpak_0_csoaa3000_5timesdata_datareplay0_batchbybatch2_SL_conf_cost5/results/pred_true.txt",
    # "/home/cc/Praxi-study/Praxi-Pipeline/prediction_openshift_image/model_testing_scripts/incremental_batchbybatch/cwd_1000_verpak_0_csoaa3000_5timesdata_datareplay0_batchbybatch4_SL_conf_cost5/results/pred_true.txt",
    # "/home/cc/Praxi-study/Praxi-Pipeline/prediction_openshift_image/model_testing_scripts/incremental_batchbybatch/cwd_1000_verpak_0_csoaa3000_5timesdata_datareplay0_batchbybatch6_SL_conf_cost5/results/pred_true.txt",
    # "/home/cc/Praxi-study/Praxi-Pipeline/prediction_openshift_image/model_testing_scripts/incremental_batchbybatch/cwd_1000_verpak_0_csoaa3000_5timesdata_datareplay0_batchbybatch8_SL_conf_cost5/results/pred_true.txt",
    # "/home/cc/Praxi-study/Praxi-Pipeline/prediction_openshift_image/model_testing_scripts/incremental_batchbybatch/cwd_1000_verpak_0_csoaa3000_5timesdata_datareplay1_batchbybatch2_SL_conf_cost5/results/pred_true.txt",
    # "/home/cc/Praxi-study/Praxi-Pipeline/prediction_openshift_image/model_testing_scripts/incremental_batchbybatch/cwd_1000_verpak_0_csoaa3000_5timesdata_datareplay1_batchbybatch4_SL_conf_cost5/results/pred_true.txt",
    # "/home/cc/Praxi-study/Praxi-Pipeline/prediction_openshift_image/model_testing_scripts/incremental_batchbybatch/cwd_1000_verpak_0_csoaa3000_5timesdata_datareplay1_batchbybatch6_SL_conf_cost5/results/pred_true.txt",
    # "/home/cc/Praxi-study/Praxi-Pipeline/prediction_openshift_image/model_testing_scripts/incremental_batchbybatch/cwd_1000_verpak_0_csoaa3000_5timesdata_datareplay1_batchbybatch8_SL_conf_cost5/results/pred_true.txt",
    # "/home/cc/Praxi-study/Praxi-Pipeline/prediction_openshift_image/model_testing_scripts/incremental_batchbybatch/cwd_1000_verpak_0_csoaa3000_5timesdata_fullbatchdatareplay1_batchbybatch2_SL_conf_cost5/results/pred_true.txt",
    # "/home/cc/Praxi-study/Praxi-Pipeline/prediction_openshift_image/model_testing_scripts/incremental_batchbybatch/cwd_1000_verpak_0_csoaa3000_5timesdata_fullbatchdatareplay1_batchbybatch4_SL_conf_cost5/results/pred_true.txt",
    # "/home/cc/Praxi-study/Praxi-Pipeline/prediction_openshift_image/model_testing_scripts/incremental_batchbybatch/cwd_1000_verpak_0_csoaa3000_5timesdata_fullbatchdatareplay1_batchbybatch6_SL_conf_cost5/results/pred_true.txt",
    # "/home/cc/Praxi-study/Praxi-Pipeline/prediction_openshift_image/model_testing_scripts/incremental_batchbybatch/cwd_1000_verpak_0_csoaa3000_5timesdata_fullbatchdatareplay1_batchbybatch8_SL_conf_cost5/results/pred_true.txt",




    # "/home/cc/Praxi-study/Praxi-Pipeline/prediction_openshift_image/model_testing_scripts/incremental_batchbybatch/cwd_1000_verpak_1_csoaa3000_5timesdata_datareplay0_batchbybatch2_SL_conf/results/pred_true.txt",
    # "/home/cc/Praxi-study/Praxi-Pipeline/prediction_openshift_image/model_testing_scripts/incremental_batchbybatch/cwd_1000_verpak_1_csoaa3000_5timesdata_datareplay0_batchbybatch4_SL_conf/results/pred_true.txt",
    # "/home/cc/Praxi-study/Praxi-Pipeline/prediction_openshift_image/model_testing_scripts/incremental_batchbybatch/cwd_1000_verpak_1_csoaa3000_5timesdata_datareplay0_batchbybatch6_SL_conf/results/pred_true.txt",
    # "/home/cc/Praxi-study/Praxi-Pipeline/prediction_openshift_image/model_testing_scripts/incremental_batchbybatch/cwd_1000_verpak_1_csoaa3000_5timesdata_datareplay0_batchbybatch8_SL_conf/results/pred_true.txt",


    "/home/cc/Praxi-study/Praxi-Pipeline/prediction_openshift_image/model_testing_scripts/incremental_batchbybatch/cwd_1000_verpak_0_csoaa3000_7timesdata_fullbatchdatareplay10_batchbybatch6_SL_conf_cost5/results/pred_true.txt",
]


fig_path = '/home/cc/Praxi-study/Praxi-Pipeline/prediction_openshift_image/figs/'
os.makedirs(fig_path, exist_ok=True)

for input_data_pathname, input_data_true_labels_pathname in zip(input_data_pathnames_l, input_data_true_labels_pathnames_l):
    if not Path(input_data_pathname).exists():
        print(f"Missing: {input_data_pathname}")
        continue
    output_filename = input_data_pathname.split("/")[-3]

    # Load true labels
    with open(input_data_true_labels_pathname, 'r') as file:
        true_labels = yaml.safe_load(file)  # This depends on the actual structure; adjust as necessary

    # Extract ids if the ids directly correlate with the line numbers
    true_line_ids = defaultdict(set)
    for line_idx, true_labels_per_line in enumerate(true_labels):
        for true_label in true_labels_per_line:
            true_line_ids[true_label].add(line_idx)

    for true_label in true_line_ids.keys():
        # Reduced dataset as provided
        with open(input_data_pathname, 'r') as file:
            # Parse the data
            label_costs = defaultdict(list)

            # Read each line and extract the data
            for line_idx, line in enumerate(file):
                if not line_idx in true_line_ids[true_label]:
                    continue
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
            bar = plt.bar(labels, max_values, color='lightblue', label='Max Value')
            plt. bar_label(bar, fmt='%.2f')
            bar = plt.bar(labels, min_values, color='salmon', label='Min Value')
            plt. bar_label(bar, fmt='%.2f')
            plt.yscale('log')
            plt.xlabel('Labels')
            plt.ylabel('Cost Values')
            plt.title('Range of Predicted Costs per Label')
            plt.legend()
            plt.xticks(labels)
            # plt.show()
            plt.savefig(fig_path+output_filename+f'_true_label_{true_label}.pdf', bbox_inches='tight')
            plt.close()
