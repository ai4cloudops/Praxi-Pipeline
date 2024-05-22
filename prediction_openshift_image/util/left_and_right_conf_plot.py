from collections import defaultdict
import os
from pathlib import Path
import statistics
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
import numpy as np
import yaml



input_data_pathnames_l = [
    # "/home/cc/Praxi-study/Praxi-Pipeline/prediction_openshift_image/model_testing_scripts/incremental_batchbybatch/cwd_1000_verpak_0_csoaa3000_5timesdata_datareplay0_batchbybatch8_SL_conf/results/pred_output-iterative.txt",
    # "/home/cc/Praxi-study/Praxi-Pipeline/prediction_openshift_image/model_testing_scripts/incremental_batchbybatch/cwd_1000_verpak_1_csoaa3000_5timesdata_datareplay0_batchbybatch8_SL_conf/results/pred_output-iterative.txt",
    # "/home/cc/Praxi-study/Praxi-Pipeline/prediction_openshift_image/model_testing_scripts/incremental_batchbybatch/cwd_1000_verpak_2_csoaa3000_5timesdata_datareplay0_batchbybatch8_SL_conf/results/pred_output-iterative.txt",

    "/home/cc/Praxi-study/Praxi-Pipeline/prediction_openshift_image/model_testing_scripts/incremental_batchbybatch/cwd_1000_verpak_0_csoaa3000_5timesdata_datareplay1_batchbybatch8_SL_conf/results/pred_output-iterative.txt",
    "/home/cc/Praxi-study/Praxi-Pipeline/prediction_openshift_image/model_testing_scripts/incremental_batchbybatch/cwd_1000_verpak_1_csoaa3000_5timesdata_datareplay1_batchbybatch8_SL_conf/results/pred_output-iterative.txt",
    "/home/cc/Praxi-study/Praxi-Pipeline/prediction_openshift_image/model_testing_scripts/incremental_batchbybatch/cwd_1000_verpak_2_csoaa3000_5timesdata_datareplay1_batchbybatch8_SL_conf/results/pred_output-iterative.txt",

    # "/home/cc/Praxi-study/Praxi-Pipeline/prediction_openshift_image/model_testing_scripts/incremental_batchbybatch/cwd_1000_verpak_0_csoaa3000_5timesdata_datareplay0_batchbybatch8_SL_conf_cost5/results/pred_true.txt",
]
input_data_true_labels_pathnames_l = [
    # "/home/cc/Praxi-study/Praxi-Pipeline/prediction_openshift_image/model_testing_scripts/incremental_batchbybatch/cwd_1000_verpak_0_csoaa3000_5timesdata_datareplay0_batchbybatch8_SL_conf/results/pred_true.txt",
    # "/home/cc/Praxi-study/Praxi-Pipeline/prediction_openshift_image/model_testing_scripts/incremental_batchbybatch/cwd_1000_verpak_1_csoaa3000_5timesdata_datareplay0_batchbybatch8_SL_conf/results/pred_true.txt",
    # "/home/cc/Praxi-study/Praxi-Pipeline/prediction_openshift_image/model_testing_scripts/incremental_batchbybatch/cwd_1000_verpak_2_csoaa3000_5timesdata_datareplay0_batchbybatch8_SL_conf/results/pred_true.txt",

    "/home/cc/Praxi-study/Praxi-Pipeline/prediction_openshift_image/model_testing_scripts/incremental_batchbybatch/cwd_1000_verpak_0_csoaa3000_5timesdata_datareplay1_batchbybatch8_SL_conf/results/pred_true.txt",
    "/home/cc/Praxi-study/Praxi-Pipeline/prediction_openshift_image/model_testing_scripts/incremental_batchbybatch/cwd_1000_verpak_1_csoaa3000_5timesdata_datareplay1_batchbybatch8_SL_conf/results/pred_true.txt",
    "/home/cc/Praxi-study/Praxi-Pipeline/prediction_openshift_image/model_testing_scripts/incremental_batchbybatch/cwd_1000_verpak_2_csoaa3000_5timesdata_datareplay1_batchbybatch8_SL_conf/results/pred_true.txt",

    # "/home/cc/Praxi-study/Praxi-Pipeline/prediction_openshift_image/model_testing_scripts/incremental_batchbybatch/cwd_1000_verpak_0_csoaa3000_5timesdata_datareplay0_batchbybatch8_SL_conf_cost5/results/pred_true.txt",
]
label_table_pathnames_l = [
    # "/home/cc/Praxi-study/Praxi-Pipeline/prediction_openshift_image/model_testing_scripts/incremental_batchbybatch/cwd_1000_verpak_0_csoaa3000_5timesdata_datareplay0_batchbybatch8_SL_conf/results/label_table-iterative.yaml",
    # "/home/cc/Praxi-study/Praxi-Pipeline/prediction_openshift_image/model_testing_scripts/incremental_batchbybatch/cwd_1000_verpak_1_csoaa3000_5timesdata_datareplay0_batchbybatch8_SL_conf/results/label_table-iterative.yaml",
    # "/home/cc/Praxi-study/Praxi-Pipeline/prediction_openshift_image/model_testing_scripts/incremental_batchbybatch/cwd_1000_verpak_2_csoaa3000_5timesdata_datareplay0_batchbybatch8_SL_conf/results/label_table-iterative.yaml",
  
    "/home/cc/Praxi-study/Praxi-Pipeline/prediction_openshift_image/model_testing_scripts/incremental_batchbybatch/cwd_1000_verpak_0_csoaa3000_5timesdata_datareplay1_batchbybatch8_SL_conf/results/label_table-iterative.yaml",
    "/home/cc/Praxi-study/Praxi-Pipeline/prediction_openshift_image/model_testing_scripts/incremental_batchbybatch/cwd_1000_verpak_1_csoaa3000_5timesdata_datareplay1_batchbybatch8_SL_conf/results/label_table-iterative.yaml",
    "/home/cc/Praxi-study/Praxi-Pipeline/prediction_openshift_image/model_testing_scripts/incremental_batchbybatch/cwd_1000_verpak_2_csoaa3000_5timesdata_datareplay1_batchbybatch8_SL_conf/results/label_table-iterative.yaml",

    # "/home/cc/Praxi-study/Praxi-Pipeline/prediction_openshift_image/model_testing_scripts/incremental_batchbybatch/cwd_1000_verpak_0_csoaa3000_5timesdata_datareplay0_batchbybatch8_SL_conf_cost5/results/label_table-iterative.yaml",
]


fig_path = '/home/cc/Praxi-study/Praxi-Pipeline/prediction_openshift_image/figs/'
os.makedirs(fig_path, exist_ok=True)


lhs_costs_d = defaultdict(list)
true_label_costs_d = defaultdict(list)
rhs_costs_d = defaultdict(list)

for input_data_pathname, input_data_true_labels_pathname, label_table_pathname in zip(input_data_pathnames_l, input_data_true_labels_pathnames_l, label_table_pathnames_l):
    if not Path(input_data_pathname).exists():
        print(f"Missing: {input_data_pathname}")
        continue
    # output_filename = input_data_pathname.split("/")[-3]

    # Load order of labels added
    with open(label_table_pathname, 'r') as file:
        label_table = yaml.safe_load(file)
    true_line_ids = {}  # Maintain order of labels 
    for label_idx, labelname in label_table.items():
        true_line_ids[labelname] = set()

    # Load true labels
    with open(input_data_true_labels_pathname, 'r') as file:
        true_labels = yaml.safe_load(file)  # This depends on the actual structure; adjust as necessary

    # Extract ids if the ids directly correlate with the line numbers
    # true_line_ids = defaultdict(set)
    for line_idx, true_labels_per_line in enumerate(true_labels):
        for true_label in true_labels_per_line:
            true_line_ids[true_label].add(line_idx)

    for true_label_idx, true_label in enumerate(true_line_ids.keys()):
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
            
            # lhs_costs, true_label_costs, rhs_costs = [], [], []
            for label_idx, (label, costs) in enumerate(label_costs.items()):
                if label_idx < true_label_idx:
                    # lhs_costs.extend(costs)
                    lhs_costs_d[true_label_idx].extend(costs)
                if label_idx == true_label_idx:
                    # true_label_costs.extend(costs)
                    true_label_costs_d[true_label_idx].extend(costs)
                if label_idx > true_label_idx:
                    # rhs_costs.extend(costs)
                    rhs_costs_d[true_label_idx].extend(costs)

            # if not lhs_costs:
            #     lhs_costs.extend([-1,-1])
            # if not rhs_costs:
            #     rhs_costs.extend([-1,-1])
            # print(true_label_idx, true_label)
            # print(statistics.mean(lhs_costs), statistics.mean(true_label_costs), statistics.mean(rhs_costs))
            # print(statistics.stdev(lhs_costs), statistics.stdev(true_label_costs), statistics.stdev(rhs_costs))
            # print()




lhs_costs_d[min(true_label_costs_d.keys())].extend([np.nan,np.nan])
rhs_costs_d[max(true_label_costs_d.keys())].extend([np.nan,np.nan])

# for true_label_idx,  true_label_costs in true_label_costs_d.items():
#     print(true_label_idx)
#     print(statistics.mean(lhs_costs_d[true_label_idx]), statistics.mean(true_label_costs), statistics.mean(rhs_costs_d[true_label_idx]))
#     print(statistics.stdev(lhs_costs_d[true_label_idx]), statistics.stdev(true_label_costs), statistics.stdev(rhs_costs_d[true_label_idx]))
#     print()

# Arrays to store mean values
means = []

# Calculate means
for true_label_idx, true_label_costs in true_label_costs_d.items():
    lhs_mean = statistics.mean(lhs_costs_d[true_label_idx])
    true_label_mean = statistics.mean(true_label_costs)
    rhs_mean = statistics.mean(rhs_costs_d[true_label_idx])

    # Append the means to the list
    means.append([lhs_mean, true_label_mean, rhs_mean])

# Convert list of means to a NumPy array
means_array = np.array(means)

# Plotting the heatmap
fig, ax = plt.subplots(figsize=(10, 5))
# cax = ax.matshow(means_array, cmap='coolwarm', norm=LogNorm(vmin=np.nanmin(means_array[means_array > 0]), vmax=np.nanmax(means_array)))
cax = ax.matshow(means_array, cmap='coolwarm', vmin=np.nanmin(means_array), vmax=2)

# Adding color bar
fig.colorbar(cax)

# Set axis labels
ax.set_xticks(np.arange(means_array.shape[1]))
ax.set_yticks(np.arange(means_array.shape[0]))
ax.set_xticklabels(['L', 'T', 'R'])
ax.set_yticklabels([f'Label {i}' for i in true_label_costs_d.keys()])
plt.savefig(fig_path+f'cost_summary_datareplay1.pdf', bbox_inches='tight')
plt.close()