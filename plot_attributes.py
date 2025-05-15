import os
import json
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap, BoundaryNorm
from adjustText import adjust_text


log_dir = "/home/cc/Praxi-Pipeline/prediction_XGBoost_openshift_image/model_testing_scripts_online/verification/cwd_ML_with_data_4_1000_train_0shuffleidx_0testsamplebatchidx_4nsamples_1njobs_32clfnjobs_100trees_1depth_None-1rawinput_sampling1_exacttreemethod_1maxbin_modize_par_True25removesharedornoisestags_verpak_on_demand_expert_flask_client/"
json_file = "lambda_responses.json-tagset_ML_3_test-trial1"
target_dir = "tagset_ML_3_test"
dirname = "/home/cc/Praxi-Pipeline/data/data_4/"
plot_dir = "plots-2/"

os.makedirs(os.path.join(dirname, plot_dir), exist_ok=True)

with open(f"{log_dir}{json_file}", 'r') as f:
    data = json.load(f)

models = data[0]['response']['encoder_metrics']
attributes = ["selector", "gen_mat", "mat_builder", "predict_time", "list_to_mat"]
# attributes = ["predict_time"]

for attr in attributes:
    values = []
    input_sizes = []

    for model_data in models.values():
        if attr in model_data:
            values.append(model_data[attr])
            input_sizes.append(
                # model_data.get("feature_matrix_xsize", 0) * model_data.get("feature_matrix_ysize", 0)
                model_data.get("feature_matrix_xsize", 0)
            )

    values = np.array(values)
    input_sizes = np.array(input_sizes)
    model_names = np.array(list(models.keys()))

    print(f"Attribute: {attr}")
    print(f"model names: {model_names}")
    print(f"values: {values}")
    print(f"input sizes: {input_sizes}")

    sort_idx = np.argsort(values)
    sorted_values = values[sort_idx]
    sorted_input_sizes = input_sizes[sort_idx]
    cdf = np.arange(1, len(sorted_values) + 1) / len(sorted_values)
    sorted_model_names = model_names[sort_idx]

    print(f"Attribute: {attr}")
    print(f"Sorted model names: {sorted_model_names}")
    print(f"Sorted values: {sorted_values}")
    print(f"Sorted input sizes: {sorted_input_sizes}")
    # print(f"Sorted CDF: {cdf}")
    print()

    # Compute 9 quantile thresholds + top 10%
    percentile_bins = np.percentile(sorted_input_sizes, [10 * i for i in range(1, 10)])
    top_10_cutoff = np.percentile(sorted_input_sizes, 90)
    bins = [0] + list(percentile_bins) + [np.max(sorted_input_sizes) + 1]

    # Custom vivid color palette
    color_list = [
        "#d73027", "#fc8d59", "#fee090", "#ffffbf", "#d9ef8b",
        "#91cf60", "#1a9850", "#66bd63", "#3288bd", "#5e4fa2"
    ]
    cmap = ListedColormap(color_list)
    norm = BoundaryNorm(bins, cmap.N)

    # Plot
    plt.figure()
    sc = plt.scatter(sorted_values, cdf, c=sorted_input_sizes, cmap=cmap, norm=norm, s=30)
    # Annotate top 10 model names with automatic adjustment
    top_10_idx = np.argsort(values)[-20:]
    texts = []

    for idx in top_10_idx:
        model_name = list(models.keys())[idx]
        value = values[idx]
        input_size = input_sizes[idx]
        cdf_val = np.searchsorted(sorted_values, value) / len(values)

        # Add text object (but don't render yet)
        texts.append(
            plt.text(
                value,
                cdf_val,
                model_name,
                fontsize=6,
                ha='left',
                va='center'
            )
        )

    # Auto-adjust to reduce overlap
    adjust_text(texts, 
                arrowprops=dict(arrowstyle='->', color='gray', lw=0.5),
                expand_points=(1.2, 1.2),
                force_text=0.5)
    plt.title(f"CDF of {attr} (color = input matrix size bins)")
    plt.xlabel(f"{attr} value")
    plt.ylabel("CDF")
    plt.grid(True)

    # Colorbar with bin centers and custom labels
    bin_centers = [(bins[i] + bins[i + 1]) / 2 for i in range(len(bins) - 1)]
    cbar = plt.colorbar(sc, ticks=bin_centers)
    labels = [f'{int(bins[i])}–{int(bins[i + 1])}' for i in range(len(bins) - 2)]
    labels.append(f'> {int(top_10_cutoff)} (Top 10%)')
    cbar.ax.set_yticklabels(labels)
    # cbar.set_label('Feature Matrix Area (xsize × ysize)')
    cbar.set_label('Feature Matrix Area (xsize)')

    plt.tight_layout()
    # save_path = os.path.join(dirname, plot_dir, f"{target_dir}_{attr}_cdf_vivid_({json_file}).pdf")
    save_path = os.path.join(dirname, plot_dir, f"{target_dir}_{attr}_cdf_vivid_xsize_({json_file}).pdf")
    plt.savefig(save_path)
    plt.close()
