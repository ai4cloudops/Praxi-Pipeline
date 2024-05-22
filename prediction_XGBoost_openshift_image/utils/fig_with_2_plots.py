import numpy as np
import matplotlib.pyplot as plt

# Data for Set 1 and Set 2
total_clf_decompressing_time_set0 = [[14.214271068572998, 11.424229621887207, 14.370496988296509]]
total_clf_load_time_set0 = [[7.305550813674927, 6.65357780456543, 7.137504577636719]]
total_data_load_time_set0 = [[19.343135118484497, 16.416025638580322, 14.686767578125]]
# total_encoder_time_set0 = [[1.7366256713867188, 1.7113926410675049, 1.706197738647461, 2.316041946411133, 1.9386999607086182]]
encoder_gen_mapping_time_set0 = [[0.5421335697174072, 0.5344305038452148, 0.5318853855133057]]
encoder_get_feature_time_set0 = [[0.014863967895507812, 0.015505313873291016, 0.014655590057373047]]
encoder_selector_time_set0 = [[0.9775185585021973, 0.9474830627441406, 0.9641199111938477]]
encoder_mat_builder_time_set0 = [[0.0017881393432617188, 0.0018186569213867188, 0.0018553733825683594]]
encoder_list_to_mat_time_set0 = [[0.15726447105407715, 0.1549973487854004, 0.15615391731262207]]
total_inference_time_set0 = [[0.03154873847961426, 0.03263568878173828, 0.03185153007507324]]
total_decoding_time_set0 = [[0.09236693382263184, 0.05791926383972168, 0.05998539924621582]]
total_time_set0 = [[43.18622660636902, 36.7392783164978, 38.458709955215454]]

total_clf_decompressing_time_set1 = [[2.7411389350891113, 3.6058781147003174, 2.6865200996398926]]
total_clf_load_time_set1 = [[6.66372537612915, 6.671632766723633, 6.6716148853302]]
total_data_load_time_set1 = [[16.06651782989502, 17.41927194595337, 18.434086084365845]]
# total_encoder_time_set1 = [[6.972375154495239, 6.928048372268677, 6.895395994186401]]
encoder_gen_mapping_time_set1 = [[0.6138122081756592, 0.6066784858703613, 0.6052720546722412]]
encoder_get_feature_time_set1 = [[0.004528045654296875, 0.004506587982177734, 0.0046367645263671875]]
encoder_selector_time_set1 = [[1.0821304321289062, 1.0794153213500977, 1.093968391418457]]
encoder_mat_builder_time_set1 = [[0.1072537899017334, 0.1073305606842041, 0.11080574989318848]]
encoder_list_to_mat_time_set1 = [[0.033544301986694336, 0.03353619575500488, 0.03954887390136719]]
total_inference_time_set1 = [[13.857200145721436, 15.893882274627686, 14.133164882659912]]
total_decoding_time_set1 = [[0.01883840560913086, 0.01852560043334961, 0.019051551818847656]]
total_time_set1 = [[41.39731454849243, 45.659366846084595, 44.00324082374573]]

# Function to calculate mean and 2 standard deviations
def trial_stats(trial_list):
    mean_val = np.mean(trial_list)
    std_dev = np.std(trial_list) * 2  # Calculate two standard deviations
    return mean_val, std_dev

# Define indices of the components to keep ('Encode', 'Inference', 'Decode')
indices = [3, 4, 5]  # Assuming 'Encode' is 3rd, 'Inference' is 4th, 'Decode' is 5th in the original list

# Data extraction and processing for Set 1
data_set0 = [encoder_gen_mapping_time_set0[0], encoder_get_feature_time_set0[0], encoder_selector_time_set0[0], encoder_mat_builder_time_set0[0], encoder_list_to_mat_time_set0[0], total_inference_time_set0[0], total_decoding_time_set0[0]]
avg_components_set0, errors_set0 = zip(*[trial_stats(data) for data in data_set0])
avg_total_time_set0 = sum(avg_components_set0)  # New total is the sum of the included components
total_error_set0 = np.sqrt(sum(np.array(errors_set0) ** 2 / 4))  # Combine errors using quadrature

# Data extraction and processing for Set 2
data_set1 = [encoder_gen_mapping_time_set1[0], encoder_get_feature_time_set1[0], encoder_selector_time_set1[0], encoder_mat_builder_time_set1[0], encoder_list_to_mat_time_set1[0], total_inference_time_set1[0], total_decoding_time_set1[0]]
avg_components_set1, errors_set1 = zip(*[trial_stats(data) for data in data_set1])
avg_total_time_set1 = sum(avg_components_set1)  # New total is the sum of the included components
total_error_set1 = np.sqrt(sum(np.array(errors_set1) ** 2 / 4))  # Combine errors using quadrature


# Data for Set 1 and Set 2
total_clf_decompressing_time_set2 = [[13.430506944656372, 12.69049882888794, 15.26503038406372]]
total_clf_load_time_set2 = [[7.162744998931885, 7.937798261642456, 7.605685234069824]]
total_data_load_time_set2 = [[8.019973516464233, 9.227576494216919, 8.07615041732788]]
# total_encoder_time_set2 = [[1.7366256713867188, 1.7113926410675049, 1.706197738647461, 2.316041946411133, 1.9386999607086182]]
encoder_gen_mapping_time_set2 = [[0.5287468433380127, 0.5731692314147949, 0.558147668838501]]
encoder_get_feature_time_set2 = [[0.014415502548217773, 0.017134428024291992, 0.015570878982543945]]
encoder_selector_time_set2 = [[0.5390725135803223, 0.5247557163238525, 0.5774226188659668]]
encoder_mat_builder_time_set2 = [[0.0009701251983642578, 0.0009982585906982422, 0.0010325908660888672]]
encoder_list_to_mat_time_set2 = [[0.15552425384521484, 0.1760096549987793, 0.16386675834655762]]
total_inference_time_set2 = [[0.025096893310546875, 0.029154300689697266, 0.02819037437438965]]
total_decoding_time_set2 = [[0.03638482093811035, 0.03706765174865723, 0.040502309799194336]]
total_time_set2 = [[30.277548789978027, 31.59302043914795, 32.713255167007446]]

total_clf_decompressing_time_set3 = [[3.252439022064209, 3.385455846786499, 3.055346965789795]]
total_clf_load_time_set3 = [[6.767310380935669, 8.262137174606323, 7.784078121185303]]
total_data_load_time_set3 = [[9.106859683990479, 8.54998230934143, 7.889105796813965]]
# total_encoder_time_set3 = [[6.972375154495239, 6.928048372268677, 6.895395994186401]]
encoder_gen_mapping_time_set3 = [[0.591240406036377, 0.7099721431732178, 0.6318862438201904]]
encoder_get_feature_time_set3 = [[0.004441022872924805, 0.004575252532958984, 0.004580259323120117]]
encoder_selector_time_set3 = [[0.5193667411804199, 0.575005054473877, 0.5253713130950928]]
encoder_mat_builder_time_set3 = [[0.05068397521972656, 0.05514645576477051, 0.05410313606262207]]
encoder_list_to_mat_time_set3 = [[0.017007112503051758, 0.016924381256103516, 0.017632722854614258]]
total_inference_time_set3 = [[7.564438819885254, 7.895132541656494, 6.900334358215332]]
total_decoding_time_set3 = [[0.010154247283935547, 0.010138988494873047, 0.010035276412963867]]
total_time_set3 = [[28.099493741989136, 29.680709838867188, 27.093430995941162]]

# Function to calculate mean and 2 standard deviations
def trial_stats(trial_list):
    mean_val = np.mean(trial_list)
    std_dev = np.std(trial_list) * 2  # Calculate two standard deviations
    return mean_val, std_dev

# Define indices of the components to keep ('Encode', 'Inference', 'Decode')
indices = [3, 4, 5]  # Assuming 'Encode' is 3rd, 'Inference' is 4th, 'Decode' is 5th in the original list

# Data extraction and processing for Set 1
data_set2 = [encoder_gen_mapping_time_set2[0], encoder_get_feature_time_set2[0], encoder_selector_time_set2[0], encoder_mat_builder_time_set2[0], encoder_list_to_mat_time_set2[0], total_inference_time_set2[0], total_decoding_time_set2[0]]
avg_components_set2, errors_set2 = zip(*[trial_stats(data) for data in data_set2])
avg_total_time_set2 = sum(avg_components_set2)  # New total is the sum of the included components
total_error_set2 = np.sqrt(sum(np.array(errors_set2) ** 2 / 4))  # Combine errors using quadrature

# Data extraction and processing for Set 2
data_set3 = [encoder_gen_mapping_time_set3[0], encoder_get_feature_time_set3[0], encoder_selector_time_set3[0], encoder_mat_builder_time_set3[0], encoder_list_to_mat_time_set3[0], total_inference_time_set3[0], total_decoding_time_set3[0]]
avg_components_set3, errors_set3 = zip(*[trial_stats(data) for data in data_set3])
avg_total_time_set3 = sum(avg_components_set3)  # New total is the sum of the included components
total_error_set3 = np.sqrt(sum(np.array(errors_set3) ** 2 / 4))  # Combine errors using quadrature


# Example data setup
components = ['Encode: Load Feat-Idx Mapping', 'Encode: Subset Pos Weight Feats', 'Encode: Sample-Encoder Feat Intersect', 'Encode: Mat Builder', 'Encode: List to Mat', 'Inference', 'Decode']
colors = ['lightcoral', 'magenta', 'orange', 'lightblue', 'lightgreen', 'cyan', 'yellow']

# Simplified average times for four sets
avg_components = [
    avg_components_set2,  # Set 2
    avg_components_set3,  # Set 3
    avg_components_set0,  # Set 0
    avg_components_set1,  # Set 1
]

# Errors (simplified and same structure)
errors = [
    errors_set2,
    errors_set3,
    errors_set0,
    errors_set1,
]

# Create a figure and a set of subplots
fig, axs = plt.subplots(1, 2, figsize=(12, 6), sharey=True)

# Function to plot stacked bars for a set
def plot_stacked_bar(ax, position, component_averages, components, colors):
    bottom = 0
    bars = []
    for i, component_avg in enumerate(component_averages):
        bar = ax.bar(position, component_avg, width=0.35, color=colors[i], bottom=bottom, label=components[i] if bottom == 0 else "")
        bars.append(bar)
        bottom += component_avg
    return bars

# Plot for each set in each subplot
positions = [0, 1]  # Positions for the bars in each subplot
for i, pos in enumerate(positions):
    bars = plot_stacked_bar(axs[0], pos, avg_components[i], components, colors)
    plot_stacked_bar(axs[1], pos, avg_components[i+2], components, colors)

# Adding error bars to the total values
for i, pos in enumerate(positions):
    axs[0].errorbar(pos, sum(avg_components[i]), yerr=np.sqrt(sum(np.array(errors[i])**2 / 4)), fmt='none', ecolor='red', capsize=5)
    axs[1].errorbar(pos, sum(avg_components[i+2]), yerr=np.sqrt(sum(np.array(errors[i+2])**2 / 4)), fmt='none', ecolor='red', capsize=5)

# # Setting labels, ticks, and titles
# axs[0].set_title('Sets 0 and 1')
# axs[1].set_title('Sets 2 and 3')
for ax in axs:
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Set 0", "Set 1"])
axs[1].set_xticklabels(["Set 2", "Set 3"])

# Share one legend
fig.legend(bars, components, loc='upper center', ncol=4, fontsize='small')

# Adjust layout to fit the legend without overlapping
plt.tight_layout(rect=[0, 0, 1, 0.9])

# Save and show plot
plt.savefig('comparison_plot.pdf')
plt.show()
