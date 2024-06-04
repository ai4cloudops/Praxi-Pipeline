import re

def sum_training_times(logfile_path):
    total_time = 0.0
    times_l = []
    time_pattern = re.compile(r"Training took ([\d.]+) secs")
    # time_pattern = re.compile(r"vw fit took ([\d.]+) secs")
    # time_pattern = re.compile(r"rm_common_tags took ([\d.]+) secs")

    with open(logfile_path, 'r') as file:
        for line in file:
            match = time_pattern.search(line)
            if match:
                total_time += float(match.group(1))
                times_l.append(float(match.group(1)))

    return total_time, times_l

# Path to your log file
# log_path = '/home/cc/Praxi-study/Praxi-Pipeline/prediction_openshift_image/model_testing_scripts/submodeling/cwd_1000_verpak_3_csoaa3000_5timesdata_batchdatareplay0_batchbybatch10000_SL_conf_testing_filterTrue1/results/praxi_exp-0.log'
# log_path = '/home/cc/Praxi-study/Praxi-Pipeline/prediction_openshift_image/model_testing_scripts/submodeling/cwd_1000_verpak_0_csoaa3000_5timesdata_batchdatareplay0_batchbybatch10000_SL_conf_testing_filterTrue10/results/praxi_exp-0.log'
# log_path = '/home/cc/Praxi-study/Praxi-Pipeline/prediction_openshift_image/model_testing_scripts/submodeling/cwd_1000_verpak_1_csoaa3000_5timesdata_batchdatareplay0_batchbybatch10000_SL_conf_testing_filterTrue25_rerun/results/praxi_exp-0.log'
# log_path = '/home/cc/Praxi-study/Praxi-Pipeline/prediction_openshift_image/model_testing_scripts/submodeling/cwd_1000_verpak_0_csoaa3000_5timesdata_batchdatareplay0_batchbybatch10000_SL_conf_testing_filterTrue25_threads/results/praxi_exp-0.log'
# log_path = '/home/cc/Praxi-study/Praxi-Pipeline/prediction_openshift_image/model_testing_scripts/submodeling/cwd_1000_verpak_0_csoaa3000_5timesdata_batchdatareplay0_batchbybatch10000_SL_conf_testing_filterTrue25_no_cache/results/praxi_exp-0.log'
log_path = '/home/cc/Praxi-study/Praxi-Pipeline/prediction_openshift_image/model_testing_scripts/submodeling/cwd_1000_verpak_3_csoaa3000_5timesdata_batchdatareplay0_batchbybatch10000_SL_conf_testing_filterTrue25/results/praxi_exp-0.log'
# log_path = '/home/cc/Praxi-study/Praxi-Pipeline/prediction_openshift_image/model_testing_scripts/submodeling/cwd_1000_verpak_4_csoaa3000_5timesdata_batchdatareplay0_batchbybatch10000_SL_conf_testing_filterFalse100/results/praxi_exp-0.log'
total_training_time, times_l = sum_training_times(log_path)
print(f"The total training time is: {total_training_time} seconds.")
# print(times_l)
