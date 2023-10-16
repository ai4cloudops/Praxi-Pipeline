import matplotlib.pyplot as plt
import numpy as np
import math, copy
import scipy

def plotting(fig_path, filename, cates_values, labels, cates_stds= None, yaxis_label=None, xaxis_label=None, title=None, figsize=(10, 7)):
    width = 0.4

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    for i, cate_values in enumerate(cates_values):
        bottom = np.zeros(len(labels))
        for cate, value in cate_values.items():
            if labels[0]==None:
                p = ax.bar([float(entry) for entry in cate.split("-")], value, width/len(cates_values), bottom=bottom)
            else:
                p = ax.bar([idx - width/len(cates_values)/2 + i*width/len(cates_values) for idx, _ in enumerate(value)], value, width/len(cates_values), label=cate, bottom=bottom)
            bottom += value
            ax.bar_label(p)
    # if cates_stds!=None:
    #     for i, cate_stds in enumerate(cates_stds):
    #         for cate, (y, yerr) in cate_stds.items():
    #             if labels[0]==None:
    #                 p = ax.bar([float(entry) for entry in cate.split("-")], y, yerr)
    #             else:
    #                 p = ax.bar([idx - width/len(cates_stds)/2 + i*width/len(cates_stds) for idx, _ in enumerate(y)], y, yerr)

    if title == None:
        ax.set_title(" ".join(filename.split("_")))
    else:
        ax.set_title(title, fontsize=20)
    if labels[0]!=None:
        ax.legend(loc="best", prop={'size': 16})
        ax.set_xticks(list(range(len(labels))))
        ax.set_xticklabels(labels)
    if yaxis_label != None:
        ax.set_ylabel(yaxis_label, fontsize=20)
    if xaxis_label != None:
        ax.set_xlabel(xaxis_label, fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.tick_params(axis='both', which='minor', labelsize=18)

    # plt.show()
    plt.savefig(fig_path+filename+'.pdf', bbox_inches='tight')
    plt.close()



if __name__ == "__main__":
    # # by_N_estimators
    # fig_path = '/home/cc/Praxi-study/Praxi-Pipeline/prediction_XGBoost_openshift_image/model_testing_scripts/figs/'
    # filename = "inference_latency_by_N_estimators_with_data_0&2"
    # labels = (
    #     "1*",
    #     "1",
    #     "5",
    #     "10",
    #     "50",
    #     "100"
    # )
    # cate_values_1 = {
    #     "M80": np.array([1.68, 1.68, 1.71, 1.73, 1.91, 2.03]),
    #     "M9": np.array([0.05, 0.0495, 0.050, 0.052, 0.058, 0.064])
    # }
    # cate_values_2 = {
    #     "M89": np.array([1.72, 1.72, 1.75, 1.76, 1.92, 2.00])
    # }
    # cate_values = [cate_values_1, cate_values_2]
    # plotting(fig_path, filename, cate_values, labels)

    # # N models
    # fig_path = '/home/cc/Praxi-study/Praxi-Pipeline/prediction_XGBoost_openshift_image/model_testing_scripts/figs/'
    # filename = "inference_latency_by_N_models_with_data_0"
    # labels = (
    #     "8","4","1"
    # )
    # cate_values_1 = {
    #     "M12-M23-M80": np.array([0.03, 0.10, 1.87]),
    #     "M11-M18-MX": np.array([0.07, 0.23, 0]),
    #     "M7-M19-MX": np.array([0.02, 0.22, 0]),
    #     "M11-M20-MX": np.array([0.22, 1.23, 0]),
    #     "M7-MX-MX": np.array([0.07, 0, 0]),
    #     "M12-MX-MX": np.array([0.16, 0, 0]),
    #     "M10-MX-MX_1": np.array([1.07, 0, 0]),
    #     "M10-MX-MX_2": np.array([0.14, 0, 0]),
    # }
    # # cate_values_2 = {
    # #     "-": np.array([0, 0, 0])
    # # }
    # cate_values = [cate_values_1]
    # plotting(fig_path, filename, cate_values, labels)


    # fig_path = '/home/cc/Praxi-study/Praxi-Pipeline/prediction_XGBoost_openshift_image/model_testing_scripts/figs/'
    # filename = "train_latency_by_N_models_with_data_0"
    # labels = (
    #     "8_1000trees","4_2000trees","1_8000trees_64jobs","8_8000trees_64jobs","4_8000trees_64jobs"
    # )
    # cate_values_1 = {
    #     "M12-M23-M80-M12_800t-M23_400t": np.array([0.567, 4.024, 954.138, 4.745, 14.930]),
    #     "M11-M18-MX-M11_800t-M18_400t": np.array([1.965, 9.069, 0, 13.004, 37.666]),
    #     "M7-M19-MX-M17_800t-M19_400t": np.array([0.391, 9.117, 0, 2.896, 46.874]),
    #     "M11-M20-MX-M11_800t-M20_400t": np.array([5.592, 163.841, 0, 41.572, 660.138]),
    #     "M7-MX-MX-M7_800t-MX": np.array([1.020, 0, 0, 8.108, 0]),
    #     "M12-MX-MX-M12_800t-MX": np.array([4.595, 0, 0, 33.791, 0]),
    #     "M10-MX-MX_1-M10_800t-MX": np.array([71.705, 0, 0, 567.631, 0]),
    #     "M10-MX-MX_2-M10_800t-MX": np.array([3.430, 0, 0, 23.669, 0]),
    # }
    # # cate_values_2 = {
    # #     "-": np.array([0, 0, 0])
    # # }
    # cate_values = [cate_values_1]
    # plotting(fig_path, filename, cate_values, labels)


    # fig_path = '/home/cc/Praxi-study/Praxi-Pipeline/prediction_XGBoost_openshift_image/model_testing_scripts/figs/'
    # filename = "train_latency_by_N_models_with_data_0"
    # labels = (
    #     "8","4","1"
    # )
    # cate_values_1 = {
    #     "M10-M20-M80": np.array([4.351, 19.378, 1012.601]),
    #     "M10-M20-MX_1": np.array([4.476, 19.424, 0]),
    #     "M10-M20-MX_2": np.array([4.585, 19.025, 0]),
    #     "M10-M20-MX_3": np.array([4.337, 18.938, 0]),
    #     "M10-MX-MX_1": np.array([4.997, 0, 0]),
    #     "M10-MX-MX_2": np.array([4.649, 0, 0]),
    #     "M10-MX-MX_3": np.array([6.566, 0, 0]),
    #     "M10-MX-MX_4": np.array([4.633, 0, 0]),
    # }
    # # cate_values_2 = {
    # #     "-": np.array([0, 0, 0])
    # # }
    # cate_values = [cate_values_1]
    # plotting(fig_path, filename, cate_values, labels)


    # fig_path = '/home/cc/Praxi-study/Praxi-Pipeline/prediction_XGBoost_openshift_image/model_testing_scripts/figs/'
    # filename = "train_latency_by_N_models_with_data_0_randomint10000000_109319_64jobs"
    # labels = (
    #     "8","4","1"
    # )
    # cate_values_1 = {
    #     "M10-M20-M80": np.array([32.517, 170.592, 2211.328]),
    #     "M10-M20-MX_1": np.array([30.359, 184.956, 0]),
    #     "M10-M20-MX_2": np.array([33.489, 180.434, 0]),
    #     "M10-M20-MX_3": np.array([31.810, 174.204, 0]),
    #     "M10-MX-MX_1": np.array([29.711, 0, 0]),
    #     "M10-MX-MX_2": np.array([32.490, 0, 0]),
    #     "M10-MX-MX_3": np.array([30.686, 0, 0]),
    #     "M10-MX-MX_4": np.array([33.413, 0, 0]),
    # }
    # # cate_values_2 = {
    # #     "-": np.array([0, 0, 0])
    # # }
    # cate_values = [cate_values_1]
    # plotting(fig_path, filename, cate_values, labels)


    # fig_path = '/home/cc/Praxi-study/Praxi-Pipeline/prediction_XGBoost_openshift_image/model_testing_scripts/figs/'
    # filename = "train_latency_by_N_models_with_data_0_zeros_109319_4jobs"
    # labels = (
    #     "8","4","2","1"
    # )
    # cate_values_1 = {
    #     "M10-M20-M40-M80": np.array([3.763, 15.673, 62.270, 1609.677]),
    #     "M10-M20-M40-MX_1": np.array([3.946, 15.368, 64.685, 0]),
    #     "M10-M20-M40-MX_2": np.array([3.891, 15.265, 0, 0]),
    #     "M10-M20-M40-MX_3": np.array([3.959, 16.010, 0, 0]),
    #     "M10-MX-MX-MX_1": np.array([3.846, 0, 0, 0]),
    #     "M10-MX-MX-MX_2": np.array([3.881, 0, 0, 0]),
    #     "M10-MX-MX-MX_3": np.array([3.975, 0, 0, 0]),
    #     "M10-MX-MX-MX_4": np.array([3.967, 0, 0, 0]),
    # }
    # # cate_values_2 = {
    # #     "-": np.array([0, 0, 0])
    # # }
    # cate_values = [cate_values_1]
    # plotting(fig_path, filename, cate_values, labels)

    # fig_path = '/home/cc/Praxi-study/Praxi-Pipeline/prediction_XGBoost_openshift_image/model_testing_scripts/figs/'
    # filename = "train_latency_by_N_models_with_data_0_zeros_109319_8jobs"
    # labels = (
    #     "8","4","2","1"
    # )
    # cate_values_1 = {
    #     "M10-M20-M40-M80": np.array([3.916, 15.770, 62.100, 870.774]),
    #     "M10-M20-M40-MX_1": np.array([4.182, 15.187, 64.122, 0]),
    #     "M10-M20-M40-MX_2": np.array([3.854, 15.509, 0, 0]),
    #     "M10-M20-M40-MX_3": np.array([3.911, 15.700, 0, 0]),
    #     "M10-MX-MX-MX_1": np.array([3.835, 0, 0, 0]),
    #     "M10-MX-MX-MX_2": np.array([3.966, 0, 0, 0]),
    #     "M10-MX-MX-MX_3": np.array([3.861, 0, 0, 0]),
    #     "M10-MX-MX-MX_4": np.array([3.918, 0, 0, 0]),
    # }
    # # cate_values_2 = {
    # #     "-": np.array([0, 0, 0])
    # # }
    # cate_values = [cate_values_1]
    # plotting(fig_path, filename, cate_values, labels)


    # fig_path = '/home/cc/Praxi-study/Praxi-Pipeline/prediction_XGBoost_openshift_image/model_testing_scripts/figs/'
    # filename = "train_latency_by_N_models_with_data_0_zeros_109319_16jobs"
    # labels = (
    #     "8","4","2","1"
    # )
    # cate_values_1 = {
    #     "M10-M20-M40-M80": np.array([3.978, 15.304, 63.182, 934.183]),
    #     "M10-M20-M40-MX_1": np.array([4.342, 14.966, 67.809, 0]),
    #     "M10-M20-M40-MX_2": np.array([3.833, 15.322, 0, 0]),
    #     "M10-M20-M40-MX_3": np.array([3.941, 15.755, 0, 0]),
    #     "M10-MX-MX-MX_1": np.array([4.139, 0, 0, 0]),
    #     "M10-MX-MX-MX_2": np.array([4.078, 0, 0, 0]),
    #     "M10-MX-MX-MX_3": np.array([3.850, 0, 0, 0]),
    #     "M10-MX-MX-MX_4": np.array([3.854, 0, 0, 0]),
    # }
    # # cate_values_2 = {
    # #     "-": np.array([0, 0, 0])
    # # }
    # cate_values = [cate_values_1]
    # plotting(fig_path, filename, cate_values, labels)


    # fig_path = '/home/cc/Praxi-study/Praxi-Pipeline/prediction_XGBoost_openshift_image/model_testing_scripts/figs/'
    # filename = "train_latency_by_N_models_with_data_0_zeros_109319_32jobs"
    # labels = (
    #     "8","4","2","1"
    # )
    # cate_values_1 = {
    #     "M10-M20-M40-M80": np.array([4.062, 16.344, 69.733, 953.460]),
    #     "M10-M20-M40-MX_1": np.array([4.249, 16.941, 70.464, 0]),
    #     "M10-M20-M40-MX_2": np.array([4.023, 16.753, 0, 0]),
    #     "M10-M20-M40-MX_3": np.array([4.315, 16.776, 0, 0]),
    #     "M10-MX-MX-MX_1": np.array([4.178, 0, 0, 0]),
    #     "M10-MX-MX-MX_2": np.array([3.963, 0, 0, 0]),
    #     "M10-MX-MX-MX_3": np.array([4.115, 0, 0, 0]),
    #     "M10-MX-MX-MX_4": np.array([4.303, 0, 0, 0]),
    # }
    # # cate_values_2 = {
    # #     "-": np.array([0, 0, 0])
    # # }
    # cate_values = [cate_values_1]
    # plotting(fig_path, filename, cate_values, labels)



    # fig_path = '/home/cc/Praxi-study/Praxi-Pipeline/prediction_XGBoost_openshift_image/model_testing_scripts/figs/'
    # filename = "train_latency_by_N_models_with_data_0_zeros_109319_64jobs"
    # labels = (
    #     "8","4","2","1"
    # )
    # cate_values_1 = {
    #     "M10-M20-M40-M80": np.array([4.141, 17.000, 72.818, 941.116]),
    #     "M10-M20-M40-MX_1": np.array([4.319, 16.714, 70.411, 0]),
    #     "M10-M20-M40-MX_2": np.array([4.381, 16.409, 0, 0]),
    #     "M10-M20-M40-MX_3": np.array([4.169, 17.238, 0, 0]),
    #     "M10-MX-MX-MX_1": np.array([4.085, 0, 0, 0]),
    #     "M10-MX-MX-MX_2": np.array([4.419, 0, 0, 0]),
    #     "M10-MX-MX-MX_3": np.array([4.298, 0, 0, 0]),
    #     "M10-MX-MX-MX_4": np.array([4.306, 0, 0, 0]),
    # }
    # # cate_values_2 = {
    # #     "-": np.array([0, 0, 0])
    # # }
    # cate_values = [cate_values_1]
    # plotting(fig_path, filename, cate_values, labels)



    # fig_path = '/home/cc/Praxi-study/Praxi-Pipeline/prediction_XGBoost_openshift_image/model_testing_scripts/figs/'
    # filename = "train_latency_by_N_models_with_data_0_zeros_109319_128jobs"
    # labels = (
    #     "8","4","2","1"
    # )
    # cate_values_1 = {
    #     "M10-M20-M40-M80": np.array([4.458, 18.530, 70.480, 989.714]),
    #     "M10-M20-M40-MX_1": np.array([4.408, 17.776, 70.406, 0]),
    #     "M10-M20-M40-MX_2": np.array([4.385, 18.414, 0, 0]),
    #     "M10-M20-M40-MX_3": np.array([4.235, 18.987, 0, 0]),
    #     "M10-MX-MX-MX_1": np.array([4.712, 0, 0, 0]),
    #     "M10-MX-MX-MX_2": np.array([4.650, 0, 0, 0]),
    #     "M10-MX-MX-MX_3": np.array([5.174, 0, 0, 0]),
    #     "M10-MX-MX-MX_4": np.array([4.713, 0, 0, 0]),
    # }
    # # cate_values_2 = {
    # #     "-": np.array([0, 0, 0])
    # # }
    # cate_values = [cate_values_1]
    # plotting(fig_path, filename, cate_values, labels)

    # fig_path = '/home/cc/Praxi-study/Praxi-Pipeline/prediction_XGBoost_openshift_image/model_testing_scripts/figs/'
    # filename = "input_size_by_N_models_with_data_0"
    # labels = (
    #     "8","4","1"
    # )
    # cate_values_1 = {
    #     "M12-M23-M80": np.array([510, 4183, 109319]),
    #     "M11-M18-MX": np.array([3691, 13930, 0]),
    #     "M7-M19-MX": np.array([454, 13182, 0]),
    #     "M11-M20-MX": np.array([13495, 78424, 0]),
    #     "M7-MX-MX": np.array([3560, 0, 0]),
    #     "M12-MX-MX": np.array([9740, 0, 0]),
    #     "M10-MX-MX_1": np.array([70225, 0, 0]),
    #     "M10-MX-MX_2": np.array([8498, 0, 0]),
    # }
    # cate_values = [cate_values_1]
    # plotting(fig_path, filename, cate_values, labels)

    # fig_path = '/home/cc/Praxi-study/Praxi-Pipeline/prediction_XGBoost_openshift_image/model_testing_scripts/figs/'
    # filename = "input_size_by_N_models_with_data_0"
    # labels = (
    #     "8","4","1"
    # )
    # cate_values_1 = {
    #     "M10-M20-M80": np.array([13664, 27329, 109319]),
    #     "M10-M20-MX_1": np.array([13664, 27329, 0]),
    #     "M10-M20-MX_2": np.array([13664, 27329, 0]),
    #     "M10-M20-MX_3": np.array([13664, 27329, 0]),
    #     "M10-MX-MX_1": np.array([13664, 0, 0]),
    #     "M10-MX-MX_2": np.array([13664, 0, 0]),
    #     "M10-MX-MX_3": np.array([13664, 0, 0]),
    #     "M10-MX-MX_4": np.array([13664, 0, 0]),
    # }
    # cate_values = [cate_values_1]
    # plotting(fig_path, filename, cate_values, labels)


    # fig_path = '/home/cc/Praxi-study/Praxi-Pipeline/prediction_XGBoost_openshift_image/model_testing_scripts/figs/'
    # filename = "Traintime_by_Inputsize_for_Fixed_10_Labels"
    

    # data = np.array([])
    # labels = [None]*len(data)
    # cate_values_1 = {
    #     "": data
    # }
    # cate_values = [cate_values_1]
    # plotting(fig_path, filename, cate_values, labels)


    # fig_path = '/home/cc/Praxi-study/Praxi-Pipeline/prediction_XGBoost_openshift_image/model_testing_scripts/figs/'
    # filename = "Numoftrees_by_N_models_with_data_0"
    # labels = (
    #     "8","4","1"
    # )
    # cate_values_1 = {
    #     "M12-M23-M80": np.array([1200, 2300, 8000]),
    #     "M11-M18-MX": np.array([1100, 1800, 0]),
    #     "M7-M19-MX": np.array([700, 1900, 0]),
    #     "M11-M20-MX": np.array([1100, 2000, 0]),
    #     "M7-MX-MX": np.array([700, 0, 0]),
    #     "M12-MX-MX": np.array([1200, 0, 0]),
    #     "M10-MX-MX_1": np.array([1000, 0, 0]),
    #     "M10-MX-MX_2": np.array([1000, 0, 0]),
    # }
    # cate_values = [cate_values_1]
    # plotting(fig_path, filename, cate_values, labels)

    # fig_path = '/home/cc/Praxi-study/Praxi-Pipeline/prediction_XGBoost_openshift_image/model_testing_scripts/figs/'
    # filename = "Numoflabels_by_N_models_with_data_0"
    # labels = (
    #     "8","4","1"
    # )
    # cate_values_1 = {
    #     "M12-M23-M80": np.array([12, 23, 80]),
    #     "M11-M18-MX": np.array([11, 18, 0]),
    #     "M7-M19-MX": np.array([7, 19, 0]),
    #     "M11-M20-MX": np.array([11, 20, 0]),
    #     "M7-MX-MX": np.array([7, 0, 0]),
    #     "M12-MX-MX": np.array([12, 0, 0]),
    #     "M10-MX-MX_1": np.array([10, 0, 0]),
    #     "M10-MX-MX_2": np.array([10, 0, 0]),
    # }
    # cate_values = [cate_values_1]
    # plotting(fig_path, filename, cate_values, labels)


    fig_path = '/home/cc/Praxi-study/Praxi-Pipeline/prediction_XGBoost_openshift_image/model_testing_scripts/figs/'
    filename = "outputsize_by_N_models_with_rawinput_data_0"
    labels = (
        "8","4","2","1"
    )
    cate_values_1 = {
        "M10-M20-M40-M80": np.array([13664, 27329, 54659, 109319]),
        "M10-M20-M40-MX_1": np.array([13664, 27329, 54659, 0]),
        "M10-M20-M40-MX_2": np.array([13664, 27329, 0, 0]),
        "M10-M20-M40-MX_3": np.array([13664, 27329, 0, 0]),
        "M10-MX-MX-MX_1": np.array([13664, 0, 0, 0]),
        "M10-MX-MX-MX_2": np.array([13664, 0, 0, 0]),
        "M10-MX-MX-MX_3": np.array([13664, 0, 0, 0]),
        "M10-MX-MX-MX_4": np.array([13664, 0, 0, 0]),
    }
    # cate_values_2 = {
    #     "-": np.array([0, 0, 0])
    # }
    cate_values = [cate_values_1]
    plotting(fig_path, filename, cate_values, labels)



    fig_path = '/home/cc/Praxi-study/Praxi-Pipeline/prediction_XGBoost_openshift_image/model_testing_scripts/figs/'
    filename = "trainlatency_by_N_models_with_rawinput_data_0"
    labels = (
        "8","4","2","1"
    )
    cate_values_1 = {
        "M10-M20-M40-M80": np.array([2.407, 18.840, 143.565, 1090.042]),
        "M10-M20-M40-MX_1": np.array([2.396, 18.788, 143.717, 0]),
        "M10-M20-M40-MX_2": np.array([2.413, 19.016, 0, 0]),
        "M10-M20-M40-MX_3": np.array([2.407, 18.862, 0, 0]),
        "M10-MX-MX-MX_1": np.array([2.406, 0, 0, 0]),
        "M10-MX-MX-MX_2": np.array([2.396, 0, 0, 0]),
        "M10-MX-MX-MX_3": np.array([2.440, 0, 0, 0]),
        "M10-MX-MX-MX_4": np.array([2.396, 0, 0, 0]),
    }
    cate_values_2 = {
        "Estimated": np.array([8*2.4, 4*2.4*2*(4*math.log(25*20)/math.log(25*10)), 2*19.0*2*(4*math.log(25*40)/math.log(25*20)), 143.5*2*(4*math.log(25*80)/math.log(25*40))])
    }
    cate_values = [cate_values_1, cate_values_2]
    plotting(fig_path, filename, cate_values, labels, xaxis_label="Number of Models", yaxis_label="Training Time(s)", title="Train Latency by N Models with the Same Train Dataset")

    # fig_path = '/home/cc/Praxi-study/Praxi-Pipeline/prediction_XGBoost_openshift_image/model_testing_scripts/figs/'
    # filename = "trainlatencypermodel_by_N_models_with_rawinput_data_0"
    # labels = (
    #     "10","20","40","80"
    # )
    # cate_values_1 = {
    #     "M10-M20-M40-M80": np.array([2.407, 18.840, 143.565, 1090.042]),
    #     # "M10-M20-M40-MX_1": np.array([2.396, 18.788, 143.717, 0]),
    #     # "M10-M20-M40-MX_2": np.array([2.413, 19.016, 0, 0]),
    #     # "M10-M20-M40-MX_3": np.array([2.407, 18.862, 0, 0]),
    #     # "M10-MX-MX-MX_1": np.array([2.406, 0, 0, 0]),
    #     # "M10-MX-MX-MX_2": np.array([2.396, 0, 0, 0]),
    #     # "M10-MX-MX-MX_3": np.array([2.440, 0, 0, 0]),
    #     # "M10-MX-MX-MX_4": np.array([2.396, 0, 0, 0]),
    # }
    # # cate_values_2 = {
    # #     "Estimated": np.array([8*2.4, 4*2.4*2*(4*math.log(25*20)/math.log(25*10)), 2*19.0*2*(4*math.log(25*40)/math.log(25*20)), 143.5*2*(4*math.log(25*80)/math.log(25*40))])
    # # }
    # cate_values = [cate_values_1]
    # plotting(fig_path, filename, cate_values, labels, xaxis_label="Number of Labels Per Model", yaxis_label="Training Time(s)", title="Train Latency by N Models with the Same Train Dataset")


    fig_path = '/home/cc/Praxi-study/Praxi-Pipeline/prediction_XGBoost_openshift_image/model_testing_scripts/figs/'
    filename = "trainlatencypermodel_by_N_models_with_rawinput_data_0"
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    xs=["10","20","40","80"]
    # labels = ("Observed", "Estimated")
    ys=[2.407, 18.840, 143.565, 1090.042]
    # for ys, label in zip(ys_l, labels):
    p = ax.bar(xs, ys)
    ax.bar_label(p)
    ax.set_xticks(list(range(len(xs))))
    ax.set_xticklabels(xs)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.tick_params(axis='both', which='minor', labelsize=18)
    ax.set_xlabel("Number of Packages", fontsize=20)
    ax.set_ylabel("Training Time(s)", fontsize=20)
    # plt.legend(prop={'size': 16})
    # plt.show()
    plt.savefig(fig_path+filename+'.pdf', bbox_inches='tight')
    plt.close()

    # fig_path = '/home/cc/Praxi-study/Praxi-Pipeline/prediction_XGBoost_openshift_image/model_testing_scripts/figs/'
    # filename = "inferencelatency_by_N_models_with_rawinput_data_0"
    # labels = (
    #     "8","4","2","1"
    # )
    # cate_values_1_l = [{
    #     "M10-M20-M40-M80": np.array([0.026, 0.099, 0.179, 2.072]),
    #     "M10-M20-M40-MX_1": np.array([0.075, 0.090, 1.785, 0]),
    #     "M10-M20-M40-MX_2": np.array([0.025, 0.398, 0, 0]),
    #     "M10-M20-M40-MX_3": np.array([0.067, 1.325, 0, 0]),
    #     "M10-MX-MX-MX_1": np.array([0.236, 0, 0, 0]),
    #     "M10-MX-MX-MX_2": np.array([0.157, 0, 0, 0]),
    #     "M10-MX-MX-MX_3": np.array([1.110, 0, 0, 0]),
    #     "M10-MX-MX-MX_4": np.array([0.149, 0, 0, 0]),
    # },{
    #     "M10-M20-M40-M80": np.array([0.026, 0.092, 0.171, 1.954]),
    #     "M10-M20-M40-MX_1": np.array([0.073, 0.085, 1.713, 0]),
    #     "M10-M20-M40-MX_2": np.array([0.023, 0.374, 0, 0]),
    #     "M10-M20-M40-MX_3": np.array([0.064, 1.236, 0, 0]),
    #     "M10-MX-MX-MX_1": np.array([0.234, 0, 0, 0]),
    #     "M10-MX-MX-MX_2": np.array([0.148, 0, 0, 0]),
    #     "M10-MX-MX-MX_3": np.array([1.086, 0, 0, 0]),
    #     "M10-MX-MX-MX_4": np.array([0.142, 0, 0, 0]),
    # },{
    #     "M10-M20-M40-M80": np.array([0.024, 0.092, 0.171, 2.192]),
    #     "M10-M20-M40-MX_1": np.array([0.073, 0.088, 1.676, 0]),
    #     "M10-M20-M40-MX_2": np.array([0.023, 0.373, 0, 0]),
    #     "M10-M20-M40-MX_3": np.array([0.066, 1.224, 0, 0]),
    #     "M10-MX-MX-MX_1": np.array([0.235, 0, 0, 0]),
    #     "M10-MX-MX-MX_2": np.array([0.149, 0, 0, 0]),
    #     "M10-MX-MX-MX_3": np.array([1.072, 0, 0, 0]),
    #     "M10-MX-MX-MX_4": np.array([0.143, 0, 0, 0]),
    # }]
    # cate_values_1 = {"M10-M20-M40-M80": [],
    #     "M10-M20-M40-MX_1": [],
    #     "M10-M20-M40-MX_2": [],
    #     "M10-M20-M40-MX_3": [],
    #     "M10-MX-MX-MX_1": [],
    #     "M10-MX-MX-MX_2": [],
    #     "M10-MX-MX-MX_3": [],
    #     "M10-MX-MX-MX_4": []}
    # cate_stds_1 = {"errors": []}
    # for d in cate_values_1_l:
    #     for k, v in cate_values_1.items():
    #         cate_values_1[k].append(d[k])
    # for k, v in cate_values_1.items():
    #     cate_values_1[k] = np.vstack(cate_values_1[k])
    #     cate_stds_1[k] = np.var(cate_values_1[k], axis=0)
    #     cate_values_1[k] = np.mean(cate_values_1[k], axis=0)
    # cate_values = [cate_values_1]
    # cate_stds = [cate_stds_1]
    # plotting(fig_path, filename, cate_values, labels, cates_stds=cate_stds, xaxis_label="Number of Labels Per Model", yaxis_label="Inference Time(s)", title="Inference Latency by N Models with the Same Test Dataset")









    fig_path = '/home/cc/Praxi-study/Praxi-Pipeline/prediction_XGBoost_openshift_image/model_testing_scripts/figs/'
    filename = "trainlatency_by_input_size_and_by_N_models_with_data_0_3d"
    # Fixing random state for reproducibility
    np.random.seed(19680801)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    xs=[13, 106, 284, 427, 854, 1138, 1708, 3416, 6832, 13664, 27329, 54659, 109319, 
        13, 106, 284, 427, 854, 1138, 1708, 3416, 6832, 13664, 27329, 54659, 109319, 
        13, 106, 284, 427, 854, 1138, 1708, 3416, 6832, 13664, 27329, 54659, 109319, 
        13, 106, 284, 427, 854, 1138, 1708, 3416, 6832, 13664, 27329, 54659, 109319]   # input sizes
    ys=[80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 
        40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 
        20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 
        10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10]   # number of labels
    zs=[1.227, 2.012, 3.810, 5.275, 9.333, 12.124, 17.675, 34.274, 67.655, 136.472, 274.488, 546.612, 1096.709, 
        0.315, 0.550, 1.027, 1.514, 2.580, 3.246, 4.596, 8.739, 17.492, 34.972, 70.854, 143.565, 286.711, 
        0.229, 0.290, 0.406, 0.451, 0.700, 0.928, 1.255, 2.366, 4.464, 9.328, 18.840, 39.945, 80.171, 
        0.102, 0.103, 0.143, 0.156, 0.226, 0.278, 0.392, 0.664, 1.205, 2.407, 5.449, 11.784, 24.566]   # training latency
    # xs=[13, 106, 854, 13, 106, 854, 13, 106, 854, 13, 106, 854]   # input sizes
    # ys=[80, 80, 80, 40, 40, 40, 20, 20, 20, 10, 10, 10]   # number of labels
    # zs=[1.227, 2.012, 9.333, 0.315, 0.550, 2.580, 0.229, 0.290, 0.700, 0.102, 0.103, 0.226]   # training latency

    ax.scatter(xs, ys, zs, marker='o')

    ax.set_xlabel('Feature Dimensions')
    ax.set_ylabel('Labels Per Model')
    ax.set_zlabel('Training Latency')

    # plt.show()
    plt.savefig(fig_path+filename+'.pdf', bbox_inches='tight')
    plt.close()






    fig_path = '/home/cc/Praxi-study/Praxi-Pipeline/prediction_XGBoost_openshift_image/model_testing_scripts/figs/'
    filename = "trainlatencypermodel_by_N_models_with_rawinput_data_3"
    xs=[str(xlabel) for xlabel in [500//50,500//25,500//20,500//15+1,500//10,500//5,500//1]]
    labels = ["model "+str(i+1) for i in range(46)]
    ys0_l=[[0.164, 0.665, 1.129, 2.287, 6.121, 45.595, 9103.478]]
    ys0mean_l = np.array(ys0_l).mean(axis=0).tolist()
    ys0std_l  = np.array(ys0_l).std(axis=0).tolist()
    # ys0conf_l = list(scipy.stats.t.interval(0.95, len(ys0_l)-1, loc=np.mean(ys0_l,axis=0), scale=scipy.stats.sem(ys0_l,axis=0)))
    # ys1_l = [1286.38/(5*5)]
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    # bottom = np.zeros(len(xs))
    entry_count, width = 2, 0.4
    p = ax.bar([idx for idx, _ in enumerate(ys0mean_l)], ys0mean_l, width/entry_count, yerr=ys0std_l)
    # ax.errorbar([idx for idx, _ in enumerate(ys0mean_l)], ys0mean_l, yerr=ys0std_l)
    # p = ax.bar([idx - width/entry_count/2 + 0*width/entry_count for idx, _ in enumerate(ys)], ys, width/entry_count, label="observed")
    ax.bar_label(p)
    # p = ax.bar([idx - width/entry_count/2 + 1*width/entry_count for idx, _ in enumerate(ys1_l)], ys1_l, width/entry_count, label="estimated")
    # ax.set_title("Training Latency by N Models with Data 3", fontsize=20)
    # ax.legend(loc="best", prop={'size': 16})
    ax.set_xticks(list(range(len(xs))))
    ax.set_xticklabels(xs)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.tick_params(axis='both', which='minor', labelsize=18)
    ax.set_xlabel("Number of Labels", fontsize=20)
    ax.set_ylabel("Training Time(s)", fontsize=20)
    # plt.legend(prop={'size': 16})
    # plt.show()
    plt.savefig(fig_path+filename+'.pdf', bbox_inches='tight')
    plt.close()



    fig_path = '/home/cc/Praxi-study/Praxi-Pipeline/prediction_XGBoost_openshift_image/model_testing_scripts/figs/'
    filename = "nerccostandtrainlatencypermodel_by_N_models_with_rawinput_data_3"
    xs=[str(xlabel) for xlabel in [500//50,500//25,500//20,500//15+1,500//10,500//5,500//1]]
    labels = ["model "+str(i+1) for i in range(46)]
    ys0_l=[[round(0.1648/60/60*0.013*8, 2), round(0.665/60/60*0.013*8, 2), round(1.129/60/60*0.013*8, 2), round(2.287/60/60*0.013*8, 2), round(6.121/60/60*0.013*8, 2), round(45.595/60/60*0.013*8, 2), round(9103.478/60/60*0.013*8, 2)]]
    ys0mean_l = np.array(ys0_l).mean(axis=0).tolist()
    ys0std_l  = np.array(ys0_l).std(axis=0).tolist()
    ys1_l=[[0.164, 0.665, 1.129, 2.287, 6.121, 45.595, 9103.478]]
    ys1mean_l = np.array(ys1_l).mean(axis=0).tolist()
    ys1std_l  = np.array(ys1_l).std(axis=0).tolist()
    # ys0conf_l = list(scipy.stats.t.interval(0.95, len(ys0_l)-1, loc=np.mean(ys0_l,axis=0), scale=scipy.stats.sem(ys0_l,axis=0)))
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    # bottom = np.zeros(len(xs))
    entry_count, width = 2, 0.4
    p = ax.bar([idx-0.1 for idx, _ in enumerate(ys0mean_l)], ys0mean_l, width/entry_count, yerr=ys0std_l)
    # ax.bar_label(p)
    # ax.set_title("Training Latency by N Models with Data 3", fontsize=20)
    # ax.legend(loc="best", prop={'size': 16})
    ax.set_xticks(list(range(len(xs))))
    ax.set_xticklabels(xs)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.tick_params(axis='both', which='minor', labelsize=18)
    ax.set_xlabel("Number of Labels", fontsize=20)
    ax.set_ylabel("NERC Openstack Cost ($)", fontsize=20)
    ax1 = ax.twinx()
    p = ax1.bar([idx+0.1 for idx, _ in enumerate(ys1mean_l)], ys1mean_l, width/entry_count, yerr=ys1std_l)
    # ax1.bar_label(p)
    ax1.set_ylabel("Training Time(s)", fontsize=20)
    # plt.legend(prop={'size': 16})
    # plt.show()
    plt.savefig(fig_path+filename+'.pdf', bbox_inches='tight')
    plt.close()








    # by_input_size
    fig_path = '/home/cc/Praxi-study/Praxi-Pipeline/prediction_XGBoost_openshift_image/model_testing_scripts/figs/'
    filename = "testf1score_by_input_size_with_rawinput_data_0"
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    xs=[13, 106, 284, 427, 854, 1138, 1708, 3416, 6832, 13664, 27329, 54659, 109319]
    labels = ("80 labels", "40 labels","20 labels","10 labels")
    ys_l=[[0.077, 0.521, 0.697, 0.727, 0.806, 0.843, 0.847, 0.906, 0.935, 0.934, 0.948, 0.951, 0.954],
        [0.166, 0.609, 0.746, 0.789, 0.857, 0.846, 0.891, 0.896, 0.911, 0.905, 0.911, 0.904, 0.916],
        [0.210, 0.633, 0.758, 0.763, 0.829, 0.841, 0.828, 0.886, 0.889, 0.884, 0.902, 0.891, 0.898],
        [0.259, 0.588, 0.714, 0.775, 0.741, 0.836, 0.851, 0.837, 0.815, 0.843, 0.842, 0.846, 0.843]
        ]
    for ys, label in zip(ys_l, labels):
        ax.scatter(xs, ys, label=label)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.tick_params(axis='both', which='minor', labelsize=18)
    ax.set_xlabel("Input Dimensions", fontsize=20)
    ax.set_ylabel("F1-Scores", fontsize=20)
    plt.legend(prop={'size': 16})
    # plt.show()
    plt.savefig(fig_path+filename+'.pdf', bbox_inches='tight')
    plt.close()

    fig_path = '/home/cc/Praxi-study/Praxi-Pipeline/prediction_XGBoost_openshift_image/model_testing_scripts/figs/'
    filename = "trainlatency_by_input_size_with_rawinput_data_0"
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    xs=[13, 106, 284, 427, 854, 1138, 1708, 3416, 6832, 13664, 27329, 54659, 109319]
    labels = ("10 labels", "20 labels","40 labels","80 labels")
    ys_l=[[0.102, 0.103, 0.143, 0.156, 0.226, 0.278, 0.392, 0.664, 1.205, 2.407, 5.449, 11.784, 24.566],
        [0.229, 0.290, 0.406, 0.451, 0.700, 0.928, 1.255, 2.366, 4.464, 9.328, 18.840, 39.945, 80.171],
        [0.315, 0.550, 1.027, 1.514, 2.580, 3.246, 4.596, 8.739, 17.492, 34.972, 70.854, 143.565, 286.711],
        [1.227, 2.012, 3.810, 5.275, 9.333, 12.124, 17.675, 34.274, 67.655, 136.472, 274.488, 546.612, 1096.709]
        ]
    for ys, label in zip(ys_l, labels):
        ax.scatter(xs, ys, label=label)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.tick_params(axis='both', which='minor', labelsize=18)
    ax.set_xlabel("Input Dimensions", fontsize=20)
    ax.set_ylabel("Training Time(s)", fontsize=20)
    plt.legend(prop={'size': 16})
    # plt.show()
    plt.savefig(fig_path+filename+'.pdf', bbox_inches='tight')
    plt.close()

    filename = "inferencelatency_by_input_size_with_rawinput_data_0"
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    # xs=[13, 106, 284, 427, 854, 1138, 1708, 3416, 6832, 13664, 27329, 54659, 109319]
    # labels = ("10 labels", "20 labels","40 labels","80 labels")
    # ys_l=[[0.017+0.015+0.015+0.015+0.015+0.015+0.015+0.015, 0.017+0.024+0.017+0.017+0.017+0.018+0.018+0.018, 0.020+0.020+0.020+0.020+0.021+0.021+0.021+0.022, 0.022+0.023+0.023+0.023+0.024+0.023+0.023+0.024, 0.030+0.030+0.030+0.030+0.031+0.031+0.031+0.031, 0.034+0.036+0.035+0.034+0.035+0.035+0.035+0.035, 0.045+0.042+0.042+0.042+0.042+0.044+0.043+0.043, 0.065+0.065+0.067+0.066+0.065+0.065+0.065+0.065, 0.114+0.115+0.114+0.114+0.114+0.114+0.115+0.115, 0.212+0.214+0.214+0.212+0.214+0.213+0.213+0.218, 0.408+0.409+0.411+0.411+0.409+0.409+0.410+0.409, 0.810+0.817+0.817+0.815+0.814+0.809+0.814+0.813, 1.675+1.686+1.681+1.677+1.672+1.679+1.679+1.678],
    #     [0.030+0.031+0.031+0.030, 0.031+0.032+0.031+0.032, 0.034+0.034+0.034+0.034, 0.036+0.037+0.037+0.037, 0.045+0.044+0.045+0.045, 0.047+0.047+0.048+0.050, 0.055+0.059+0.056+0.056, 0.079+0.079+0.079+0.081, 0.128+0.132+0.129+0.131, 0.226+0.226+0.225+0.227, 0.425+0.424+0.423+0.425, 0.831+0.830+0.831+0.836, 1.711+1.709+1.706+1.711],
    #     [0.054+0.054, 0.056+0.058, 0.060+0.063, 0.061+0.061, 0.073+0.074, 0.079+0.081, 0.086+0.090, 0.112+0.122, 0.170+0.165, 0.268+0.263, 0.461+0.465, 0.875+0.883, 1.772+1.779],
    #     [0.101, 0.107, 0.111, 0.122, 0.136, 0.149, 0.161, 0.199, 0.256, 0.380, 0.547, 0.979, 1.959]
    #     ]
    xs=[13664, 27329, 54659, 109319]
    labels = ("10 labels", "20 labels","40 labels","80 labels")
    ys_l=[[0.212+0.214+0.214+0.212+0.214+0.213+0.213+0.218, 0.408+0.409+0.411+0.411+0.409+0.409+0.410+0.409, 0.810+0.817+0.817+0.815+0.814+0.809+0.814+0.813, 1.675+1.686+1.681+1.677+1.672+1.679+1.679+1.678],
        [0.226+0.226+0.225+0.227, 0.425+0.424+0.423+0.425, 0.831+0.830+0.831+0.836, 1.711+1.709+1.706+1.711],
        [0.268+0.263, 0.461+0.465, 0.875+0.883, 1.772+1.779],
        [0.380, 0.547, 0.979, 1.959]
        ]
    for ys, label in zip(ys_l, labels):
        ax.scatter(xs, ys, label=label)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.tick_params(axis='both', which='minor', labelsize=18)
    ax.set_xlabel("Input Dimensions", fontsize=20)
    ax.set_ylabel("Inference Time(s)", fontsize=20)
    plt.legend(prop={'size': 16})
    # plt.show()
    plt.savefig(fig_path+filename+'.pdf', bbox_inches='tight')
    plt.close()

    fig_path = '/home/cc/Praxi-study/Praxi-Pipeline/prediction_XGBoost_openshift_image/model_testing_scripts/figs/'
    filename = "trainlatency_by_input_size_with_rawinput_data_0_estimated"
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    xs=[13, 106, 284, 427, 854, 1138, 1708, 3416, 6832, 13664, 27329, 54659, 109319]
    labels = ("Observed", "Estimated")
    ys_l=[[1.227, 2.012, 3.810, 5.275, 9.333, 12.124, 17.675, 34.274, 67.655, 136.472, 274.488, 546.612, 1096.709],
        #   [1.227, 1.227*2, 1.227*2**2, 1.227*2**3, 1.227*2**4, 1.227*2**5, 1.227*2**6, 1.227*2**7, 1.227*2**8, 1.227*2**9, 1.227*2**10, 1.227*2**11, 1.227*2**12],
        # [1.227, 1.227*2, 2.012*2, 3.810*2, 5.275*2, 9.333*2, 12.124*2, 17.675*2, 34.274*2, 67.655*2, 136.472*2, 274.488*2, 546.612*2]
        [1096.709/2**12, 1096.709/2**11, 1096.709/2**10, 1096.709/2**9, 1096.709/2**8, 1096.709/2**7, 1096.709/2**6, 1096.709/2**5, 1096.709/2**4, 1096.709/2**3, 1096.709/2**2, 1096.709/2, 1096.709]
        ]
    for ys, label in zip(ys_l, labels):
        ax.scatter(xs, ys, label=label)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.tick_params(axis='both', which='minor', labelsize=18)
    ax.set_xlabel("Input Dimensions", fontsize=20)
    ax.set_ylabel("Training Time(s)", fontsize=20)
    plt.legend(prop={'size': 16})
    # plt.show()
    plt.savefig(fig_path+filename+'.pdf', bbox_inches='tight')
    plt.close()

    # fig_path = '/home/cc/Praxi-study/Praxi-Pipeline/prediction_XGBoost_openshift_image/model_testing_scripts/figs/'
    # filename = "trainlatency_by_input_size_with_rawinput_data_0_estimated_sanitycheck"
    # fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    # xs=[13, 106, 284, 427, 854, 1138, 1708, 3416, 6832, 13664, 27329, 54659, 109319]
    # labels = ("Observed", "Estimated")
    # ys_l=[[1.227, 2.012, 3.810, 5.275, 9.333, 12.124, 17.675, 34.274, 67.655, 136.472, 274.488, 546.612, 1096.709],
    #       [1.227, 1.227*2, 1.227*2**2, 1.227*2**3, 1.227*2**4, 1.227*2**5, 1.227*2**6, 1.227*2**7, 1.227*2**8, 1.227*2**9, 1.227*2**10, 1.227*2**11, 1.227*2**12]
    #     ]
    # for ys, label in zip(ys_l, labels):
    #     ax.scatter(xs, ys, label=label)
    # ax.set_xlabel("Input Dimensions")
    # ax.set_ylabel("Training Time(s)")
    # plt.legend(prop={'size': 16})
    # # plt.show()
    # plt.savefig(fig_path+filename+'.pdf', bbox_inches='tight')
    # plt.close()





















    # by_labels_per_model

    fig_path = '/home/cc/Praxi-study/Praxi-Pipeline/prediction_XGBoost_openshift_image/model_testing_scripts/figs/'
    filename = "trainlatency_by_labels_per_model_with_rawinput_data_0"
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    xs=[10, 20, 40, 80]
    labels = ("13 dims", "106 dims", "284 dims", "427 dims", "854 dims", "1138 dims", "1708 dims", "3416 dims", "6832 dims", "13664 dims", "27329 dims", "54659 dims", "109319 dims")
    # ys_l=list(map(list, zip(*ys_l)))
    # print(ys_l)
    ys_l = [[0.102, 0.229, 0.315, 1.227], 
            [0.103, 0.29, 0.55, 2.012], 
            [0.143, 0.406, 1.027, 3.81], 
            [0.156, 0.451, 1.514, 5.275], 
            [0.226, 0.7, 2.58, 9.333], 
            [0.278, 0.928, 3.246, 12.124], 
            [0.392, 1.255, 4.596, 17.675], 
            [0.664, 2.366, 8.739, 34.274], 
            [1.205, 4.464, 17.492, 67.655], 
            [2.407, 9.328, 34.972, 136.472], 
            [5.449, 18.84, 70.854, 274.488], 
            [11.784, 39.945, 143.565, 546.612], 
            [24.566, 80.171, 286.711, 1096.709]
            ]
    for ys, label in zip(ys_l, labels):
        ax.scatter(xs, ys, label=label)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.tick_params(axis='both', which='minor', labelsize=18)
    ax.set_xlabel("Labels Per Model", fontsize=20)
    ax.set_ylabel("Training Time(s)", fontsize=20)
    plt.legend(prop={'size': 16})
    # plt.show()
    plt.savefig(fig_path+filename+'.pdf', bbox_inches='tight')
    plt.close()

    fig_path = '/home/cc/Praxi-study/Praxi-Pipeline/prediction_XGBoost_openshift_image/model_testing_scripts/figs/'
    filename = "inferencelatency_by_labels_per_model_with_rawinput_data_0"
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    # xs=[10, 20, 40, 80]
    # labels = ("13 dims", "106 dims", "284 dims", "427 dims", "854 dims", "1138 dims", "1708 dims", "3416 dims", "6832 dims", "13664 dims", "27329 dims", "54659 dims", "109319 dims")
    # # ys_l=list(map(list, zip(*ys_l)))
    # # print(ys_l)
    # ys_l = [[0.017+0.015+0.015+0.015+0.015+0.015+0.015+0.015, 0.030+0.031+0.031+0.030, 0.054+0.054, 0.101],
    #         [0.017+0.024+0.017+0.017+0.017+0.018+0.018+0.018, 0.031+0.032+0.031+0.032, 0.056+0.058, 0.107],
    #         [0.020+0.020+0.020+0.020+0.021+0.021+0.021+0.022, 0.034+0.034+0.034+0.034, 0.060+0.063, 0.111],
    #         [0.022+0.023+0.023+0.023+0.024+0.023+0.023+0.024, 0.036+0.037+0.037+0.037, 0.061+0.061, 0.122],
    #         [0.030+0.030+0.030+0.030+0.031+0.031+0.031+0.031, 0.045+0.044+0.045+0.045, 0.073+0.074, 0.136],
    #         [0.034+0.036+0.035+0.034+0.035+0.035+0.035+0.035, 0.047+0.047+0.048+0.050, 0.079+0.081, 0.149],
    #         [0.045+0.042+0.042+0.042+0.042+0.044+0.043+0.043, 0.055+0.059+0.056+0.056, 0.086+0.090, 0.161],
    #         [0.065+0.065+0.067+0.066+0.065+0.065+0.065+0.065, 0.079+0.079+0.079+0.081, 0.112+0.122, 0.199],
    #         [0.114+0.115+0.114+0.114+0.114+0.114+0.115+0.115, 0.128+0.132+0.129+0.131, 0.170+0.165, 0.256],
    #         [0.212+0.214+0.214+0.212+0.214+0.213+0.213+0.218, 0.226+0.226+0.225+0.227, 0.268+0.263, 0.380],
    #         [0.408+0.409+0.411+0.411+0.409+0.409+0.410+0.409, 0.425+0.424+0.423+0.425, 0.461+0.465, 0.547],
    #         [0.810+0.817+0.817+0.815+0.814+0.809+0.814+0.813, 0.831+0.830+0.831+0.836, 0.875+0.883, 0.979],
    #         [1.675+1.686+1.681+1.677+1.672+1.679+1.679+1.678, 1.711+1.709+1.706+1.711, 1.772+1.779, 1.959]
    #         ]
    xs=[10, 20, 40, 80]
    labels = ("13664 dims", "27329 dims", "54659 dims", "109319 dims")
    # ys_l=list(map(list, zip(*ys_l)))
    # print(ys_l)
    ys_l = [[0.212+0.214+0.214+0.212+0.214+0.213+0.213+0.218, 0.226+0.226+0.225+0.227, 0.268+0.263, 0.380],
            [0.408+0.409+0.411+0.411+0.409+0.409+0.410+0.409, 0.425+0.424+0.423+0.425, 0.461+0.465, 0.547],
            [0.810+0.817+0.817+0.815+0.814+0.809+0.814+0.813, 0.831+0.830+0.831+0.836, 0.875+0.883, 0.979],
            [1.675+1.686+1.681+1.677+1.672+1.679+1.679+1.678, 1.711+1.709+1.706+1.711, 1.772+1.779, 1.959]
            ]
    for ys, label in zip(ys_l, labels):
        ax.scatter(xs, ys, label=label)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.tick_params(axis='both', which='minor', labelsize=18)
    ax.set_xlabel("Labels Per Model", fontsize=20)
    ax.set_ylabel("Inference Time(s)", fontsize=20)
    plt.legend(prop={'size': 16})
    # plt.show()
    plt.savefig(fig_path+filename+'.pdf', bbox_inches='tight')
    plt.close()

    fig_path = '/home/cc/Praxi-study/Praxi-Pipeline/prediction_XGBoost_openshift_image/model_testing_scripts/figs/'
    filename = "testf1score_by_labels_per_model_with_rawinput_data_0"
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    xs=[10, 20, 40, 80]
    labels = ("13 dims", "106 dims", "284 dims", "427 dims", "854 dims", "1138 dims", "1708 dims", "3416 dims", "6832 dims", "13664 dims", "27329 dims", "54659 dims", "109319 dims")
    ys_l=[[0.259, 0.21, 0.166, 0.077], 
          [0.588, 0.633, 0.609, 0.521], 
          [0.714, 0.758, 0.746, 0.697], 
          [0.775, 0.763, 0.789, 0.727], 
          [0.741, 0.829, 0.857, 0.806], 
          [0.836, 0.841, 0.846, 0.843], 
          [0.851, 0.828, 0.891, 0.847], 
          [0.837, 0.886, 0.896, 0.906], 
          [0.815, 0.889, 0.911, 0.935], 
          [0.843, 0.884, 0.905, 0.934], 
          [0.842, 0.902, 0.911, 0.948], 
          [0.846, 0.891, 0.904, 0.951], 
          [0.843, 0.898, 0.916, 0.954]]
    # ys_l=list(map(list, zip(*ys_l)))
    # print(ys_l)
    for ys, label in zip(ys_l, labels):
        ax.scatter(xs, ys, label=label)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.tick_params(axis='both', which='minor', labelsize=18)
    ax.set_xlabel("Labels Per Model", fontsize=20)
    ax.set_ylabel("F1-Score", fontsize=20)
    plt.legend(prop={'size': 16})
    # plt.show()
    plt.savefig(fig_path+filename+'.pdf', bbox_inches='tight')
    plt.close()

    fig_path = '/home/cc/Praxi-study/Praxi-Pipeline/prediction_XGBoost_openshift_image/model_testing_scripts/figs/'
    filename = "testf1score_by_labels_per_model_with_rawinput_data_3"
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    xs=[10, 20, 25, 34, 50, 100, 500]
    labels = ("500labels-F1-with-filter", "500labels-F1-no-filter")
    # labels = ("500labels-F1-with-filter", "500labels-Precision-with-filter", "500labels-F1-no-filter", "500labels-Precision-no-filter")
    ys_l=[
        [0.884, 0.885, 0.885, 0.883, 0.884, 0.901, 0.954],
        # [0.834, 0.836, 0.837, 0.834, 0.834, 0.855, 0.926],
        [0.618, 0.641, 0.655, 0.679, 0.696, 0, 0],
        # [0.557, 0.58, 0.592, 0.617, 0.635, 0, 0]
    ]
    # ys_l=list(map(list, zip(*ys_l)))
    # print(ys_l)
    for ys, label in zip(ys_l, labels):
        ax.scatter(xs, ys, label=label)
    xs=[10, 20, 40, 80]
    labels = ("80labels-F1-with-filter", "80labels-F1-no-filter")
    # labels = ("80labels-F1-with-filter", "80labels-Precision-with-filter", "80labels-F1-no-filter", "80labels-Precision-no-filter")
    ys_l=[
        [0.883, 0.918, 0.957, 0.991],
        # [0.817, -1,-1,-1],
        [0.857, 0.903, 0.945, 0.998],
        # [0.843, 0.855, 0.868, 0.996]
    ]
    for ys, label in zip(ys_l, labels):
        ax.scatter(xs, ys, label=label)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.tick_params(axis='both', which='minor', labelsize=18)
    # ax.set_xscale('log')
    ax.set_xlabel("Labels Per Model", fontsize=20)
    ax.set_ylim(0,1)
    ax.set_ylabel("F1-Score", fontsize=20)
    plt.legend(prop={'size': 16})
    # plt.show()
    plt.savefig(fig_path+filename+'.pdf', bbox_inches='tight')
    plt.close()

    fig_path = '/home/cc/Praxi-study/Praxi-Pipeline/prediction_XGBoost_openshift_image/model_testing_scripts/figs/'
    filename = "trainlatency_by_labels_per_model_with_rawinput_data_0_estimated"
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    xs=[10, 20, 40, 80]
    labels = ("Observed", "Estimated", "Estimated_nosorting")
    # ys_l=list(map(list, zip(*ys_l)))
    # print(ys_l)
    ys_l = [[24.566, 80.171, 286.711, 1096.709],
            # [24.566,
            # 24.566*(4*math.log(25*20)/math.log(25*10)),
            # 80.171*(4*math.log(25*40)/math.log(25*20)), 
            # 286.711*(4*math.log(25*80)/math.log(25*40))]
            [1096.709/(4*math.log(25*80)/math.log(25*40))/(4*math.log(25*40)/math.log(25*20))/(4*math.log(25*20)/math.log(25*10)),
            1096.709/(4*math.log(25*80)/math.log(25*40))/(4*math.log(25*40)/math.log(25*20)), 
            1096.709/(4*math.log(25*80)/math.log(25*40)),
            1096.709],
            [1096.709/(4)/(4)/(4),
            1096.709/(4)/(4), 
            1096.709/(4),
            1096.709]
            ]
    for ys, label in zip(ys_l, labels):
        ax.scatter(xs, ys, label=label)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.tick_params(axis='both', which='minor', labelsize=18)
    ax.set_xlabel("Labels Per Model", fontsize=20)
    ax.set_ylabel("Training Time(s)", fontsize=20)
    plt.legend(prop={'size': 16})
    # plt.show()
    plt.savefig(fig_path+filename+'.pdf', bbox_inches='tight')
    plt.close()

    # fig_path = '/home/cc/Praxi-study/Praxi-Pipeline/prediction_XGBoost_openshift_image/model_testing_scripts/figs/'
    # filename = "trainlatency_by_labels_per_model_with_rawinput_data_3_estimated"
    # fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    # xs=[10, 20, 40, 80]
    # labels = ("Observed", "Estimated", "Estimated_nosorting")
    # # ys_l=list(map(list, zip(*ys_l)))
    # # print(ys_l)
    # ys_l = [[24.566, 80.171, 286.711, 1096.709],
    #         # [24.566,
    #         # 24.566*(4*math.log(25*20)/math.log(25*10)),
    #         # 80.171*(4*math.log(25*40)/math.log(25*20)), 
    #         # 286.711*(4*math.log(25*80)/math.log(25*40))]
    #         [1096.709/(4*math.log(25*80)/math.log(25*40))/(4*math.log(25*40)/math.log(25*20))/(4*math.log(25*20)/math.log(25*10)),
    #         1096.709/(4*math.log(25*80)/math.log(25*40))/(4*math.log(25*40)/math.log(25*20)), 
    #         1096.709/(4*math.log(25*80)/math.log(25*40)),
    #         1096.709],
    #         [1096.709/(4)/(4)/(4),
    #         1096.709/(4)/(4), 
    #         1096.709/(4),
    #         1096.709]
    #         ]
    # for ys, label in zip(ys_l, labels):
    #     ax.scatter(xs, ys, label=label)
    # ax.tick_params(axis='both', which='major', labelsize=20)
    # ax.tick_params(axis='both', which='minor', labelsize=18)
    # ax.set_xlabel("Labels Per Model", fontsize=20)
    # ax.set_ylabel("Training Time(s)", fontsize=20)
    # plt.legend(prop={'size': 16})
    # # plt.show()
    # plt.savefig(fig_path+filename+'.pdf', bbox_inches='tight')
    # plt.close()

    fig_path = '/home/cc/Praxi-study/Praxi-Pipeline/prediction_XGBoost_openshift_image/model_testing_scripts/figs/'
    filename = "trainlatency_by_labels_per_model_with_rawinput_data_0_estimated_1njob_no_sorting"
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    xs=[10, 20, 40, 80]
    labels = ("Observed", "Estimated")
    # ys_l=list(map(list, zip(*ys_l)))
    # print(ys_l)
    ys_l = [[148.500, 555.019, 2153.236, 8474.915],
            # [148.500,
            # 148.500*(4*(math.log(25*20)+1)/(math.log(25*10)+1)),
            # 555.019*(4*(math.log(25*40)+1)/(math.log(25*20)+1)), 
            # 2153.236*(4*(math.log(25*80)+1)/(math.log(25*40)+1))]
            [148.500,
            148.500*(4),
            555.019*(4), 
            2153.236*(4)]
            ]
    for ys, label in zip(ys_l, labels):
        ax.scatter(xs, ys, label=label)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.tick_params(axis='both', which='minor', labelsize=18)
    ax.set_xlabel("Labels Per Model", fontsize=20)
    ax.set_ylabel("Training Time(s)", fontsize=20)
    plt.legend(prop={'size': 16})
    # plt.show()
    plt.savefig(fig_path+filename+'.pdf', bbox_inches='tight')
    plt.close()

    fig_path = '/home/cc/Praxi-study/Praxi-Pipeline/prediction_XGBoost_openshift_image/model_testing_scripts/figs/'
    filename = "trainlatency_by_labels_per_model_with_rawinput_data_0_estimated_1njob_doublesample_nosorting"
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    xs=[10, 20, 40, 80]
    labels = ("Observed", "Estimated")
    # ys_l=list(map(list, zip(*ys_l)))
    # print(ys_l)
    ys_l = [[282.00, 1080.01, 4254.03, 17168.33],
            # [282.00,
            # 282.00*(4*(math.log(50*20)+1)/(math.log(50*10)+1)),
            # 1080.01*(4*(math.log(50*40)+1)/(math.log(50*20)+1)), 
            # 4254.03*(4*(math.log(50*80)+1)/(math.log(50*40)+1))]
            [282.00,
            282.00*(4*1),
            1080.01*(4*1), 
            4254.03*(4*1)]
            ]
    for ys, label in zip(ys_l, labels):
        ax.scatter(xs, ys, label=label)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.tick_params(axis='both', which='minor', labelsize=18)
    ax.set_xlabel("Labels Per Model", fontsize=20)
    ax.set_ylabel("Training Time(s)", fontsize=20)
    plt.legend(prop={'size': 16})
    # plt.show()
    plt.savefig(fig_path+filename+'.pdf', bbox_inches='tight')
    plt.close()

    fig_path = '/home/cc/Praxi-study/Praxi-Pipeline/prediction_XGBoost_openshift_image/model_testing_scripts/figs/'
    filename = "trainlatency_by_labels_per_model_with_randomint10000000_109319_data_0_estimated_1njob_doublesample"
    # filename = "trainlatency_by_labels_per_model_with_randomint10000000_109319_data_0_estimated_1njob_doublesample_nosorting"
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    xs=[10, 20, 40, 80]
    labels = ("Observed", "Estimated")
    # ys_l=list(map(list, zip(*ys_l)))
    # print(ys_l)
    ys_l = [[546.535, 2094.348, 8248.998, 33385.295],
            [546.535,
            546.535*(4*(math.log(50*20)+1)/(math.log(50*10)+1)),
            2094.348*(4*(math.log(50*40)+1)/(math.log(50*20)+1)), 
            8248.998*(4*(math.log(50*80)+1)/(math.log(50*40)+1))]
            # [546.535,
            # 546.535*(4*1),
            # 2094.348*(4*1), 
            # 8248.998*(4*1)]
            ]
    for ys, label in zip(ys_l, labels):
        ax.scatter(xs, ys, label=label)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.tick_params(axis='both', which='minor', labelsize=18)
    ax.set_xlabel("Labels Per Model", fontsize=20)
    ax.set_ylabel("Training Time(s)", fontsize=20)
    plt.legend(prop={'size': 16})
    # plt.show()
    plt.savefig(fig_path+filename+'.pdf', bbox_inches='tight')
    plt.close()

    fig_path = '/home/cc/Praxi-study/Praxi-Pipeline/prediction_XGBoost_openshift_image/model_testing_scripts/figs/'
    filename = "trainlatency_by_labels_per_model_with_rawinput_data_0_estimated_2njob_no_sorting"
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    xs=[10, 20, 40, 80]
    labels = ("Observed", "Estimated")
    # ys_l=list(map(list, zip(*ys_l)))
    # print(ys_l)
    ys_l = [[84.723, 288.412, 1085.795, 4272.973],
            # [84.723,
            # 84.723*(4*(math.log(25*20)+1)/(math.log(25*10)+1)),
            # 288.412*(4*(math.log(25*40)+1)/(math.log(25*20)+1)), 
            # 1085.795*(4*(math.log(25*80)+1)/(math.log(25*40)+1))]
            [84.723,
            84.723*(4),
            288.412*(4), 
            1085.795*(4)]
            ]
    for ys, label in zip(ys_l, labels):
        ax.scatter(xs, ys, label=label)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.tick_params(axis='both', which='minor', labelsize=18)
    ax.set_xlabel("Labels Per Model", fontsize=20)
    ax.set_ylabel("Training Time(s)", fontsize=20)
    plt.legend(prop={'size': 16})
    # plt.show()
    plt.savefig(fig_path+filename+'.pdf', bbox_inches='tight')
    plt.close()

    # fig_path = '/home/cc/Praxi-study/Praxi-Pipeline/prediction_XGBoost_openshift_image/model_testing_scripts/figs/'
    # filename = "trainlatency_by_labels_per_model_with_randomint10000000_109319_estimated_sanitycheck"
    # fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    # xs=[10, 20, 40, 80]
    # labels = ("Observed", "Estimated")
    # # ys_l=list(map(list, zip(*ys_l)))
    # # print(ys_l)
    # ys_l = [[6.733, 51.841, 397.018, 3195.553],
    #         [6.733,
    #          6.733*(4*math.log(25*20)/math.log(25*10)),
    #          6.733*(4*math.log(25*20)/math.log(25*10))*(4*math.log(25*40)/math.log(25*20)), 
    #          6.733*(4*math.log(25*20)/math.log(25*10))*(4*math.log(25*40)/math.log(25*20))*(4*math.log(25*80)/math.log(25*40))]
    #         ]
    # for ys, label in zip(ys_l, labels):
    #     ax.scatter(xs, ys, label=label)
    # ax.set_xlabel("Labels Per Model")
    # ax.set_ylabel("Training Time(s)")
    # plt.legend(prop={'size': 16})
    # # plt.show()
    # plt.savefig(fig_path+filename+'.pdf', bbox_inches='tight')
    # plt.close()