import matplotlib.pyplot as plt
import numpy as np
import math, copy
import scipy
# from pylab import *
from matplotlib import cm

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
    # # ######################Cross-Validated Plots##############################
    fig_path = '/home/cc/Praxi-study/Praxi-Pipeline/prediction_XGBoost_openshift_image/model_testing_scripts/figs/figs-1/'
    # filename = "trainlatency_by_input_size_with_rawinput_data_0"
    # fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    # xs=[6832, 27329, 109319]
    # labels = ("10 labels", "20 labels","40 labels","80 labels")
    # ys_l=[#models
    #     [#dims
    #         [#CV [test_sample_batch_idx: 4,1,2,0,3 [shuffle_idx: 0,1,2]]
    #             1.094, 1.086, 1.091,  1.107, 1.044, 1.058,  1.057, 1.111, 1.083,  1.072, 1.107, 1.075,  1.11, 1.104, 1.093
    #         ],
    #         [
    #             4.405, 4.321, 4.51,  4.572, 4.287, 4.187,  4.527, 4.353, 4.329,  4.497, 4.294, 4.314,  4.318, 4.291, 4.279
    #         ],
    #         [
    #             22.704, 20.515, 22.478,  22.788, 22.591, 21.906,  21.741, 21.696, 21.355,  21.591, 21.919, 21.791,  21.453, 21.298, 21.085
    #         ],
    #     ],    
    #     [
    #         [
    #             3.733, 3.762, 3.685,  3.841, 3.733, 3.71,  3.752, 3.726, 3.684,  3.714, 3.733, 3.702,  3.748, 3.722, 3.694
    #         ],
    #         [
    #             15.567, 15.439, 15.547,  15.632, 15.519, 15.301,  15.557, 15.48, 15.214,  15.579, 15.578, 15.316,  15.725, 15.489, 15.263
    #         ],
    #         [
    #             67.613, 67.038, 65.915,  67.647, 66.098, 66.157,  67.923, 67.973, 66.999,  66.89, 67.01, 66.24,  66.579, 65.433, 65.681
    #         ],
    #     ],
    #     [
    #         [
    #             14.257, 14.327, 14.17,  14.307, 14.42, 14.428,  14.395, 14.452, 14.417,  14.212, 14.255, 14.241,  14.253, 14.357, 14.194
    #         ],
    #         [
    #             58.057, 56.905, 57.034,  57.689, 57.717, 57.719,  58.633, 57.526, 57.535,  57.4, 57.786, 57.024,  57.53, 57.741, 57.365
    #         ],
    #         [
    #             231.224, 230.094, 232.82,  231.556, 233.015, 230.776,  231.963, 234.409, 232.444,  230.481, 235.052, 230.225,  231.681, 231.671, 230.629
    #         ],
    #     ],
    #     [
    #         [
    #             53.933, 53.914, 53.883,  54.448, 54.299, 54.273,  54.158, 54.129, 54.086,  53.984, 54.109, 54.198,  54.035, 54.31, 54.073
    #         ],
    #         [
    #             219.871, 219.945, 219.656,  221.598, 222.077, 220.707,  220.384, 220.636, 220.454,  221.146, 221.284, 221.921,  220.838, 221.206, 219.983
    #         ],
    #         [
    #             872.889, 864.231, 865.572,  874.202, 887.441, 871.679,  846.84, 874.245, 876.482,  873.451, 872.806, 873.382,  871.592, 871.109, 872.448
    #         ],
    #     ]
    # ]
    # ys_array = np.mean(ys_l, axis=(2))
    # ci_array = 1.96 * np.std(ys_l, axis=2)/np.sqrt(len(ys_l[0][0]))
    # for ys, ci, label in zip(ys_array, ci_array, labels):
    #     ax.scatter(xs, ys, label=label)
    #     ax.errorbar(xs, ys, yerr=ci, fmt='o', capsize=10)
    # ax.tick_params(axis='both', which='major', labelsize=20)
    # ax.tick_params(axis='both', which='minor', labelsize=18)
    # ax.set_xlabel("Input Dimensions", fontsize=20)
    # ax.set_ylabel("Training Time(s)", fontsize=20)
    # ax.grid()
    # plt.legend(prop={'size': 16})
    # # plt.show()
    # plt.savefig(fig_path+filename+'.pdf', bbox_inches='tight')
    # plt.close()




    # # fig_path = '/home/cc/Praxi-study/Praxi-Pipeline/prediction_XGBoost_openshift_image/model_testing_scripts/figs/figs-1/'
    # filename = "trainlatency_by_input_size_with_rawinput_data_0_estimated"
    # fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    # xs=[6832, 27329, 109319]
    # labels = ("Observed", "Estimated")
    # ys_l=[#models
    #     [#dims
    #         [
    #             53.933, 53.914, 53.883,  54.448, 54.299, 54.273,  54.158, 54.129, 54.086,  53.984, 54.109, 54.198,  54.035, 54.31, 54.073
    #         ],
    #         [
    #             219.871, 219.945, 219.656,  221.598, 222.077, 220.707,  220.384, 220.636, 220.454,  221.146, 221.284, 221.921,  220.838, 221.206, 219.983
    #         ],
    #         [
    #             872.889, 864.231, 865.572,  874.202, 887.441, 871.679,  846.84, 874.245, 876.482,  873.451, 872.806, 873.382,  871.592, 871.109, 872.448
    #         ],
    #     ]
    # ]
    # ysestimate_l = [[871.2246/16,871.2246/4,871.2246]]
    # ys_array = np.mean(ys_l, axis=(2))
    # ci_array = 1.96 * np.std(ys_l, axis=2)/np.sqrt(len(ys_l[0][0]))
    # for ys, ci, label in zip(ys_array, ci_array, [labels[0]]):
    #     ax.scatter(xs, ys, label=label)
    #     ax.errorbar(xs, ys, yerr=ci, fmt='o', capsize=10)
    # for ys, label in zip(ysestimate_l, [labels[1]]):
    #     ax.scatter(xs, ys, label=label)
    #     # ax.errorbar(xs, ys, yerr=ci, fmt='o', capsize=10)
    # ax.tick_params(axis='both', which='major', labelsize=20)
    # ax.tick_params(axis='both', which='minor', labelsize=18)
    # ax.set_xlabel("Input Dimensions", fontsize=20)
    # ax.set_ylabel("Training Time(s)", fontsize=20)
    # ax.grid()
    # plt.legend(prop={'size': 16})
    # # plt.show()
    # plt.savefig(fig_path+filename+'.pdf', bbox_inches='tight')
    # plt.close()


    # fig_path = '/home/cc/Praxi-study/Praxi-Pipeline/prediction_XGBoost_openshift_image/model_testing_scripts/figs/'
    filename = "testf1score_by_input_size_with_rawinput_data_0"
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    xs=[6832, 27329, 109319]
    labels = ("10 labels", "20 labels","40 labels","80 labels")
    ys_l=[#models
        [#dims
            [#CV [test_sample_batch_idx: 4,1,2,0,3 [shuffle_idx: 0,1,2]]
                0.825, 0.81, 0.836, 0.849, 0.866, 0.88, 0.852, 0.88, 0.836, 0.914, 0.863, 0.814, 0.855, 0.856, 0.821
            ],
            [
                0.845, 0.807, 0.843, 0.867, 0.869, 0.877, 0.862, 0.874, 0.864, 0.902, 0.863, 0.809, 0.865, 0.859, 0.827
            ],
            [
                0.847, 0.82, 0.849, 0.867, 0.867, 0.87, 0.867, 0.893, 0.872, 0.902, 0.861, 0.816, 0.872, 0.858, 0.821
            ],
        ],    
        [
            [
                0.886, 0.92, 0.865, 0.877, 0.871, 0.877, 0.882, 0.891, 0.905, 0.845, 0.893, 0.85, 0.876, 0.888, 0.876
            ],
            [
                0.89, 0.892, 0.903, 0.877, 0.89, 0.902, 0.905, 0.889, 0.891, 0.879, 0.879, 0.901, 0.885, 0.907, 0.869
            ],
            [
                0.891, 0.897, 0.905, 0.893, 0.875, 0.888, 0.91, 0.881, 0.891, 0.889, 0.903, 0.909, 0.903, 0.894, 0.877
            ],
        ],
        [
            [
                0.897, 0.87, 0.896, 0.896, 0.905, 0.914, 0.932, 0.949, 0.913, 0.92, 0.915, 0.897, 0.896, 0.89, 0.903
            ],
            [
                0.905, 0.92, 0.924, 0.904, 0.905, 0.932, 0.934, 0.923, 0.956, 0.94, 0.925, 0.911, 0.926, 0.914, 0.923
            ],
            [
                0.915, 0.933, 0.916, 0.925, 0.912, 0.932, 0.932, 0.925, 0.944, 0.937, 0.942, 0.93, 0.918, 0.926, 0.907
            ],
        ],
        [
            [
                0.9, 0.908, 0.913, 0.91, 0.909, 0.9, 0.908, 0.913, 0.91, 0.909, 0.9, 0.908, 0.913, 0.91, 0.909
            ],
            [
                0.94, 0.937, 0.94, 0.942, 0.946, 0.94, 0.937, 0.94, 0.942, 0.946, 0.94, 0.937, 0.94, 0.942, 0.946
            ],
            [
                0.951, 0.952, 0.956, 0.955, 0.955, 0.951, 0.952, 0.956, 0.955, 0.955, 0.951, 0.952, 0.956, 0.955, 0.955
            ],
        ]
    ]
    ys_array = np.mean(ys_l, axis=(2))
    ci_array = 1.96 * np.std(ys_l, axis=2)/np.sqrt(len(ys_l[0][0]))
    for ys, ci, label in zip(ys_array, ci_array, labels):
        ax.scatter(xs, ys, label=label)
        ax.errorbar(xs, ys, yerr=ci, fmt='o', capsize=10)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.tick_params(axis='both', which='minor', labelsize=18)
    ax.set_xlabel("Input Dimensions", fontsize=20)
    ax.set_ylabel("F1-Scores", fontsize=20)
    plt.legend(prop={'size': 16})
    # plt.show()
    plt.savefig(fig_path+filename+'.pdf', bbox_inches='tight')
    plt.close()


    print(0)
    # ###########################################################################




























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



    # fig_path = '/home/cc/Praxi-study/Praxi-Pipeline/prediction_XGBoost_openshift_image/model_testing_scripts/figs/'
    # filename = "trainlatency_by_N_models_with_rawinput_data_0"
    # labels = (
    #     "8","4","2","1"
    # )
    # cate_values_1 = {
    #     "M10-M20-M40-M80": np.array([2.407, 18.840, 143.565, 1090.042]),
    #     "M10-M20-M40-MX_1": np.array([2.396, 18.788, 143.717, 0]),
    #     "M10-M20-M40-MX_2": np.array([2.413, 19.016, 0, 0]),
    #     "M10-M20-M40-MX_3": np.array([2.407, 18.862, 0, 0]),
    #     "M10-MX-MX-MX_1": np.array([2.406, 0, 0, 0]),
    #     "M10-MX-MX-MX_2": np.array([2.396, 0, 0, 0]),
    #     "M10-MX-MX-MX_3": np.array([2.440, 0, 0, 0]),
    #     "M10-MX-MX-MX_4": np.array([2.396, 0, 0, 0]),
    # }
    # cate_values_2 = {
    #     "Estimated": np.array([8*2.4, 4*2.4*2*(4*math.log(25*20)/math.log(25*10)), 2*19.0*2*(4*math.log(25*40)/math.log(25*20)), 143.5*2*(4*math.log(25*80)/math.log(25*40))])
    # }
    # cate_values = [cate_values_1, cate_values_2]
    # plotting(fig_path, filename, cate_values, labels, xaxis_label="Number of Models", yaxis_label="Training Time(s)", title="Train Latency by N Models with the Same Train Dataset")

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









    # fig_path = '/home/cc/Praxi-study/Praxi-Pipeline/prediction_XGBoost_openshift_image/model_testing_scripts/figs/'
    # filename = "trainlatency_by_input_size_and_by_N_models_with_data_0_3d"
    # # Fixing random state for reproducibility
    # np.random.seed(19680801)

    # fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')

    # xs=[13, 106, 284, 427, 854, 1138, 1708, 3416, 6832, 13664, 27329, 54659, 109319, 
    #     13, 106, 284, 427, 854, 1138, 1708, 3416, 6832, 13664, 27329, 54659, 109319, 
    #     13, 106, 284, 427, 854, 1138, 1708, 3416, 6832, 13664, 27329, 54659, 109319, 
    #     13, 106, 284, 427, 854, 1138, 1708, 3416, 6832, 13664, 27329, 54659, 109319]   # input sizes
    # ys=[80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 
    #     40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 
    #     20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 
    #     10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10]   # number of labels
    # zs=[1.227, 2.012, 3.810, 5.275, 9.333, 12.124, 17.675, 34.274, 67.655, 136.472, 274.488, 546.612, 1096.709, 
    #     0.315, 0.550, 1.027, 1.514, 2.580, 3.246, 4.596, 8.739, 17.492, 34.972, 70.854, 143.565, 286.711, 
    #     0.229, 0.290, 0.406, 0.451, 0.700, 0.928, 1.255, 2.366, 4.464, 9.328, 18.840, 39.945, 80.171, 
    #     0.102, 0.103, 0.143, 0.156, 0.226, 0.278, 0.392, 0.664, 1.205, 2.407, 5.449, 11.784, 24.566]   # training latency
    # # xs=[13, 106, 854, 13, 106, 854, 13, 106, 854, 13, 106, 854]   # input sizes
    # # ys=[80, 80, 80, 40, 40, 40, 20, 20, 20, 10, 10, 10]   # number of labels
    # # zs=[1.227, 2.012, 9.333, 0.315, 0.550, 2.580, 0.229, 0.290, 0.700, 0.102, 0.103, 0.226]   # training latency

    # ax.scatter(xs, ys, zs, marker='o')

    # ax.set_xlabel('Feature Dimensions')
    # ax.set_ylabel('Labels Per Model')
    # ax.set_zlabel('Training Latency')

    # # plt.show()
    # plt.savefig(fig_path+filename+'.pdf', bbox_inches='tight')
    # plt.close()




    # fig_path = '/home/cc/Praxi-study/Praxi-Pipeline/prediction_XGBoost_openshift_image/model_testing_scripts/figs/'
    # filename = "trainlatency_by_inputsize_and_by_samplesize_with_data_3_25_3d"

    # fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')

    # samplesize=[357.0, 419.0, 377.0, 357.0, 356.0, 336.0, 360.0, 339.0, 297.0, 338.0, 358.0, 378.0, 393.0, 397.0, 337.0, 375.0, 353.0, 374.0, 294.0, 355.0, 339.0, 330.0, 375.0, 375.0, 317.0, 334.0, 355.0, 335.0, 333.0, 396.0, 378.0, 349.0, 275.0, 395.0, 339.0, 318.0, 399.0, 299.0, 355.0, 357.0, 359.0, 336.0, 337.0, 378.0, 334.0, 334.0, 380.0, 355.0, 337.0, 375.0, 379.0, 357.0, 328.0, 396.0, 317.0, 331.0, 392.0, 358.0, 359.0, 378.0, 356.0, 338.0, 380.0, 399.0, 360.0, 299.0, 374.0, 359.0, 412.0, 278.0, 259.0, 259.0, 357.0, 357.0, 351.0, 395.0, 418.0, 335.0, 398.0, 357.0, 377.0, 313.0, 353.0, 391.0, 333.0, 375.0, 351.0, 316.0, 358.0, 334.0, 337.0, 353.0, 357.0, 318.0, 340.0, 320.0, 372.0, 298.0, 319.0, 418.0, 295.0, 397.0, 400.0, 359.0, 377.0, 395.0, 311.0, 337.0, 357.0, 339.0, 338.0, 319.0, 377.0, 379.0, 280.0, 398.0, 336.0, 332.0, 355.0, 318.0, 375.0, 376.0, 359.0, 336.0, 397.0, 394.0, 378.0, 358.0, 299.0, 337.0, 335.0, 359.0, 375.0, 397.0, 336.0, 339.0, 372.0, 398.0, 356.0, 275.0, 371.0, 296.0, 279.0, 299.0, 336.0, 375.0, 353.0, 336.0, 355.0, 297.0, 335.0, 356.0, 355.0, 353.0, 315.0, 399.0, 276.0, 339.0, 338.0, 359.0, 396.0, 300.0, 355.0, 351.0, 371.0, 337.0, 400.0, 377.0, 396.0, 377.0, 313.0, 330.0, 319.0, 355.0, 314.0, 393.0, 316.0, 378.0, 394.0, 339.0, 400.0, 400.0, 297.0, 340.0, 256.0, 375.0, 357.0, 356.0, 395.0, 398.0, 357.0, 372.0, 336.0, 335.0, 397.0, 358.0, 359.0, 360.0, 359.0, 371.0, 416.0, 375.0, 359.0, 335.0, 358.0, 358.0, 340.0, 336.0, 353.0, 354.0, 376.0, 299.0, 360.0, 334.0, 259.0, 318.0, 314.0, 380.0, 320.0, 377.0, 351.0, 340.0, 377.0, 337.0, 358.0, 319.0, 379.0, 370.0, 339.0, 393.0, 314.0, 378.0, 356.0, 338.0, 355.0, 299.0, 359.0, 392.0, 373.0, 359.0, 337.0, 358.0, 358.0, 298.0, 397.0, 331.0, 399.0, 354.0, 395.0, 376.0, 357.0, 360.0, 395.0, 359.0, 355.0, 357.0, 315.0, 298.0, 359.0, 319.0, 395.0, 355.0, 357.0, 299.0, 337.0, 295.0, 377.0, 355.0, 414.0, 350.0, 394.0, 332.0, 378.0, 316.0, 355.0, 376.0, 375.0, 378.0, 352.0, 315.0, 316.0, 393.0, 354.0, 339.0, 358.0, 259.0, 319.0, 299.0, 353.0, 360.0, 336.0, 319.0, 357.0, 340.0, 337.0, 319.0, 339.0, 316.0, 394.0, 355.0, 360.0, 379.0, 395.0, 348.0, 396.0, 357.0, 331.0, 376.0, 367.0, 338.0, 336.0, 376.0, 360.0, 357.0, 317.0, 338.0, 376.0, 296.0, 394.0, 333.0, 320.0, 313.0, 336.0, 376.0, 354.0, 378.0, 356.0, 360.0, 356.0, 377.0, 420.0, 391.0, 338.0, 356.0, 319.0, 376.0, 415.0, 339.0, 376.0, 337.0, 300.0, 400.0, 335.0, 396.0, 359.0, 319.0, 319.0, 298.0, 396.0, 310.0, 392.0, 380.0, 375.0, 336.0, 357.0, 355.0, 359.0, 296.0, 275.0, 291.0]
    # dimensions=[10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 5000.0, 5000.0, 5000.0, 5000.0, 5000.0, 5000.0, 5000.0, 5000.0, 5000.0, 5000.0, 5000.0, 5000.0, 5000.0, 5000.0, 5000.0, 5000.0, 5000.0, 5000.0, 5000.0, 5000.0, 5000.0, 5000.0, 5000.0, 5000.0, 5000.0, 5000.0, 5000.0, 5000.0, 5000.0, 5000.0, 5000.0, 5000.0, 5000.0, 5000.0, 5000.0, 5000.0, 5000.0, 5000.0, 5000.0, 5000.0, 5000.0, 5000.0, 5000.0, 5000.0, 5000.0, 5000.0, 5000.0, 5000.0, 5000.0, 5000.0, 5000.0, 5000.0, 5000.0, 5000.0, 5000.0, 5000.0, 5000.0, 5000.0, 5000.0, 5000.0, 5000.0, 5000.0, 5000.0, 5000.0, 5000.0, 5000.0, 5000.0, 5000.0, 5000.0, 5000.0, 5000.0, 5000.0, 15000.0, 15000.0, 15000.0, 15000.0, 15000.0, 15000.0, 15000.0, 15000.0, 15000.0, 15000.0, 15000.0, 15000.0, 15000.0, 15000.0, 15000.0, 15000.0, 15000.0, 15000.0, 15000.0, 15000.0, 15000.0, 15000.0, 15000.0, 15000.0, 15000.0, 15000.0, 15000.0, 15000.0, 15000.0, 15000.0, 15000.0, 15000.0, 15000.0, 15000.0, 15000.0, 15000.0, 15000.0, 15000.0, 15000.0, 15000.0, 15000.0, 15000.0, 15000.0, 15000.0, 15000.0, 15000.0, 15000.0, 15000.0, 15000.0, 15000.0, 15000.0, 15000.0, 15000.0, 15000.0, 15000.0, 15000.0, 15000.0, 15000.0, 15000.0, 15000.0, 15000.0, 15000.0, 15000.0, 15000.0, 15000.0, 15000.0, 15000.0, 15000.0, 15000.0, 15000.0, 15000.0, 15000.0]
    # traintimes=[1.548, 1.972, 1.678, 1.549, 1.506, 1.4, 1.53, 1.524, 1.134, 1.439, 1.493, 1.616, 1.877, 1.758, 1.373, 1.693, 1.508, 1.604, 1.137, 1.537, 1.426, 1.388, 1.682, 1.66, 1.287, 1.377, 1.376, 1.378, 1.366, 1.675, 1.674, 1.501, 0.944, 1.82, 1.402, 1.184, 1.811, 1.143, 1.468, 1.494, 1.532, 1.363, 1.395, 1.732, 1.452, 1.377, 1.681, 1.506, 1.432, 1.783, 1.697, 1.57, 1.383, 1.852, 1.269, 1.378, 1.771, 1.502, 1.536, 1.673, 1.527, 1.397, 1.689, 1.895, 1.529, 1.178, 1.715, 1.525, 1.945, 1.027, 0.927, 0.907, 0.524, 0.414, 0.458, 0.572, 0.513, 0.433, 0.582, 0.466, 0.482, 0.443, 0.426, 0.532, 0.463, 0.677, 0.457, 0.342, 0.471, 0.431, 0.459, 0.437, 0.481, 0.315, 0.518, 0.437, 0.498, 0.374, 0.431, 0.539, 0.413, 0.52, 0.552, 0.463, 0.483, 0.641, 0.402, 0.473, 0.417, 0.456, 0.59, 0.371, 0.474, 0.505, 0.366, 0.457, 0.608, 0.647, 0.429, 0.604, 0.745, 0.482, 0.507, 0.495, 0.566, 0.476, 0.502, 0.491, 0.507, 0.463, 0.478, 0.483, 0.525, 0.552, 0.385, 0.465, 0.493, 0.503, 0.496, 0.319, 0.533, 0.309, 0.386, 0.43, 0.388, 0.419, 0.41, 0.375, 0.367, 0.248, 0.341, 0.353, 0.279, 0.36, 0.332, 0.325, 0.287, 0.355, 0.254, 0.364, 0.443, 0.223, 0.369, 0.395, 0.444, 0.327, 0.41, 0.364, 0.419, 0.39, 0.36, 0.336, 0.349, 0.412, 0.326, 0.426, 0.386, 0.376, 0.416, 0.372, 0.351, 0.433, 0.316, 0.295, 0.263, 0.41, 0.295, 0.378, 0.401, 0.378, 0.352, 0.459, 0.33, 0.36, 0.44, 0.368, 0.392, 0.368, 0.371, 0.429, 0.439, 0.402, 0.405, 0.368, 0.373, 0.415, 0.393, 0.347, 0.381, 0.395, 0.418, 0.317, 0.398, 0.383, 0.283, 0.382, 0.817, 1.041, 0.804, 0.986, 0.93, 0.882, 0.999, 0.869, 0.955, 0.811, 1.022, 1.016, 0.815, 1.124, 0.773, 1.009, 0.96, 0.868, 0.927, 0.736, 0.946, 1.084, 1.014, 0.95, 0.839, 0.937, 0.933, 0.72, 1.096, 0.818, 1.11, 0.938, 1.059, 1.018, 0.937, 0.935, 1.066, 0.962, 0.94, 0.938, 0.817, 0.733, 0.938, 0.795, 1.13, 0.947, 0.938, 0.713, 0.849, 0.733, 1.012, 0.927, 1.196, 0.959, 1.086, 0.874, 1.018, 0.8, 0.948, 1.064, 1.012, 1.031, 0.955, 0.785, 0.823, 1.117, 0.95, 0.865, 0.951, 0.611, 0.808, 0.721, 2.067, 2.165, 2.045, 1.779, 2.155, 2.015, 2.009, 1.809, 1.878, 1.81, 2.49, 2.053, 2.102, 2.298, 2.488, 2.114, 2.432, 2.055, 2.033, 2.26, 2.266, 2.04, 1.947, 2.254, 2.177, 2.068, 1.752, 1.93, 2.272, 1.633, 2.561, 1.967, 1.706, 1.798, 1.99, 2.268, 2.084, 2.286, 2.081, 2.239, 2.183, 2.288, 2.754, 2.578, 1.886, 2.121, 1.74, 2.251, 2.749, 2.034, 2.298, 1.971, 1.607, 2.479, 1.97, 2.514, 2.083, 1.741, 1.734, 1.615, 2.539, 1.709, 2.469, 2.317, 2.256, 1.859, 2.158, 2.065, 2.091, 1.576, 1.409, 1.549]
    
    # colo = traintimes 
    # color_map = cm.ScalarMappable(cmap=cm.summer) 
    # color_map.set_array(colo) 
    # colo_normalized = [c/max(colo) for c in colo]
    # # ax.plot_trisurf(samplesize, dimensions, traintimes, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    # ax.scatter(samplesize, dimensions, traintimes, facecolors=cm.summer(colo_normalized), edgecolor=cm.summer(colo_normalized), alpha=1)
    # # ax.scatter(samplesize, dimensions, facecolors=cm.PiYG(colo_normalized), edgecolor=cm.PiYG(colo_normalized), alpha=1)
    # # plt.colorbar(color_map,label='Training Latency (s)') 

    # ax.set_xlabel('Sample Size')
    # ax.set_ylabel('Feature Dimension Size')
    # ax.set_zlabel('Training Latency')
    # # ax.zaxis.labelpad = -4
    # ax.view_init(azim=270, elev=0)

    # # plt.show()
    # # fig.set_size_inches(6, 8)
    # fig.tight_layout()
    # # fig.subplots_adjust(left=-2) 
    # plt.savefig(fig_path+filename+'.pdf', bbox_inches='tight')
    # plt.close()






    # fig_path = '/home/cc/Praxi-study/Praxi-Pipeline/prediction_XGBoost_openshift_image/model_testing_scripts/figs/'
    # filename = "trainlatency_by_inputsize_and_by_samplesize_with_data_3_10_3d"

    # fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')

    # samplesize=[856.0, 872.0, 896.0, 815.0, 806.0, 916.0, 849.0, 923.0, 895.0, 851.0, 773.0, 845.0, 974.0, 917.0, 851.0, 864.0, 893.0, 868.0, 874.0, 914.0, 794.0, 910.0, 770.0, 871.0, 750.0, 909.0, 862.0, 689.0, 655.0, 634.0, 892.0, 884.0, 794.0, 763.0, 789.0, 927.0, 754.0, 894.0, 886.0, 977.0, 927.0, 850.0, 904.0, 773.0, 835.0, 873.0, 836.0, 871.0, 814.0, 915.0, 773.0, 814.0, 811.0, 910.0, 908.0, 888.0, 912.0, 733.0, 715.0, 674.0, 930.0, 910.0, 874.0, 893.0, 917.0, 847.0, 833.0, 836.0, 929.0, 850.0, 851.0, 891.0, 779.0, 889.0, 807.0, 791.0, 828.0, 835.0, 850.0, 890.0, 933.0, 906.0, 767.0, 873.0, 863.0, 829.0, 749.0, 737.0, 715.0, 694.0, 872.0, 869.0, 851.0, 786.0, 807.0, 828.0, 897.0, 883.0, 914.0, 863.0, 810.0, 830.0, 883.0, 936.0, 808.0, 813.0, 912.0, 830.0, 869.0, 797.0, 936.0, 896.0, 850.0, 810.0, 859.0, 854.0, 892.0, 694.0, 714.0, 733.0, 856.0, 801.0, 855.0, 814.0, 832.0, 829.0, 846.0, 817.0, 933.0, 823.0, 891.0, 777.0, 910.0, 868.0, 890.0, 896.0, 839.0, 908.0, 856.0, 870.0, 767.0, 824.0, 949.0, 889.0, 856.0, 830.0, 893.0, 751.0, 735.0, 691.0]
    # dimensions=[10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 5000.0, 5000.0, 5000.0, 5000.0, 5000.0, 5000.0, 5000.0, 5000.0, 5000.0, 5000.0, 5000.0, 5000.0, 5000.0, 5000.0, 5000.0, 5000.0, 5000.0, 5000.0, 5000.0, 5000.0, 5000.0, 5000.0, 5000.0, 5000.0, 5000.0, 5000.0, 5000.0, 5000.0, 5000.0, 5000.0, 15000.0, 15000.0, 15000.0, 15000.0, 15000.0, 15000.0, 15000.0, 15000.0, 15000.0, 15000.0, 15000.0, 15000.0, 15000.0, 15000.0, 15000.0, 15000.0, 15000.0, 15000.0, 15000.0, 15000.0, 15000.0, 15000.0, 15000.0, 15000.0, 15000.0, 15000.0, 15000.0, 15000.0, 15000.0, 15000.0]
    # traintimes=[6.934, 7.13, 7.448, 6.346, 6.214, 7.82, 6.868, 7.988, 7.506, 6.899, 5.759, 6.852, 8.728, 7.82, 6.87, 7.046, 7.462, 7.118, 7.131, 7.794, 6.013, 7.739, 5.673, 7.187, 5.464, 7.775, 7.105, 4.659, 4.217, 3.96, 1.521, 2.004, 1.224, 1.27, 1.189, 1.542, 1.645, 1.961, 1.451, 1.801, 1.53, 2.028, 1.56, 1.175, 1.335, 1.447, 1.267, 1.423, 1.277, 1.477, 1.237, 1.277, 1.63, 2.064, 1.485, 1.467, 1.533, 1.119, 1.081, 0.996, 1.153, 1.545, 1.075, 1.082, 1.183, 1.029, 0.955, 0.977, 1.205, 1.024, 1.012, 1.09, 0.904, 1.09, 0.992, 1.356, 0.989, 1.018, 0.989, 1.109, 1.207, 1.137, 0.887, 1.086, 1.055, 1.012, 0.945, 0.968, 0.865, 1.056, 4.196, 4.138, 3.913, 3.469, 3.61, 3.689, 4.418, 4.229, 4.42, 4.087, 3.536, 3.654, 4.267, 4.571, 3.493, 3.631, 4.295, 3.689, 4.16, 3.438, 4.531, 4.401, 3.872, 3.51, 3.968, 3.897, 4.2, 2.776, 2.812, 2.971, 10.203, 9.079, 10.084, 9.181, 9.764, 9.593, 10.081, 9.308, 12.051, 9.63, 11.009, 8.458, 11.548, 10.519, 10.946, 11.1, 9.797, 11.51, 10.125, 10.545, 8.346, 9.56, 12.463, 11.048, 10.171, 9.644, 11.024, 7.95, 7.699, 6.843]
    
    # colo = traintimes 
    # color_map = cm.ScalarMappable(cmap=cm.summer) 
    # color_map.set_array(colo) 
    # colo_normalized = [c/max(colo) for c in colo]
    # # ax.plot_trisurf(samplesize, dimensions, traintimes, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    # ax.scatter(samplesize, dimensions, traintimes, facecolors=cm.summer(colo_normalized), edgecolor=cm.summer(colo_normalized), alpha=1)
    # # ax.scatter(samplesize, dimensions, facecolors=cm.PiYG(colo_normalized), edgecolor=cm.PiYG(colo_normalized), alpha=1)
    # # plt.colorbar(color_map,label='Training Latency (s)') 

    # ax.set_xlabel('Sample Size')
    # ax.set_ylabel('Feature Dimension Size')
    # ax.set_zlabel('Training Latency')
    # # ax.zaxis.labelpad = -4
    # ax.view_init(azim=180, elev=0)

    # # plt.show()
    # # fig.set_size_inches(6, 8)
    # fig.tight_layout()
    # # fig.subplots_adjust(left=-2) 
    # plt.savefig(fig_path+filename+'.pdf', bbox_inches='tight')
    # plt.close()





    # fig_path = '/home/cc/Praxi-study/Praxi-Pipeline/prediction_XGBoost_openshift_image/model_testing_scripts/figs/'
    # filename = "trainf1score_by_inputsize_and_by_samplesize_with_data_3_heat"

    # fig = plt.figure()
    # ax = fig.add_subplot()

    # samplesize=[
    #     336.0, 333.0, 371.0, 373.0, 256.0, 379.0, 356.0, 379.0, 357.0, 353.0, 297.0, 358.0, 378.0, 336.0, 358.0, 339.0, 376.0, 378.0, 359.0, 335.0, 377.0, 372.0, 374.0, 399.0, 313.0, 379.0, 318.0, 334.0, 394.0, 374.0, 333.0, 379.0, 275.0, 374.0, 373.0, 349.0, 338.0, 378.0, 396.0, 378.0, 318.0, 375.0, 379.0, 353.0, 375.0, 398.0, 355.0, 255.0, 319.0, 335.0, 337.0, 360.0, 356.0, 380.0, 336.0, 377.0, 334.0, 336.0, 359.0, 360.0, 379.0, 360.0, 359.0, 336.0, 360.0, 372.0, 357.0, 376.0, 339.0, 296.0, 294.0, 257.0, 357.0, 419.0, 377.0, 357.0, 356.0, 336.0, 360.0, 339.0, 297.0, 338.0, 358.0, 378.0, 393.0, 397.0, 337.0, 375.0, 353.0, 374.0, 294.0, 355.0, 339.0, 330.0, 375.0, 375.0, 317.0, 334.0, 355.0, 335.0, 333.0, 396.0, 378.0, 349.0, 275.0, 395.0, 339.0, 318.0, 399.0, 299.0, 355.0, 357.0, 359.0, 336.0, 337.0, 378.0, 334.0, 334.0, 380.0, 355.0, 337.0, 375.0, 379.0, 357.0, 328.0, 396.0, 317.0, 331.0, 392.0, 358.0, 359.0, 378.0, 356.0, 338.0, 380.0, 399.0, 360.0, 299.0, 374.0, 359.0, 412.0, 278.0, 259.0, 259.0, 357.0, 357.0, 351.0, 395.0, 418.0, 335.0, 398.0, 357.0, 377.0, 313.0, 353.0, 391.0, 333.0, 375.0, 351.0, 316.0, 358.0, 334.0, 337.0, 353.0, 357.0, 318.0, 340.0, 320.0, 372.0, 298.0, 319.0, 418.0, 295.0, 397.0, 400.0, 359.0, 377.0, 395.0, 311.0, 337.0, 357.0, 339.0, 338.0, 319.0, 377.0, 379.0, 280.0, 398.0, 336.0, 332.0, 355.0, 318.0, 375.0, 376.0, 359.0, 336.0, 397.0, 394.0, 378.0, 358.0, 299.0, 337.0, 335.0, 359.0, 375.0, 397.0, 336.0, 339.0, 372.0, 398.0, 356.0, 275.0, 371.0, 296.0, 279.0, 299.0, 336.0, 375.0, 353.0, 336.0, 355.0, 297.0, 335.0, 356.0, 355.0, 353.0, 315.0, 399.0, 276.0, 339.0, 338.0, 359.0, 396.0, 300.0, 355.0, 351.0, 371.0, 337.0, 400.0, 377.0, 396.0, 377.0, 313.0, 330.0, 319.0, 355.0, 314.0, 393.0, 316.0, 378.0, 394.0, 339.0, 400.0, 400.0, 297.0, 340.0, 256.0, 375.0, 357.0, 356.0, 395.0, 398.0, 357.0, 372.0, 336.0, 335.0, 397.0, 358.0, 359.0, 360.0, 359.0, 371.0, 416.0, 375.0, 359.0, 335.0, 358.0, 358.0, 340.0, 336.0, 353.0, 354.0, 376.0, 299.0, 360.0, 334.0, 259.0, 318.0, 314.0, 380.0, 320.0, 377.0, 351.0, 340.0, 377.0, 337.0, 358.0, 319.0, 379.0, 370.0, 339.0, 393.0, 314.0, 378.0, 356.0, 338.0, 355.0, 299.0, 359.0, 392.0, 373.0, 359.0, 337.0, 358.0, 358.0, 298.0, 397.0, 331.0, 399.0, 354.0, 395.0, 376.0, 357.0, 360.0, 395.0, 359.0, 355.0, 357.0, 315.0, 298.0, 359.0, 319.0, 395.0, 355.0, 357.0, 299.0, 337.0, 295.0, 377.0, 355.0, 414.0, 350.0, 394.0, 332.0, 378.0, 316.0, 355.0, 376.0, 375.0, 378.0, 352.0, 315.0, 316.0, 393.0, 354.0, 339.0, 358.0, 259.0, 319.0, 299.0, 353.0, 360.0, 336.0, 319.0, 357.0, 340.0, 337.0, 319.0, 339.0, 316.0, 394.0, 355.0, 360.0, 379.0, 395.0, 348.0, 396.0, 357.0, 331.0, 376.0, 367.0, 338.0, 336.0, 376.0, 360.0, 357.0, 317.0, 338.0, 376.0, 296.0, 394.0, 333.0, 320.0, 313.0, 336.0, 376.0, 354.0, 378.0, 356.0, 360.0, 356.0, 377.0, 420.0, 391.0, 338.0, 356.0, 319.0, 376.0, 415.0, 339.0, 376.0, 337.0, 300.0, 400.0, 335.0, 396.0, 359.0, 319.0, 319.0, 298.0, 396.0, 310.0, 392.0, 380.0, 375.0, 336.0, 357.0, 355.0, 359.0, 296.0, 275.0, 291.0,
    #     933.0, 752.0, 934.0, 732.0, 873.0, 834.0, 837.0, 887.0, 784.0, 913.0, 949.0, 872.0, 865.0, 887.0, 832.0, 895.0, 772.0, 835.0, 848.0, 830.0, 889.0, 849.0, 952.0, 965.0, 828.0, 873.0, 811.0, 732.0, 657.0, 676.0, 856.0, 872.0, 896.0, 815.0, 806.0, 916.0, 849.0, 923.0, 895.0, 851.0, 773.0, 845.0, 974.0, 917.0, 851.0, 864.0, 893.0, 868.0, 874.0, 914.0, 794.0, 910.0, 770.0, 871.0, 750.0, 909.0, 862.0, 689.0, 655.0, 634.0, 892.0, 884.0, 794.0, 763.0, 789.0, 927.0, 754.0, 894.0, 886.0, 977.0, 927.0, 850.0, 904.0, 773.0, 835.0, 873.0, 836.0, 871.0, 814.0, 915.0, 773.0, 814.0, 811.0, 910.0, 908.0, 888.0, 912.0, 733.0, 715.0, 674.0, 930.0, 910.0, 874.0, 893.0, 917.0, 847.0, 833.0, 836.0, 929.0, 850.0, 851.0, 891.0, 779.0, 889.0, 807.0, 791.0, 828.0, 835.0, 850.0, 890.0, 933.0, 906.0, 767.0, 873.0, 863.0, 829.0, 749.0, 737.0, 715.0, 694.0, 872.0, 869.0, 851.0, 786.0, 807.0, 828.0, 897.0, 883.0, 914.0, 863.0, 810.0, 830.0, 883.0, 936.0, 808.0, 813.0, 912.0, 830.0, 869.0, 797.0, 936.0, 896.0, 850.0, 810.0, 859.0, 854.0, 892.0, 694.0, 714.0, 733.0, 856.0, 801.0, 855.0, 814.0, 832.0, 829.0, 846.0, 817.0, 933.0, 823.0, 891.0, 777.0, 910.0, 868.0, 890.0, 896.0, 839.0, 908.0, 856.0, 870.0, 767.0, 824.0, 949.0, 889.0, 856.0, 830.0, 893.0, 751.0, 735.0, 691.0
    # ]
    # dimensions=[
    #     1919.0, 812.0, 5972.0, 1005.0, 504.0, 1338.0, 854.0, 5483.0, 2373.0, 209.0, 2555.0, 701.0, 570.0, 264.0, 1060.0, 359.0, 1426.0, 558.0, 657.0, 791.0, 274.0, 1130.0, 2081.0, 3647.0, 483.0, 1309.0, 1076.0, 446.0, 570.0, 526.0, 2215.0, 934.0, 1170.0, 712.0, 447.0, 683.0, 787.0, 2179.0, 995.0, 6628.0, 651.0, 230.0, 604.0, 285.0, 3025.0, 1879.0, 6594.0, 281.0, 700.0, 2436.0, 6490.0, 2996.0, 2326.0, 2445.0, 6272.0, 399.0, 272.0, 1884.0, 2149.0, 1794.0, 460.0, 891.0, 1012.0, 3686.0, 499.0, 1250.0, 716.0, 997.0, 336.0, 632.0, 1143.0, 252.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 5000.0, 5000.0, 5000.0, 5000.0, 5000.0, 5000.0, 5000.0, 5000.0, 5000.0, 5000.0, 5000.0, 5000.0, 5000.0, 5000.0, 5000.0, 5000.0, 5000.0, 5000.0, 5000.0, 5000.0, 5000.0, 5000.0, 5000.0, 5000.0, 5000.0, 5000.0, 5000.0, 5000.0, 5000.0, 5000.0, 5000.0, 5000.0, 5000.0, 5000.0, 5000.0, 5000.0, 5000.0, 5000.0, 5000.0, 5000.0, 5000.0, 5000.0, 5000.0, 5000.0, 5000.0, 5000.0, 5000.0, 5000.0, 5000.0, 5000.0, 5000.0, 5000.0, 5000.0, 5000.0, 5000.0, 5000.0, 5000.0, 5000.0, 5000.0, 5000.0, 5000.0, 5000.0, 5000.0, 5000.0, 5000.0, 5000.0, 5000.0, 5000.0, 5000.0, 5000.0, 5000.0, 5000.0, 15000.0, 15000.0, 15000.0, 15000.0, 15000.0, 15000.0, 15000.0, 15000.0, 15000.0, 15000.0, 15000.0, 15000.0, 15000.0, 15000.0, 15000.0, 15000.0, 15000.0, 15000.0, 15000.0, 15000.0, 15000.0, 15000.0, 15000.0, 15000.0, 15000.0, 15000.0, 15000.0, 15000.0, 15000.0, 15000.0, 15000.0, 15000.0, 15000.0, 15000.0, 15000.0, 15000.0, 15000.0, 15000.0, 15000.0, 15000.0, 15000.0, 15000.0, 15000.0, 15000.0, 15000.0, 15000.0, 15000.0, 15000.0, 15000.0, 15000.0, 15000.0, 15000.0, 15000.0, 15000.0, 15000.0, 15000.0, 15000.0, 15000.0, 15000.0, 15000.0, 15000.0, 15000.0, 15000.0, 15000.0, 15000.0, 15000.0, 15000.0, 15000.0, 15000.0, 15000.0, 15000.0, 15000.0,
    #     3745.0, 1473.0, 1918.0, 6098.0, 2951.0, 4813.0, 2525.0, 3981.0, 6887.0, 8209.0, 3656.0, 2869.0, 826.0, 7237.0, 1211.0, 5554.0, 1921.0, 2998.0, 2778.0, 11220.0, 7947.0, 2094.0, 1190.0, 2327.0, 3749.0, 2375.0, 3150.0, 1771.0, 1453.0, 3085.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 5000.0, 5000.0, 5000.0, 5000.0, 5000.0, 5000.0, 5000.0, 5000.0, 5000.0, 5000.0, 5000.0, 5000.0, 5000.0, 5000.0, 5000.0, 5000.0, 5000.0, 5000.0, 5000.0, 5000.0, 5000.0, 5000.0, 5000.0, 5000.0, 5000.0, 5000.0, 5000.0, 5000.0, 5000.0, 5000.0, 15000.0, 15000.0, 15000.0, 15000.0, 15000.0, 15000.0, 15000.0, 15000.0, 15000.0, 15000.0, 15000.0, 15000.0, 15000.0, 15000.0, 15000.0, 15000.0, 15000.0, 15000.0, 15000.0, 15000.0, 15000.0, 15000.0, 15000.0, 15000.0, 15000.0, 15000.0, 15000.0, 15000.0, 15000.0, 15000.0
    # ]   # number of labels
    # traintimes=[
    #     0.506, 0.435, 1.12, 0.371, 0.269, 0.659, 0.317, 1.055, 0.635, 0.242, 0.562, 0.433, 0.315, 0.333, 0.506, 0.254, 0.627, 0.403, 0.345, 0.436, 0.33, 0.518, 0.612, 0.904, 0.302, 0.617, 0.409, 0.329, 0.438, 0.355, 0.533, 0.555, 0.533, 0.65, 0.401, 0.415, 0.391, 0.654, 0.56, 1.189, 0.453, 0.341, 0.418, 0.331, 0.743, 0.774, 1.123, 0.223, 0.4, 0.635, 0.982, 0.675, 0.754, 0.626, 0.969, 0.425, 0.299, 0.989, 0.655, 0.536, 0.399, 0.593, 0.452, 0.705, 0.392, 0.55, 0.449, 0.53, 0.346, 0.335, 0.497, 0.234, 1.548, 1.972, 1.678, 1.549, 1.506, 1.4, 1.53, 1.524, 1.134, 1.439, 1.493, 1.616, 1.877, 1.758, 1.373, 1.693, 1.508, 1.604, 1.137, 1.537, 1.426, 1.388, 1.682, 1.66, 1.287, 1.377, 1.376, 1.378, 1.366, 1.675, 1.674, 1.501, 0.944, 1.82, 1.402, 1.184, 1.811, 1.143, 1.468, 1.494, 1.532, 1.363, 1.395, 1.732, 1.452, 1.377, 1.681, 1.506, 1.432, 1.783, 1.697, 1.57, 1.383, 1.852, 1.269, 1.378, 1.771, 1.502, 1.536, 1.673, 1.527, 1.397, 1.689, 1.895, 1.529, 1.178, 1.715, 1.525, 1.945, 1.027, 0.927, 0.907, 0.524, 0.414, 0.458, 0.572, 0.513, 0.433, 0.582, 0.466, 0.482, 0.443, 0.426, 0.532, 0.463, 0.677, 0.457, 0.342, 0.471, 0.431, 0.459, 0.437, 0.481, 0.315, 0.518, 0.437, 0.498, 0.374, 0.431, 0.539, 0.413, 0.52, 0.552, 0.463, 0.483, 0.641, 0.402, 0.473, 0.417, 0.456, 0.59, 0.371, 0.474, 0.505, 0.366, 0.457, 0.608, 0.647, 0.429, 0.604, 0.745, 0.482, 0.507, 0.495, 0.566, 0.476, 0.502, 0.491, 0.507, 0.463, 0.478, 0.483, 0.525, 0.552, 0.385, 0.465, 0.493, 0.503, 0.496, 0.319, 0.533, 0.309, 0.386, 0.43, 0.388, 0.419, 0.41, 0.375, 0.367, 0.248, 0.341, 0.353, 0.279, 0.36, 0.332, 0.325, 0.287, 0.355, 0.254, 0.364, 0.443, 0.223, 0.369, 0.395, 0.444, 0.327, 0.41, 0.364, 0.419, 0.39, 0.36, 0.336, 0.349, 0.412, 0.326, 0.426, 0.386, 0.376, 0.416, 0.372, 0.351, 0.433, 0.316, 0.295, 0.263, 0.41, 0.295, 0.378, 0.401, 0.378, 0.352, 0.459, 0.33, 0.36, 0.44, 0.368, 0.392, 0.368, 0.371, 0.429, 0.439, 0.402, 0.405, 0.368, 0.373, 0.415, 0.393, 0.347, 0.381, 0.395, 0.418, 0.317, 0.398, 0.383, 0.283, 0.382, 0.817, 1.041, 0.804, 0.986, 0.93, 0.882, 0.999, 0.869, 0.955, 0.811, 1.022, 1.016, 0.815, 1.124, 0.773, 1.009, 0.96, 0.868, 0.927, 0.736, 0.946, 1.084, 1.014, 0.95, 0.839, 0.937, 0.933, 0.72, 1.096, 0.818, 1.11, 0.938, 1.059, 1.018, 0.937, 0.935, 1.066, 0.962, 0.94, 0.938, 0.817, 0.733, 0.938, 0.795, 1.13, 0.947, 0.938, 0.713, 0.849, 0.733, 1.012, 0.927, 1.196, 0.959, 1.086, 0.874, 1.018, 0.8, 0.948, 1.064, 1.012, 1.031, 0.955, 0.785, 0.823, 1.117, 0.95, 0.865, 0.951, 0.611, 0.808, 0.721, 2.067, 2.165, 2.045, 1.779, 2.155, 2.015, 2.009, 1.809, 1.878, 1.81, 2.49, 2.053, 2.102, 2.298, 2.488, 2.114, 2.432, 2.055, 2.033, 2.26, 2.266, 2.04, 1.947, 2.254, 2.177, 2.068, 1.752, 1.93, 2.272, 1.633, 2.561, 1.967, 1.706, 1.798, 1.99, 2.268, 2.084, 2.286, 2.081, 2.239, 2.183, 2.288, 2.754, 2.578, 1.886, 2.121, 1.74, 2.251, 2.749, 2.034, 2.298, 1.971, 1.607, 2.479, 1.97, 2.514, 2.083, 1.741, 1.734, 1.615, 2.539, 1.709, 2.469, 2.317, 2.256, 1.859, 2.158, 2.065, 2.091, 1.576, 1.409, 1.549,
    #     3.562, 1.364, 2.353, 3.614, 2.632, 3.691, 2.229, 3.458, 4.212, 6.476, 3.767, 2.44, 1.397, 5.533, 1.259, 4.448, 1.708, 2.298, 2.432, 7.326, 5.842, 1.981, 1.689, 2.584, 2.925, 2.258, 2.379, 1.46, 1.141, 1.75, 6.934, 7.13, 7.448, 6.346, 6.214, 7.82, 6.868, 7.988, 7.506, 6.899, 5.759, 6.852, 8.728, 7.82, 6.87, 7.046, 7.462, 7.118, 7.131, 7.794, 6.013, 7.739, 5.673, 7.187, 5.464, 7.775, 7.105, 4.659, 4.217, 3.96, 1.521, 2.004, 1.224, 1.27, 1.189, 1.542, 1.645, 1.961, 1.451, 1.801, 1.53, 2.028, 1.56, 1.175, 1.335, 1.447, 1.267, 1.423, 1.277, 1.477, 1.237, 1.277, 1.63, 2.064, 1.485, 1.467, 1.533, 1.119, 1.081, 0.996, 1.153, 1.545, 1.075, 1.082, 1.183, 1.029, 0.955, 0.977, 1.205, 1.024, 1.012, 1.09, 0.904, 1.09, 0.992, 1.356, 0.989, 1.018, 0.989, 1.109, 1.207, 1.137, 0.887, 1.086, 1.055, 1.012, 0.945, 0.968, 0.865, 1.056, 4.196, 4.138, 3.913, 3.469, 3.61, 3.689, 4.418, 4.229, 4.42, 4.087, 3.536, 3.654, 4.267, 4.571, 3.493, 3.631, 4.295, 3.689, 4.16, 3.438, 4.531, 4.401, 3.872, 3.51, 3.968, 3.897, 4.2, 2.776, 2.812, 2.971, 10.203, 9.079, 10.084, 9.181, 9.764, 9.593, 10.081, 9.308, 12.051, 9.63, 11.009, 8.458, 11.548, 10.519, 10.946, 11.1, 9.797, 11.51, 10.125, 10.545, 8.346, 9.56, 12.463, 11.048, 10.171, 9.644, 11.024, 7.95, 7.699, 6.843
    # ]   # training latency

    # colo = traintimes 
    # color_map = cm.ScalarMappable(cmap=cm.PiYG) 
    # color_map.set_array(colo) 
    # colo_normalized = [c/max(colo) for c in colo]
    # # ax.scatter(samplesize, dimensions, traintimes, facecolors=cm.PiYG(colo_normalized), edgecolor=cm.PiYG(colo_normalized), alpha=1)
    # ax.scatter(samplesize, dimensions, facecolors=cm.PiYG(colo_normalized), edgecolor=cm.PiYG(colo_normalized), alpha=1)
    # plt.colorbar(color_map,label='Training Latency (s)') 

    # ax.set_xlabel('Sample Size')
    # ax.set_ylabel('Feature Dimension Size')
    # # ax.set_zlabel('Training Latency')
    # # ax.zaxis.labelpad = -4

    # # plt.show()
    # # fig.set_size_inches(6, 8)
    # fig.tight_layout()
    # # fig.subplots_adjust(left=-2) 
    # plt.savefig(fig_path+filename+'.pdf', bbox_inches='tight')
    # plt.close()






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
    ys0_l=[[ys0/60/60*0.013*64 for ys0 in [0.421, 0.886, 1.335, 2.112, 5.392, 91.993, 9847.238]]]
    ys0mean_l = np.array(ys0_l).mean(axis=0).tolist()
    ys0std_l  = np.array(ys0_l).std(axis=0).tolist()
    ys2_l=[[ys2/60/60*2.448 for ys2 in [0.421, 0.886, 1.335, 2.112, 5.392, 91.993, 9847.238]]]
    ys2mean_l = np.array(ys2_l).mean(axis=0).tolist()
    ys2std_l  = np.array(ys2_l).std(axis=0).tolist()
    ys1_l=[[0.421, 0.886, 1.335, 2.112, 5.392, 91.993, 9847.238]]
    ys1mean_l = np.array(ys1_l).mean(axis=0).tolist()
    ys1std_l  = np.array(ys1_l).std(axis=0).tolist()
    # ys0conf_l = list(scipy.stats.t.interval(0.95, len(ys0_l)-1, loc=np.mean(ys0_l,axis=0), scale=scipy.stats.sem(ys0_l,axis=0)))
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    # bottom = np.zeros(len(xs))
    entry_count, width = 3, 0.4
    p = ax.bar([idx-width/entry_count-width/entry_count/2 for idx, _ in enumerate(ys0mean_l)], ys0mean_l, width/entry_count, yerr=ys0std_l, color='#0067ff', edgecolor="black", hatch="x", label="NERC VM ($)")
    p2 = ax.bar([idx-width/entry_count/2 for idx, _ in enumerate(ys2mean_l)], ys2mean_l, width/entry_count, yerr=ys2std_l, color='#00ff67', edgecolor="black", hatch="-", label="AWS EC2 ($)")
    # ax.bar_label(p)
    # ax.set_title("Training Latency by N Models with Data 3", fontsize=20)
    ax.grid()
    ax.legend(loc="upper left", prop={'size': 16})
    ax.set_xticks(list(range(len(xs))))
    ax.set_xticklabels(xs)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.tick_params(axis='both', which='minor', labelsize=18)
    ax.set_xlabel("Number of Labels", fontsize=20)
    ax.set_ylabel("Resource Cost ($)", fontsize=20)
    ax1 = ax.twinx()
    p = ax1.bar([idx+width/entry_count/2 for idx, _ in enumerate(ys1mean_l)], ys1mean_l, width/entry_count, yerr=ys1std_l, color='#ff6700', edgecolor="black", hatch="o", label="Time (s)")
    # ax1.bar_label(p)
    ax1.tick_params(axis='both', which='major', labelsize=20)
    ax1.tick_params(axis='both', which='minor', labelsize=18)
    ax1.set_ylabel("Training Time(s)", fontsize=20)
    ax1.legend(loc="upper center", prop={'size': 16})
    # plt.show()
    plt.savefig(fig_path+filename+'.pdf', bbox_inches='tight')
    plt.close()


    fig_path = '/home/cc/Praxi-study/Praxi-Pipeline/prediction_XGBoost_openshift_image/model_testing_scripts/figs/'
    filename = "nerccostandtrainlatencysummodel_by_N_models_with_rawinput_data_3"
    xs=[str(xlabel) for xlabel in [500//50,500//25,500//20,500//15+1,500//10,500//5,500//1]]
    labels = ["model "+str(i+1) for i in range(46)]
    ys0_l=[[ys0/60/60*0.013*64 for ys0 in [20.14, 36.68, 51.58, 101.09, 249.6, 942.14, 9847.24]]]
    ys0mean_l = np.array(ys0_l).mean(axis=0).tolist()
    ys0std_l  = np.array(ys0_l).std(axis=0).tolist()
    ys2_l=[[ys2/60/60*2.448 for ys2 in [20.14, 36.68, 51.58, 101.09, 249.6, 942.14, 9847.24]]]
    ys2mean_l = np.array(ys2_l).mean(axis=0).tolist()
    ys2std_l  = np.array(ys2_l).std(axis=0).tolist()
    ys1_l=[
        [0.42, 0.89, 1.33, 2.11, 5.39, 91.99, 9847.24] ,
        [0.46, 0.81, 0.86, 1.36, 6.53, 84.55, 0] ,
        [0.34, 0.59, 1.1, 2.19, 5.14, 221.68, 0] ,
        [0.36, 0.86, 1.11, 2.41, 5.75, 237.0, 0] ,
        [0.28, 0.82, 1.28, 1.45, 33.52, 306.92, 0] ,
        [0.31, 0.87, 1.02, 2.2, 34.58, 0, 0] ,
        [0.36, 0.67, 1.32, 9.1, 26.02, 0, 0] ,
        [0.41, 0.59, 0.86, 2.88, 45.59, 0, 0] ,
        [0.32, 0.88, 3.38, 3.46, 53.61, 0, 0] ,
        [0.27, 0.74, 1.74, 13.32, 33.47, 0, 0] ,
        [0.36, 2.08, 1.06, 10.91, 0, 0, 0] ,
        [0.35, 1.21, 5.5, 8.48, 0, 0, 0] ,
        [0.36, 0.9, 2.04, 1.93, 0, 0, 0] ,
        [0.31, 1.18, 1.84, 26.65, 0, 0, 0] ,
        [0.29, 1.86, 3.45, 12.62, 0, 0, 0] ,
        [0.26, 1.42, 2.4, 0, 0, 0, 0] ,
        [0.32, 0.78, 1.08, 0, 0, 0, 0] ,
        [0.35, 2.21, 15.83, 0, 0, 0, 0] ,
        [0.32, 0.94, 2.06, 0, 0, 0, 0] ,
        [0.26, 1.68, 2.3, 0, 0, 0, 0] ,
        [0.57, 1.11, 0, 0, 0, 0, 0] ,
        [0.44, 9.68, 0, 0, 0, 0, 0] ,
        [0.43, 0.99, 0, 0, 0, 0, 0] ,
        [0.39, 1.31, 0, 0, 0, 0, 0] ,
        [0.31, 1.63, 0, 0, 0, 0, 0] ,
        [0.58, 0, 0, 0, 0, 0, 0] ,
        [0.18, 0, 0, 0, 0, 0, 0] ,
        [0.47, 0, 0, 0, 0, 0, 0] ,
        [0.33, 0, 0, 0, 0, 0, 0] ,
        [0.62, 0, 0, 0, 0, 0, 0] ,
        [0.4, 0, 0, 0, 0, 0, 0] ,
        [0.37, 0, 0, 0, 0, 0, 0] ,
        [0.29, 0, 0, 0, 0, 0, 0] ,
        [0.43, 0, 0, 0, 0, 0, 0] ,
        [0.41, 0, 0, 0, 0, 0, 0] ,
        [0.61, 0, 0, 0, 0, 0, 0] ,
        [0.44, 0, 0, 0, 0, 0, 0] ,
        [0.2, 0, 0, 0, 0, 0, 0] ,
        [0.6, 0, 0, 0, 0, 0, 0] ,
        [0.31, 0, 0, 0, 0, 0, 0] ,
        [0.27, 0, 0, 0, 0, 0, 0] ,
        [0.43, 0, 0, 0, 0, 0, 0] ,
        [0.5, 0, 0, 0, 0, 0, 0] ,
        [1.42, 0, 0, 0, 0, 0, 0] ,
        [0.56, 0, 0, 0, 0, 0, 0] ,
        [0.28, 0, 0, 0, 0, 0, 0]
        ]
    # ys1mean_l = np.array(ys1_l).mean(axis=0).tolist()
    # ys1std_l  = np.array(ys1_l).std(axis=0).tolist()
    # ys0conf_l = list(scipy.stats.t.interval(0.95, len(ys0_l)-1, loc=np.mean(ys0_l,axis=0), scale=scipy.stats.sem(ys0_l,axis=0)))
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    # bottom = np.zeros(len(xs))
    entry_count, width = 3, 0.4
    p = ax.bar([idx-width/entry_count-width/entry_count/2 for idx, _ in enumerate(ys0mean_l)], ys0mean_l, width/entry_count, yerr=ys0std_l, color='#0067ff', edgecolor="black", hatch="x", label="NERC VM ($)")
    p2 = ax.bar([idx-width/entry_count/2 for idx, _ in enumerate(ys2mean_l)], ys2mean_l, width/entry_count, yerr=ys2std_l, color='#00ff67', edgecolor="black", hatch="|", label="AWS EC2 ($)")
    # ax.bar_label(p)
    # ax.set_title("Training Latency by N Models with Data 3", fontsize=20)
    ax.grid()
    ax.legend(loc="upper left", prop={'size': 16})
    ax.set_xticks(list(range(len(xs))))
    ax.set_xticklabels(xs)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.tick_params(axis='both', which='minor', labelsize=18)
    ax.set_xlabel("Number of Labels", fontsize=20)
    ax.set_ylabel("Resource Cost ($)", fontsize=20)
    ax1 = ax.twinx()
    bottom = [0 for _ in range(len(ys1_l[0]))]
    for row in ys1_l[:-1]:
        p = ax1.bar([idx+width/entry_count/2 for idx, _ in enumerate(row)], row, width/entry_count, bottom, color='#ff6700', edgecolor="black", hatch="o")
        bottom = [v1+v2 for v1,v2 in zip(bottom,row)]
    p = ax1.bar([idx+width/entry_count/2 for idx, _ in enumerate(ys1_l[-1])], ys1_l[-1], width/entry_count, bottom, color='#ff6700', edgecolor="black", hatch="o", label="Time for each submodel accumulated (s)")
    # p = ax1.bar([idx+width/entry_count/2 for idx, _ in enumerate(ys1mean_l)], ys1mean_l, width/entry_count, yerr=ys1std_l, color='#ff6700', edgecolor="black", hatch="o", label="Time (s)")
    # ax1.bar_label(p)
    ax1.tick_params(axis='both', which='major', labelsize=20)
    ax1.tick_params(axis='both', which='minor', labelsize=18)
    ax1.set_ylabel("Training Time(s)", fontsize=20)
    ax1.legend(loc="center left", prop={'size': 16})
    # plt.show()
    plt.savefig(fig_path+filename+'.pdf', bbox_inches='tight')
    plt.close()


    fig_path = '/home/cc/Praxi-study/Praxi-Pipeline/prediction_XGBoost_openshift_image/model_testing_scripts/figs/'
    filename = "trainlatency_by_N_models_with_rawinput_data_0"
    xs=[str(xlabel) for xlabel in [10,20,40,80]]
    labels = ["model "+str(i+1) for i in range(46)]
    ys0_l=[[8*2.4, 4*2.4*2*(4*math.log(25*20)/math.log(25*10)), 2*19.0*2*(4*math.log(25*40)/math.log(25*20)), 143.5*2*(4*math.log(25*80)/math.log(25*40))]]
    ys1_l=[
        [2.407, 18.840, 143.565, 1090.042],
        [2.396, 18.788, 143.717, 0],
        [2.413, 19.016, 0, 0],
        [2.407, 18.862, 0, 0],
        [2.406, 0, 0, 0],
        [2.396, 0, 0, 0],
        [2.440, 0, 0, 0],
        [2.396, 0, 0, 0]
        ]
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    # bottom = np.zeros(len(xs))
    entry_count, width = 2, 0.4
    p = ax.bar([idx-width/entry_count/2 for idx, _ in enumerate(ys0_l[0])], ys0_l[0], width/entry_count, color='#00ab00', edgecolor="black", hatch="x", label="Estimated")
    ax.bar_label(p)
    # ax.set_title("Training Latency by N Models with Data 3", fontsize=20)
    ax.grid()
    ax.legend(loc="upper left", prop={'size': 16})
    ax.set_xticks(list(range(len(xs))))
    ax.set_xticklabels(xs)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.tick_params(axis='both', which='minor', labelsize=18)
    ax.set_xlabel("Number of Labels", fontsize=20)
    ax.set_ylabel("Resource Cost ($)", fontsize=20)
    ax1 = ax.twinx()
    bottom = [0 for _ in range(len(ys1_l[0]))]
    for row in ys1_l[:-1]:
        p = ax1.bar([idx+width/entry_count/2 for idx, _ in enumerate(row)], row, width/entry_count, bottom, color='#ff6700', edgecolor="black", hatch="o")
        bottom = [v1+v2 for v1,v2 in zip(bottom,row)]
    p = ax1.bar([idx+width/entry_count/2 for idx, _ in enumerate(ys1_l[-1])], ys1_l[-1], width/entry_count, bottom, color='#ff6700', edgecolor="black", hatch="o", label="Observed Time for each submodel accumulated (s)")
    ax1.bar_label(p)
    ax1.tick_params(axis='both', which='major', labelsize=20)
    ax1.tick_params(axis='both', which='minor', labelsize=18)
    ax1.set_ylabel("Training Time(s)", fontsize=20)
    ax1.legend(loc="center left", prop={'size': 16})
    # plt.show()
    plt.savefig(fig_path+filename+'.pdf', bbox_inches='tight')
    plt.close()



    # by_input_size
    # fig_path = '/home/cc/Praxi-study/Praxi-Pipeline/prediction_XGBoost_openshift_image/model_testing_scripts/figs/'
    # filename = "testf1score_by_input_size_with_rawinput_data_0"
    # fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    # xs=[13, 106, 284, 427, 854, 1138, 1708, 3416, 6832, 13664, 27329, 54659, 109319]
    # labels = ("80 labels", "40 labels","20 labels","10 labels")
    # ys_l=[[0.077, 0.521, 0.697, 0.727, 0.806, 0.843, 0.847, 0.906, 0.935, 0.934, 0.948, 0.951, 0.954],
    #     [0.166, 0.609, 0.746, 0.789, 0.857, 0.846, 0.891, 0.896, 0.911, 0.905, 0.911, 0.904, 0.916],
    #     [0.210, 0.633, 0.758, 0.763, 0.829, 0.841, 0.828, 0.886, 0.889, 0.884, 0.902, 0.891, 0.898],
    #     [0.259, 0.588, 0.714, 0.775, 0.741, 0.836, 0.851, 0.837, 0.815, 0.843, 0.842, 0.846, 0.843]
    #     ]
    # for ys, label in zip(ys_l, labels):
    #     ax.scatter(xs, ys, label=label)
    # ax.tick_params(axis='both', which='major', labelsize=20)
    # ax.tick_params(axis='both', which='minor', labelsize=18)
    # ax.set_xlabel("Input Dimensions", fontsize=20)
    # ax.set_ylabel("F1-Scores", fontsize=20)
    # plt.legend(prop={'size': 16})
    # # plt.show()
    # plt.savefig(fig_path+filename+'.pdf', bbox_inches='tight')
    # plt.close()

    # fig_path = '/home/cc/Praxi-study/Praxi-Pipeline/prediction_XGBoost_openshift_image/model_testing_scripts/figs/'
    # filename = "trainlatency_by_input_size_with_rawinput_data_0"
    # fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    # xs=[13, 106, 284, 427, 854, 1138, 1708, 3416, 6832, 13664, 27329, 54659, 109319]
    # labels = ("10 labels", "20 labels","40 labels","80 labels")
    # ys_l=[[0.102, 0.103, 0.143, 0.156, 0.226, 0.278, 0.392, 0.664, 1.205, 2.407, 5.449, 11.784, 24.566],
    #     [0.229, 0.290, 0.406, 0.451, 0.700, 0.928, 1.255, 2.366, 4.464, 9.328, 18.840, 39.945, 80.171],
    #     [0.315, 0.550, 1.027, 1.514, 2.580, 3.246, 4.596, 8.739, 17.492, 34.972, 70.854, 143.565, 286.711],
    #     [1.227, 2.012, 3.810, 5.275, 9.333, 12.124, 17.675, 34.274, 67.655, 136.472, 274.488, 546.612, 1096.709]
    #     ]
    # for ys, label in zip(ys_l, labels):
    #     ax.scatter(xs, ys, label=label)
    # ax.tick_params(axis='both', which='major', labelsize=20)
    # ax.tick_params(axis='both', which='minor', labelsize=18)
    # ax.set_xlabel("Input Dimensions", fontsize=20)
    # ax.set_ylabel("Training Time(s)", fontsize=20)
    # plt.legend(prop={'size': 16})
    # # plt.show()
    # plt.savefig(fig_path+filename+'.pdf', bbox_inches='tight')
    # plt.close()

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
    labels = ("10 labels per model", "20 labels per model","40 labels per model","80 labels per model")
    ys_l=[[0.212+0.214+0.214+0.212+0.214+0.213+0.213+0.218, 0.408+0.409+0.411+0.411+0.409+0.409+0.410+0.409, 0.810+0.817+0.817+0.815+0.814+0.809+0.814+0.813, 1.675+1.686+1.681+1.677+1.672+1.679+1.679+1.678],
        [0.226+0.226+0.225+0.227, 0.425+0.424+0.423+0.425, 0.831+0.830+0.831+0.836, 1.711+1.709+1.706+1.711],
        [0.268+0.263, 0.461+0.465, 0.875+0.883, 1.772+1.779],
        [0.380, 0.547, 0.979, 1.959]
        ]
    for ys, label in zip(ys_l, labels):
        ax.scatter(xs, ys, label=label)
    ax.hlines(1.92, xmin=13664, xmax=109319)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.tick_params(axis='both', which='minor', labelsize=18)
    ax.set_xlabel("Input Dimensions", fontsize=20)
    ax.set_ylabel("Inference Time(s)", fontsize=20)
    plt.legend(prop={'size': 16})
    # plt.show()
    plt.savefig(fig_path+filename+'.pdf', bbox_inches='tight')
    plt.close()

    # fig_path = '/home/cc/Praxi-study/Praxi-Pipeline/prediction_XGBoost_openshift_image/model_testing_scripts/figs/'
    # filename = "trainlatency_by_input_size_with_rawinput_data_0_estimated"
    # fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    # xs=[13, 106, 284, 427, 854, 1138, 1708, 3416, 6832, 13664, 27329, 54659, 109319]
    # labels = ("Observed", "Estimated")
    # ys_l=[[1.227, 2.012, 3.810, 5.275, 9.333, 12.124, 17.675, 34.274, 67.655, 136.472, 274.488, 546.612, 1096.709],
    #     #   [1.227, 1.227*2, 1.227*2**2, 1.227*2**3, 1.227*2**4, 1.227*2**5, 1.227*2**6, 1.227*2**7, 1.227*2**8, 1.227*2**9, 1.227*2**10, 1.227*2**11, 1.227*2**12],
    #     # [1.227, 1.227*2, 2.012*2, 3.810*2, 5.275*2, 9.333*2, 12.124*2, 17.675*2, 34.274*2, 67.655*2, 136.472*2, 274.488*2, 546.612*2]
    #     [1096.709/2**12, 1096.709/2**11, 1096.709/2**10, 1096.709/2**9, 1096.709/2**8, 1096.709/2**7, 1096.709/2**6, 1096.709/2**5, 1096.709/2**4, 1096.709/2**3, 1096.709/2**2, 1096.709/2, 1096.709]
    #     ]
    # for ys, label in zip(ys_l, labels):
    #     ax.scatter(xs, ys, label=label)
    # ax.tick_params(axis='both', which='major', labelsize=20)
    # ax.tick_params(axis='both', which='minor', labelsize=18)
    # ax.set_xlabel("Input Dimensions", fontsize=20)
    # ax.set_ylabel("Training Time(s)", fontsize=20)
    # plt.legend(prop={'size': 16})
    # # plt.show()
    # plt.savefig(fig_path+filename+'.pdf', bbox_inches='tight')
    # plt.close()

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
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    xs=[10, 20, 25, 34, 50, 100]
    labels = ("500labels-F1-with-filter", "500labels-F1-no-filter")
    # labels = ("500labels-F1-with-filter", "500labels-Precision-with-filter", "500labels-F1-no-filter", "500labels-Precision-no-filter")
    ys_l=[
        [0.884, 0.885, 0.885, 0.883, 0.884, 0.901],
        # [0.834, 0.836, 0.837, 0.834, 0.834, 0.855],
        [0.618, 0.641, 0.655, 0.679, 0.696, 0.743],
        # [0.557, 0.58, 0.592, 0.617, 0.635, 0]
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
    ax.set_ylim(0.3,1)
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