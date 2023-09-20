import matplotlib.pyplot as plt
import numpy as np

def plotting(fig_path, filename, cates_values, labels):
    width = 100

    fig, ax = plt.subplots()

    for i, cate_values in enumerate(cates_values):
        bottom = np.zeros(len(labels))
        for cate, value in cate_values.items():
            if labels[0]==None:
                p = ax.bar([float(entry) for entry in cate.split("-")], value, width/len(cates_values), bottom=bottom)
            else:
                p = ax.bar([idx - width/len(cates_values)/2 + i*width/len(cates_values) for idx, _ in enumerate(value)], value, width/len(cates_values), label=cate, bottom=bottom)
            bottom += value
            ax.bar_label(p)

    ax.set_title(" ".join(filename.split("_")))
    if labels[0]!=None:
        ax.legend(loc="best")
        ax.set_xticks(list(range(len(labels))))
        ax.set_xticklabels(labels)
    ax.set_ylabel("Train Time (Seconds)")
    ax.set_xlabel("Dimensions")

    # plt.show()
    plt.savefig(fig_path+filename+'.png', bbox_inches='tight')
    plt.close()



if __name__ == "__main__":
    # # N Estimators
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
    #     "8","4","1"
    # )
    # cate_values_1 = {
    #     "M12-M23-M80": np.array([0.567, 4.024, 954.138]),
    #     "M11-M18-MX": np.array([1.965, 9.069, 0]),
    #     "M7-M19-MX": np.array([0.391, 9.117, 0]),
    #     "M11-M20-MX": np.array([5.592, 163.841, 0]),
    #     "M7-MX-MX": np.array([1.020, 0, 0]),
    #     "M12-MX-MX": np.array([4.595, 0, 0]),
    #     "M10-MX-MX_1": np.array([71.705, 0, 0]),
    #     "M10-MX-MX_2": np.array([3.430, 0, 0]),
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




    fig_path = '/home/cc/Praxi-study/Praxi-Pipeline/prediction_XGBoost_openshift_image/model_testing_scripts/figs/'
    filename = "Traintime_by_Inputsize_for_Fixed_10_Labels"
    
    # data = np.array([0.567, 1.965, 0.391, 5.592, 1.020, 4.595, 71.705, 3.430, 63.508, 56.245, 49.186, 13.196, 11.022])
    # labels = [None]*len(data)
    # cate_values_1 = {
    #     "510-3691-454-13495-3560-9740-70225-8498-61711-55725-48127-37249-32293": data
    # }
    # data = np.array([57.646, 51.001, 44.638, 9.824, 8.782, 11.007, 11.971, 12.618, 12.865, 12.569, 13.034, 12.393, 44.370, 47.566])
    # labels = [None]*len(data)
    # cate_values_1 = {
    #     "61711-55725-48127-37249-32293-43152-45970-46161-46245-46993-46838-47857-47980-50394": data
    # }
    data = np.array([0.567, 3.430, 1.965, 0.391, 5.592, 1.020, 4.595,  11.971, 12.618, 12.865, 12.569, 13.034, 12.393, 9.824, 8.782])
    labels = [None]*len(data)
    cate_values_1 = {
        "510-8498-3691-454-13495-3560-9740-45970-46161-46245-46993-46838-47857-37249-32293": data
    }
    cate_values = [cate_values_1]
    plotting(fig_path, filename, cate_values, labels)