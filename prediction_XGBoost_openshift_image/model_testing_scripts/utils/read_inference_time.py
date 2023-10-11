# inference_times_l = []
# for input_size in [13, 106, 284, 427, 854, 1138, 1708, 3416, 6832, 13664, 27329, 54659, 109319]: # [13, 106, 284, 427, 854, 1138, 1708, 3416, 6832, 13664, 27329, 54659, 109319]
#     filename = "/home/cc/Praxi-study/Praxi-Pipeline/prediction_XGBoost_openshift_image/model_testing_scripts/cwd_ML_with_data_0_4_train_8njobs_100trees_1depth_"+str(input_size)+"rawinput_sampling1_exacttreemethod/metrics_pred.out"
#     inference_time_l = []
#     with open(filename) as file:
#         for line in file:
#             line_l = line.rstrip().split(":")
#             if line_l[0] == " BOW_XGB.predict":
#                 inference_time_l.append(line_l[1])
#     # print("+".join(inference_time_l))
#     inference_times_l.append("+".join(inference_time_l))
# print(", ".join(inference_times_l))


# # based on input dimensions
# for n_models in [8,4,2,1]:
#     inference_times_l = []
#     for input_size in [13664, 27329, 54659, 109319]:#[13, 106, 284, 427, 854, 1138, 1708, 3416, 6832, 13664, 27329, 54659, 109319]: # [13, 106, 284, 427, 854, 1138, 1708, 3416, 6832, 13664, 27329, 54659, 109319]
#                     # /home/cc/Praxi-study/Praxi-Pipeline/prediction_XGBoost_openshift_image/model_testing_scripts/cwd_ML_with_data_0_                8_train_8njobs_100trees_1depth_               Nonerawinput_sampling1_exacttreemethod_1maxbin
#         filename = "/home/cc/Praxi-study/Praxi-Pipeline/prediction_XGBoost_openshift_image/model_testing_scripts/cwd_ML_with_data_0_"+str(n_models)+"_train_8njobs_100trees_1depth_"+str(input_size)+"rawinput_sampling1_exacttreemethod/metrics_pred.out"
#         inference_time_l = []
#         with open(filename) as file:
#             for line in file:
#                 line_l = line.rstrip().split(":")
#                 if line_l[0] == " BOW_XGB.predict":
#                     inference_time_l.append(line_l[1])
#         # print("+".join(inference_time_l))
#         inference_times_l.append("+".join(inference_time_l))
#     print("["+", ".join(inference_times_l)+"],")


# based on number of labels per model
for input_size in [None]:#[13, 106, 284, 427, 854, 1138, 1708, 3416, 6832, 13664, 27329, 54659, 109319]: # [13, 106, 284, 427, 854, 1138, 1708, 3416, 6832, 13664, 27329, 54659, 109319]
    inference_times_l = []
    for n_models in [8,4,2,1]:
                    # /home/cc/Praxi-study/Praxi-Pipeline/prediction_XGBoost_openshift_image/model_testing_scripts/cwd_ML_with_data_0_                8_train_8njobs_100trees_1depth_               Nonerawinput_sampling1_exacttreemethod_1maxbin
                    # /home/cc/Praxi-study/Praxi-Pipeline/prediction_XGBoost_openshift_image/model_testing_scripts/cwd_ML_with_data_0_                8_train_1njobs_8clfnjobs_100trees_1depth_               Nonerawinput_sampling1_exacttreemethod_1maxbin
        filename = "/home/cc/Praxi-study/Praxi-Pipeline/prediction_XGBoost_openshift_image/model_testing_scripts/cwd_ML_with_data_0_"+str(n_models)+"_train_1njobs_8clfnjobs_100trees_1depth_"+str(input_size)+"rawinput_sampling1_exacttreemethod_1maxbin/metrics_pred.out"
        inference_time_l = []
        time_counter = 0
        with open(filename) as file:
            for line in file:
                line_l = line.rstrip().split(":")
                if line_l[0] == " BOW_XGB.predict":
                    inference_time_l.append(float(line_l[1]))
                    time_counter+=1
            else:
                inference_time_l.extend([0]*(8-time_counter))
        # print("+".join(inference_time_l))
        inference_times_l.append(inference_time_l)
    inference_times_l = list(map(list, zip(*inference_times_l)))
    # print(inference_times_l)
    for inference_time_l in inference_times_l:
        print(inference_time_l)
    # print("["+", ".join(inference_times_l)+"],")