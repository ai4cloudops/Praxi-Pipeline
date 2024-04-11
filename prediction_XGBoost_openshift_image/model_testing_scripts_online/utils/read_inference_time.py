import os
from collections import defaultdict
import json, pickle, yaml


# Read all data
# # data 4 experiments
data_d = dict()
n_samples = 4
for dataset in ["data_4"]:
    data_dir = ""
    for (with_filter, freq) in [[(False, 100),(True, 25)][1]]:
        for n_jobs in [1]:
            for clf_njobs in [32]:
                for n_models, sim_thr, test_batch_count in zip([1000, 1],[0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.7],[1,1,1,1,1,1,1]):
                # for n_models, sim_thr, test_batch_count in zip([25, 25, 25, 25, 25, 25],[0.98, 0.9, 0.8, 0.7, 0.6],[1,1,1,1,1,1,1,1,1,1]):
                    data_d[str(n_models)+"_models_"+str(sim_thr)+"_sim_thr"] = dict()
                    for n_estimators in [100]: # 10,50,100
                        data_d[str(n_models)+"_models_"+str(sim_thr)+"_sim_thr"][n_estimators] = dict()
                        for depth in [1]:
                            for tree_method in["exact"]: # "exact","approx","hist"
                                for max_bin in [1]:
                                    
                                    for input_size, dim_compact_factor in zip([None],[1]): # [None, 10000, 1000, 500, 5000, 15000],[1,1,1,1,1,1] # [None, 13, 106, 284, 427, 854, 1138, 1708, 3416, 6832, 13664, 27329, 54659, 109319]
                                        data_d[str(n_models)+"_models_"+str(sim_thr)+"_sim_thr"][n_estimators][input_size] = defaultdict(int)
                                        for shuffle_idx in range(1):
                                            for test_sample_batch_idx in [0]:
                                                # filename  ="/home/cc/Praxi-study/Praxi-Pipeline/prediction_XGBoost_openshift_image/model_testing_scripts_online/tagsets_ML_test_120_coo_matrix/"+data_dir+"cwd_ML_with_"+dataset+"_"+str(n_models)+"_train_"+str(shuffle_idx)+"shuffleidx_"+str(test_sample_batch_idx)+"testsamplebatchidx_"+str(n_samples)+"nsamples_"+str(n_jobs)+"njobs_"+str(clf_njobs)+"clfnjobs_"+str(n_estimators)+"trees_"+str(depth)+"depth_"+str(input_size)+"-"+str(dim_compact_factor)+"rawinput_sampling1_"+str(tree_method)+"treemethod_"+str(max_bin)+"maxbin_modize_par_"+str(with_filter)+f"{freq}removesharedornoisestags_verpak_on_demand_expert/metrics.yaml"
                                                filename  ="/home/cc/Praxi-study/Praxi-Pipeline/prediction_XGBoost_openshift_image/model_testing_scripts_online/"+data_dir+"cwd_ML_with_"+dataset+"_"+str(n_models)+"_train_"+str(shuffle_idx)+"shuffleidx_"+str(test_sample_batch_idx)+"testsamplebatchidx_"+str(n_samples)+"nsamples_"+str(n_jobs)+"njobs_"+str(clf_njobs)+"clfnjobs_"+str(n_estimators)+"trees_"+str(depth)+"depth_"+str(input_size)+"-"+str(dim_compact_factor)+"rawinput_sampling1_"+str(tree_method)+"treemethod_"+str(max_bin)+"maxbin_modize_par_"+str(with_filter)+f"{freq}removesharedornoisestags_verpak_on_demand_expert/metrics.yaml"
                                                # filename  ="/home/cc/Praxi-study/Praxi-Pipeline/prediction_XGBoost_openshift_image/model_testing_scripts_online/"+data_dir+"cwd_ML_with_"+dataset+"_"+str(n_models)+"_train_"+str(shuffle_idx)+"shuffleidx_"+str(test_sample_batch_idx)+"testsamplebatchidx_"+str(n_samples)+"nsamples_"+str(n_jobs)+"njobs_"+str(clf_njobs)+"clfnjobs_"+str(n_estimators)+"trees_"+str(depth)+"depth_"+str(input_size)+"-"+str(dim_compact_factor)+"rawinput_sampling1_"+str(tree_method)+"treemethod_"+str(max_bin)+"maxbin_modize_par_"+str(with_filter)+f"{freq}removesharedornoisestags_verpak_on_demand_expert/metrics_pred.out"
                                                # filename  ="/home/cc/Praxi-study/Praxi-Pipeline/prediction_XGBoost_openshift_image/model_testing_scripts_online/"+data_dir+"cwd_ML_with_"+dataset+"_"+str(n_models)+"_train_"+str(shuffle_idx)+"shuffleidx_"+str(test_sample_batch_idx)+"testsamplebatchidx_"+str(n_samples)+"nsamples_"+str(n_jobs)+"njobs_"+str(clf_njobs)+"clfnjobs_"+str(n_estimators)+"trees_"+str(depth)+"depth_"+str(input_size)+"-"+str(dim_compact_factor)+"rawinput_sampling1_"+str(tree_method)+"treemethod_"+str(max_bin)+"maxbin_modize_par_"+str(with_filter)+f"{freq}removesharedornoisestags_verpak/metrics_pred.out"
                                                # filename  ="/home/cc/Praxi-study/Praxi-Pipeline/prediction_XGBoost_openshift_image/model_testing_scripts_online/False100_clustering/"+data_dir+"cwd_ML_with_"+dataset+"_"+str(n_models)+"_train_"+str(shuffle_idx)+"shuffleidx_"+str(test_sample_batch_idx)+"testsamplebatchidx_"+str(n_samples)+"nsamples_"+str(n_jobs)+"njobs_"+str(clf_njobs)+"clfnjobs_"+str(n_estimators)+"trees_"+str(depth)+"depth_"+str(input_size)+"-"+str(dim_compact_factor)+"rawinput_sampling1_"+str(tree_method)+"treemethod_"+str(max_bin)+"maxbin_modize_par_"+str(with_filter)+f"{freq}removesharedornoisestags_sim{sim_thr}assignment/metrics_pred.out"
                                                          # /home/cc/Praxi-study/Praxi-Pipeline/prediction_XGBoost_openshift_image/model_testing_scripts_online/            cwd_ML_with_data_4     _50               _train_  0                 shuffleidx_  0                           testsamplebatchidx_  4               nsamples_32             njobs_  32              clfnjobs_  100                trees_  1           depth_  None             -  1                        rawinput_sampling1_  exact             treemethod_  1             maxbin_modize_par_  False              100   removesharedornoisestags_sim0.85assignment

                                                if os.path.isfile(filename):
                                                    try:
                                                        # data_d[str(n_models)+"_models_"+str(sim_thr)+"_sim_thr"][n_estimators][input_size]["total_encoder_selector_time"] = 0
                                                        with open(filename, "r") as f:
                                                            metrics = yaml.load(f, Loader=yaml.Loader)
                                                            total_encoder_selector_time = 0
                                                            total_encoder_gen_mat_time = 0
                                                            total_encoder_gen_mapping_time = 0
                                                            for clf_idx in range(n_models):
                                                                data_d[str(n_models)+"_models_"+str(sim_thr)+"_sim_thr"][n_estimators][input_size]["encoder_gen_mapping_time"]+=metrics[f"encoder{clf_idx}_op_durations"]["gen_mapping"]
                                                                data_d[str(n_models)+"_models_"+str(sim_thr)+"_sim_thr"][n_estimators][input_size]["encoder_get_feature_time"]+=metrics[f"encoder{clf_idx}_op_durations"]["get_feature"]
                                                                data_d[str(n_models)+"_models_"+str(sim_thr)+"_sim_thr"][n_estimators][input_size]["encoder_selector_time"]+=metrics[f"encoder{clf_idx}_op_durations"]["selector"]
                                                                data_d[str(n_models)+"_models_"+str(sim_thr)+"_sim_thr"][n_estimators][input_size]["encoder_mat_builder_time"]+=metrics[f"encoder{clf_idx}_op_durations"]["mat_builder"]
                                                                data_d[str(n_models)+"_models_"+str(sim_thr)+"_sim_thr"][n_estimators][input_size]["encoder_list_to_mat_time"]+=metrics[f"encoder{clf_idx}_op_durations"]["list_to_mat"]
                                                                data_d[str(n_models)+"_models_"+str(sim_thr)+"_sim_thr"][n_estimators][input_size]["encoder_gen_mat_time"]+=metrics[f"encoder{clf_idx}_op_durations"]["gen_mat"]
                                                                data_d[str(n_models)+"_models_"+str(sim_thr)+"_sim_thr"][n_estimators][input_size]["encoder_dim"]+=metrics[f"encoder{clf_idx}_op_durations"]["len(all_tags_l)"]
                                                                # data_d[str(n_models)+"_models_"+str(sim_thr)+"_sim_thr"][n_estimators][input_size]["encoder_selector_time"].append(metrics[f"encoder{clf_idx}_op_durations"]["selector"])
                                                                # data_d[str(n_models)+"_models_"+str(sim_thr)+"_sim_thr"][n_estimators][input_size]["encoder_gen_mat_time"].append(metrics[f"encoder{clf_idx}_op_durations"]["gen_mat"])
                                                                # data_d[str(n_models)+"_models_"+str(sim_thr)+"_sim_thr"][n_estimators][input_size]["encoder_gen_mapping_time"].append(metrics[f"encoder{clf_idx}_op_durations"]["gen_mapping"])
                                                            # data_d[str(n_models)+"_models_"+str(sim_thr)+"_sim_thr"][n_estimators][input_size]["total_encoder_time"]+=metrics["total_encoder_time"]
                                                            # data_d[str(n_models)+"_models_"+str(sim_thr)+"_sim_thr"][n_estimators][input_size]["total_encoder_time"].append(metrics["total_encoder_time"])
                                                            for clf_idx in range(n_models):
                                                                data_d[str(n_models)+"_models_"+str(sim_thr)+"_sim_thr"][n_estimators][input_size]["total_clf_load_time"]+=metrics[f"clf{clf_idx}_load_time"]
                                                                data_d[str(n_models)+"_models_"+str(sim_thr)+"_sim_thr"][n_estimators][input_size]["total_inference_time"]+=metrics[f"inference{clf_idx}_time"]
                                                                data_d[str(n_models)+"_models_"+str(sim_thr)+"_sim_thr"][n_estimators][input_size]["total_encoder_time"]+=metrics[f"encoder{clf_idx}_time"]
                                                                data_d[str(n_models)+"_models_"+str(sim_thr)+"_sim_thr"][n_estimators][input_size]["total_clf_time"]+=metrics[f"clf{clf_idx}_time"]
                                                            data_d[str(n_models)+"_models_"+str(sim_thr)+"_sim_thr"][n_estimators][input_size]["total_data_load_time"]+=metrics["total_data_load__time"]
                                                            data_d[str(n_models)+"_models_"+str(sim_thr)+"_sim_thr"][n_estimators][input_size]["total_clf_path_load_time"]+=metrics["total_clf_path_load__time"]
                                                            data_d[str(n_models)+"_models_"+str(sim_thr)+"_sim_thr"][n_estimators][input_size]["total_time"]+=metrics["total_time"]

                                                    except Exception as e:
                                                        print(e)
                                                else:
                                                    print(filename)

                                                    
for nmodel,datadict in data_d.items():
    print(nmodel, "!!!!!!!!!!!!!!!")
    for metric, value in datadict.items():
        print(metric, value)
        print()
    print()