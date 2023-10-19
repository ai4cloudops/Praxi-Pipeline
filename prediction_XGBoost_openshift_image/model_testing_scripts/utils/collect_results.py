import os
from collections import defaultdict

# collect info to generalize
print('Batch: ', end='')
batch_id = input()

# Read all train data
data_d = dict()
n_samples = 25
for dataset in ["data_0"]:  # data_0 = 80 labels
    for n_jobs in [8]:      # num cores used
        for clf_njobs in [8]:   # num cores used
            for n_models, test_batch_count in zip([8, 4, 2, 1],[1, 1, 1, 1]): #([50,25,20,15,10,5,1],[1,1,1,1,1,1,8]): # ([1,25,10],[8,1,1])
                data_d[str(n_models)+"_models"] = defaultdict(list)
                for n_estimators in [100]:  # 3 num trees
                    for depth in [1]:       # depth of trees
                        for tree_method in["exact"]: # "exact","approx","hist"
                            for max_bin in [1]:     
                                for input_size, dim_compact_factor in zip([109319, 27329, 6832],[1,1,1]): # [None, 10000, 1000, 500, 5000, 15000],[1,1,1,1,1,1] # [None, 13, 106, 284, 427, 854, 1138, 1708, 3416, 6832, 13664, 27329, 54659, 109319]
                                    for i in range(n_models):
                                        for shuffle_idx in range(3):
                                            for test_sample_batch_idx in [batch_id]:
                                                filename  ="/home/cc/Praxi-Pipeline/prediction_XGBoost_openshift_image/model_testing_scripts/cwd_ML_with_"+dataset+"_"+str(n_models)+"_"+str(i)+"_train_"+str(shuffle_idx)+"shuffleidx_"+str(test_sample_batch_idx)+"testsamplebatchidx_"+str(n_samples)+"nsamples_"+str(n_jobs)+"njobs_"+str(n_estimators)+"trees_"+str(depth)+"depth_"+str(input_size)+"-"+str(dim_compact_factor)+"rawinput_sampling1_"+str(tree_method)+"treemethod_"+str(max_bin)+"maxbin_modize_par_removesharedornoisestags/metrics_init.out"
                                                if os.path.isfile(filename):
                                                    with open(filename) as file:
                                                        for line in file:
                                                            line_l = line.rstrip().split(":")
                                                            if line_l[0] == "BOW_XGB_init.fit":
                                                                data_d[str(n_models)+"_models"]["traintime"].append(float(line_l[1]))
                                                            if line_l[0] == "tagsets_to_matrix-trainset_xsize":
                                                                data_d[str(n_models)+"_models"]["samplesize"].append(float(line_l[1]))
                                                            if line_l[0] == "tagsets_to_matrix-trainset_ysize":
                                                                data_d[str(n_models)+"_models"]["dimensions"].append(float(line_l[1]))
                                                # cwd =      "/home/cc/Praxi-study/Praxi-Pipeline/prediction_XGBoost_openshift_image/model_testing_scripts/cwd_ML_with_"+dataset+"_"+str(n_models)+"_train_"+str(shuffle_idx)+"_shuffleidx_"+str(test_sample_batch_idx)+"testsamplebatchidx_"+str(n_samples)+"nsamples_"+str(n_jobs)+"njobs_"+str(clf_njobs)+"clfnjobs_"+str(n_estimators)+"trees_"+str(depth)+"depth_"+str(input_size)+"-"+str(dim_compact_factor)+"rawinput_sampling1_"+str(tree_method)+"treemethod_"+str(max_bin)+"maxbin_modize_par_removesharedornoisestags/"
                                                filename  ="/home/cc/Praxi-Pipeline/prediction_XGBoost_openshift_image/model_testing_scripts/cwd_ML_with_"+dataset+"_"+str(n_models)+"_train_"+str(shuffle_idx)+"shuffleidx_"+str(test_sample_batch_idx)+"testsamplebatchidx_"+str(n_samples)+"nsamples_"+str(n_jobs)+"njobs_"+str(clf_njobs)+"clfnjobs_"+str(n_estimators)+"trees_"+str(depth)+"depth_"+str(input_size)+"-"+str(dim_compact_factor)+"rawinput_sampling1_"+str(tree_method)+"treemethod_"+str(max_bin)+"maxbin_modize_par_removesharedornoisestags/metrics_pred.out"
                                                if os.path.isfile(filename):
                                                    try:
                                                        with open(filename, "r") as f:
                                                            for line in f:
                                                                line_l = line.rstrip().split()
                                                                if len(line_l) == 9:
                                                                    data_d[str(n_models)+"_models"]["f1"].append((float(line_l[3])))
                                                    except Exception as e:
                                                        print(e)

print('##############################################')
print('#    Batch '+batch_id)
print('##############################################')
for nmodel,datadict in data_d.items():
    print('#'+nmodel+' models')
    for metric, value in datadict.items():
        print(metric+'_b'+batch_id+'_m'+nmodel,'=',value)
    print()
