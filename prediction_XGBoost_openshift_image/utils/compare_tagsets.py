import pickle





def union_tags(index_tag_mapping_path_l):
    all_tags_set = set()
    all_tags_list = list()
    for index_tag_mapping_path in index_tag_mapping_path_l:
        with open(index_tag_mapping_path, 'rb') as fp:
            temp_all_tags_list = pickle.load(fp)
            all_tags_list.extend(temp_all_tags_list)
            all_tags_set.update(set(temp_all_tags_list))
    return all_tags_set, all_tags_list


if __name__ == "__main__":
    index_tag_mapping_path_l_1 = [
        "/home/cc/Praxi-study/Praxi-Pipeline/prediction_XGBoost_openshift_image/model_testing_scripts/100_estimator_1_depth/cwd_ML_with_data_0_8_0_train/index_tag_mapping",
        "/home/cc/Praxi-study/Praxi-Pipeline/prediction_XGBoost_openshift_image/model_testing_scripts/100_estimator_1_depth/cwd_ML_with_data_0_8_1_train/index_tag_mapping",
        "/home/cc/Praxi-study/Praxi-Pipeline/prediction_XGBoost_openshift_image/model_testing_scripts/100_estimator_1_depth/cwd_ML_with_data_0_8_2_train/index_tag_mapping",
        "/home/cc/Praxi-study/Praxi-Pipeline/prediction_XGBoost_openshift_image/model_testing_scripts/100_estimator_1_depth/cwd_ML_with_data_0_8_3_train/index_tag_mapping",
        "/home/cc/Praxi-study/Praxi-Pipeline/prediction_XGBoost_openshift_image/model_testing_scripts/100_estimator_1_depth/cwd_ML_with_data_0_8_4_train/index_tag_mapping",
        "/home/cc/Praxi-study/Praxi-Pipeline/prediction_XGBoost_openshift_image/model_testing_scripts/100_estimator_1_depth/cwd_ML_with_data_0_8_5_train/index_tag_mapping",
        "/home/cc/Praxi-study/Praxi-Pipeline/prediction_XGBoost_openshift_image/model_testing_scripts/100_estimator_1_depth/cwd_ML_with_data_0_8_6_train/index_tag_mapping",
        "/home/cc/Praxi-study/Praxi-Pipeline/prediction_XGBoost_openshift_image/model_testing_scripts/100_estimator_1_depth/cwd_ML_with_data_0_8_7_train/index_tag_mapping",
    ]
    all_tags_set_1, all_tags_list_1 = union_tags(index_tag_mapping_path_l_1)

    index_tag_mapping_path_l_2 = [
        "/home/cc/Praxi-study/Praxi-Pipeline/prediction_XGBoost_openshift_image/model_testing_scripts/100_estimator_1_depth/cwd_ML_with_data_0_4_0_train/index_tag_mapping",
        "/home/cc/Praxi-study/Praxi-Pipeline/prediction_XGBoost_openshift_image/model_testing_scripts/100_estimator_1_depth/cwd_ML_with_data_0_4_1_train/index_tag_mapping",
        "/home/cc/Praxi-study/Praxi-Pipeline/prediction_XGBoost_openshift_image/model_testing_scripts/100_estimator_1_depth/cwd_ML_with_data_0_4_2_train/index_tag_mapping",
        "/home/cc/Praxi-study/Praxi-Pipeline/prediction_XGBoost_openshift_image/model_testing_scripts/100_estimator_1_depth/cwd_ML_with_data_0_4_3_train/index_tag_mapping",
    ]
    all_tags_set_2, all_tags_list_2 = union_tags(index_tag_mapping_path_l_2)

    index_tag_mapping_path_l_3 = [
        "/home/cc/Praxi-study/Praxi-Pipeline/prediction_XGBoost_openshift_image/model_testing_scripts/cwd_ML_with_data_0_1_0_train/index_tag_mapping"
    ]
    all_tags_set_3, all_tags_list_3 = union_tags(index_tag_mapping_path_l_3)

    print(all_tags_set_3.difference(all_tags_set_1))