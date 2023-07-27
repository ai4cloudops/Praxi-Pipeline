import os, sys
sys.path.insert(1, '/home/ubuntu/Praxi-Pipeline/prediction_base_image')
import json
import pickle
import time
import yaml
from tqdm import tqdm
import function.main as main
from function.hybrid_tags import Hybrid

# Setup
train_tags_init_path = "/home/ubuntu/Praxi-Pipeline/data/demo_tagsets_mostly_single_label/mix_train_tag_init/"
train_tags_iter_path = "/home/ubuntu/Praxi-Pipeline/data/demo_tagsets_mostly_single_label/mix_train_tag_iter/"
test_tags_path = "/home/ubuntu/Praxi-Pipeline/data/demo_tagsets_mostly_single_label/mix_test_tag/"
# test_tags_path = "/home/cc/Praxi-study/praxi/demos/ic2e_demo/demo_tagsets/mix_test_tag/"
cwd = "/home/ubuntu/Praxi-Pipeline/prediction_base_image/model_testing_scripts/cwd/"
clf_filepath = cwd+"clf.p"
vw_model_filepath = cwd+"vw_model.vw"
clf_iter_filepath = cwd+"clf_iter.p"
vw_model_iter_filepath = cwd+"vw_model_iter.vw"
prediction_path = cwd+"test_result.txt"


# Populate Args
args = main.get_inputs()
args['experiment'] = "multi"
args["outdir"] = cwd


# Init Train & Test
train_tags = []
# print(os.listdir(train_tags_path))
for tag_file in tqdm(os.listdir(train_tags_init_path)):
    if(tag_file[-3:] == 'tag'):
        with open(train_tags_init_path + tag_file, 'rb') as tf:
            tag = yaml.load(tf, Loader=yaml.Loader)    
            train_tags.append(tag)
            # with open(train_tags_path, 'w') as tr_tags:
            #     tr_tags.write(json.dumps(tag) + '\n')

# ./demo_main.py -t demo_tagsets/iter_init -s demo_tagsets/sl_test_tag -o results -i iter_model.vw -l
args["iterative"] = vw_model_filepath
clf = main.multilabel_train(train_tags, args)
with open(clf_filepath, 'wb') as clf_file:
    pickle.dump(clf, clf_file)



test_tags = []
for tag_file in tqdm(os.listdir(test_tags_path)):
    if(tag_file[-3:] == 'tag'):
        with open(test_tags_path + tag_file, 'rb') as tf:
            tag = yaml.load(tf, Loader=yaml.Loader)    
            test_tags.append(tag)

with open(clf_filepath, 'rb') as clf_file:
    clf = pickle.load(clf_file)
# model.vw_binary = 'docker run -v /home/cc/Praxi-study/vw-kubeflow-pipeline/Praxi-Pipeline/prediction_base_image:/workspace --rm vowpalwabbit/vw-rel-alpine:9.8.0'
# model.vw_modelfile = modfile_iter_path
print("labs",clf.all_labels)
pred = main.test(clf, test_tags, args)
print("output", pred)
with open(prediction_path, 'wb') as writer:
    pickle.dump(pred, writer) 
# time.sleep(5000)








# # Iter Train & Test
# train_iter_tags = []
# # print(os.listdir(train_tags_path))
# for tag_file in tqdm(os.listdir(train_tags_iter_path)):
#     if(tag_file[-3:] == 'tag'):
#         with open(train_tags_iter_path + tag_file, 'rb') as tf:
#             tag = yaml.load(tf, Loader=yaml.Loader)    
#             train_iter_tags.append(tag)
#             # with open(train_tags_path, 'w') as tr_tags:
#             #     tr_tags.write(json.dumps(tag) + '\n')

# # ./demo_main.py -t demo_tagsets/new_tagsets -s demo_tagsets/sl_test_tag -o results -p iter_model.p
# args['previous'] = clf_filepath
# # os.popen('cp {0} {1}'.format(vw_model_filepath, vw_model_iter_filepath))
# clf = main.iterative_train(train_iter_tags, args)
# with open(clf_iter_filepath, 'wb') as clf_file:
#     pickle.dump(clf, clf_file)



# test_tags = []
# for tag_file in tqdm(os.listdir(test_tags_path)):
#     if(tag_file[-3:] == 'tag'):
#         with open(test_tags_path + tag_file, 'rb') as tf:
#             tag = yaml.load(tf, Loader=yaml.Loader)    
#             test_tags.append(tag)

# with open(clf_iter_filepath, 'rb') as clf_file:
#     clf = pickle.load(clf_file)
# # model.vw_binary = 'docker run -v /home/cc/Praxi-study/vw-kubeflow-pipeline/Praxi-Pipeline/prediction_base_image:/workspace --rm vowpalwabbit/vw-rel-alpine:9.8.0'
# # model.vw_modelfile = modfile_iter_path
# print("labs",clf.all_labels)
# pred = main.test(clf, test_tags, args)
# print("output", pred)
# with open(prediction_path, 'wb') as writer:
#     pickle.dump(pred, writer) 
# # time.sleep(5000)