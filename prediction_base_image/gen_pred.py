import function.main as main
import os
import json
import pickle
import time
import yaml
from tqdm import tqdm
from function.hybrid_tags import Hybrid
args = main.get_inputs()


train_tags_path = "/home/cc/Praxi-study/praxi/demos/ic2e_demo/demo_tagsets/mix_train_tag/"
test_tags_path = "/home/cc/Praxi-study/praxi/demos/ic2e_demo/demo_tagsets/large_mix_test_tag/"
# test_tags_path = "/home/cc/Praxi-study/praxi/demos/ic2e_demo/demo_tagsets/mix_test_tag/"
model_path = "/home/cc/Praxi-study/praxi_vw_debug/Praxi-Pipeline/prediction_base_image/results/pred_model.p"
modfile_path = "/home/cc/Praxi-study/praxi_vw_debug/Praxi-Pipeline/prediction_base_image/results/model.vw"
prediction_path = "/home/cc/Praxi-study/praxi_vw_debug/Praxi-Pipeline/prediction_base_image/results/test_result.txt"



train_tags = []
# print(os.listdir(train_tags_path))
for tag_file in tqdm(os.listdir(train_tags_path)):
    if(tag_file[-3:] == 'tag'):
        with open(train_tags_path + tag_file, 'rb') as tf:
            tag = yaml.load(tf, Loader=yaml.Loader)    
            train_tags.append(tag)
            # with open(train_tags_path, 'w') as tr_tags:
            #     tr_tags.write(json.dumps(tag) + '\n')

model = main.multilabel_train(train_tags, args)
modfile = model.vw_modelfile
os.popen('cp {0} {1}'.format(modfile, modfile_path))
with open(model_path, 'wb') as modfile:
    pickle.dump(model, modfile)


test_tags = []
for tag_file in tqdm(os.listdir(test_tags_path)):
    if(tag_file[-3:] == 'tag'):
        with open(test_tags_path + tag_file, 'rb') as tf:
            tag = yaml.load(tf, Loader=yaml.Loader)    
            test_tags.append(tag)
#             # with open(test_tags_path, 'w') as ts_tags:
#             #     ts_tags.write(json.dumps(tag) + '\n')
# with open(test_tags_path, 'wb') as writer:
#     pickle.dump(test_tags, writer)
# print(len(test_tags))


# with open(test_tags_path, 'rb') as reader:
#     data_loaded = pickle.load(reader)

# with open(created_tags_path, 'r') as stream:
#     for line in stream:
#         temp = json.loads(line)
#         if (type(temp) != None):
#             data_loaded.append(temp)

with open(model_path, 'rb') as reader:
    model = pickle.load(reader)
# model.vw_binary = 'docker run -v /home/cc/Praxi-study/praxi_vw_debug/Praxi-Pipeline/prediction_base_image:/workspace --rm vowpalwabbit/vw-rel-alpine:9.8.0'
model.vw_modelfile = modfile_path
print("labs",model.all_labels)
pred = main.test(model, test_tags, args)
print("output", pred)
with open(prediction_path, 'wb') as writer:
    pickle.dump(pred, writer) 
# time.sleep(5000)