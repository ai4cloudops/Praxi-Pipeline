import os, sys
sys.path.insert(1, '/home/ubuntu/Praxi-Pipeline/taggen_openshift_image/')
from columbus.columbus import columbus
import json, yaml
import pickle
import os
import time
# from function import changeset_gen

# # Load data from previous component
# with open(input_args_path, 'rb') as in_argfile:
#     user_in = pickle.load(in_argfile)
with open("/home/ubuntu/Praxi-Pipeline/get_layer_changes/cwd/changesets/changesets_l", 'rb') as in_changesets_l:
    changesets_l = pickle.load(in_changesets_l)
                            
# Tagset Generator
tagsets_l = []
for changeset in changesets_l:
    # tags = tagset_gen.get_columbus_tags(changeset['changes'])
    tag_dict = columbus(changeset['changes'], freq_threshold=2)
    tags = ['{}:{}'.format(tag, freq) for tag, freq in tag_dict.items()]
    cur_dict = {'labels': changeset['labels'], 'tags': tags}
    tagsets_l.append(cur_dict)

# Debug
with open("/home/ubuntu/Praxi-Pipeline/taggen_openshift_image/cwd/changesets_logging", 'w') as writer:
    for change_dict in changesets_l:
        writer.write(json.dumps(change_dict) + '\n')
# with open("/home/ubuntu/Praxi-Pipeline/taggen_openshift_image/cwd/tagsets_logging", 'w') as writer:
#     for tag_dict in tagsets_l:
#         writer.write(json.dumps(tag_dict) + '\n')
for ind, tag_dict in enumerate(tagsets_l):
    with open("/home/ubuntu/Praxi-Pipeline/taggen_openshift_image/cwd/tagsets_logging"+str(ind)+".tag", 'w') as writer:
        # writer.write(yaml.dumps(tag_dict) + '\n')
        yaml.dump(tag_dict, writer, default_flow_style=False)
# time.sleep(5000)

# # Pass data to next component
with open("/home/ubuntu/Praxi-Pipeline/taggen_openshift_image/cwd/tagsets_l", 'wb') as writer:
    # for tag_dict in tag_dict_gen:
    #     writer.write(json.dumps(tag_dict) + '\n')
    pickle.dump(tagsets_l, writer)
# with open(output_args_path, 'wb') as argfile:
#     pickle.dump(user_in, argfile)