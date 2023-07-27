import os, sys
sys.path.insert(1, '/home/ubuntu/Praxi-Pipeline/get_layer_changes/src')
import read_layered_image
import pickle
import time
import yaml
# import os
# import json

changesets_l = read_layered_image.run()
# time.sleep(5000)
# debug
for ind, changeset in enumerate(changesets_l):
    with open("/home/ubuntu/Praxi-Pipeline/get_layer_changes/cwd/changesets/changesets_l"+str(ind)+".yaml", 'w') as writer:
        # yaml.dump(changesets_l, writer)
        yaml.dump(changeset, writer, default_flow_style=False)
# pass data to next component
with open("/home/ubuntu/Praxi-Pipeline/get_layer_changes/cwd/changesets/changesets_l", 'wb') as writer:
    pickle.dump(changesets_l, writer)
# with open(args_path, 'wb') as argfile:
#     pickle.dump(user_in, argfile)