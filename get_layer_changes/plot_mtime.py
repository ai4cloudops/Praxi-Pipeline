import os, time
import matplotlib.pyplot as plt


# layer_path = '/home/cc/Praxi-study/vw-kubeflow-pipeline/Praxi-Pipeline/get_layer_changes/image/tmp/84c06b93d0458f549f7f03b01e679d0311889e600b85e9d1a86e42760a8f521f/layer'
# mtime_log = './mtime.log'
# mtime_pdf = './mtime.pdf'
layer_path = '/home/cc/Praxi-study/data_gen_venv/venv/lib/python3.10/site-packages'
layer_path_l = ['/home/cc/Praxi-study/data_gen_venv/venv/lib/python3.10/site-packages'
                ]
mtime_log = './mtime-VM.log'
mtime_pdf = './mtime-VM.pdf'

def pick_first(ele):
    return ele[0]

# data_d = {}
data_l = []
for layer_path in layer_path_l:
    for root, dirs, files in os.walk(layer_path):
        for filename in files:
            # data_l.append((os.path.getmtime(os.path.join(root, filename)), os.path.join(root, filename)))
            if os.path.getmtime(os.path.join(root, filename)) > 1707100496:
                # doSomethingWithFile(os.path.join(root, filename))
                # print("last modified: %d" % os.path.getmtime(os.path.join(root, filename)), os.path.join(root, filename))
                data_l.append((os.path.getmtime(os.path.join(root, filename))-1707100496, os.path.join(root, filename)))
                # if len(data_l) > 2000:
                #     break
                # data_d[]
                # print("last modified: %s" % time.ctime(os.path.getmtime(os.path.join(root, filename))), os.path.join(root, filename))
    #     for dirname in dirs:
    #         doSomewthingWithDir(os.path.join(root, dirname))


data_l.sort(key=pick_first)
with open(mtime_log, 'w') as f:
    for data in data_l:
        data = [str(ele) for ele in data]
        f.write(', '.join(data))
        f.write("\n")


ts_l = [ele[0] for ele in data_l]
fig, ax = plt.subplots(1, 1, figsize=(16, 6), dpi=400)
ax.hist(ts_l, density=False, bins=1000)  # density=False would make counts
ax.set_ylabel('Probability')
ax.set_xlabel('time')
plt.savefig(mtime_pdf, format='pdf', dpi=1200, bbox_inches='tight')
