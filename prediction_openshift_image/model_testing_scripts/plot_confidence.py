import matplotlib.pyplot as plt
import numpy as np
import yaml


def plot(idx, data_l, xlabel_l, yhats):
    cmap = plt.get_cmap('gist_rainbow')
    color_l = [cmap(i) for i in np.linspace(0, 1, 100)]

    fig, ax = plt.subplots(1, 1, figsize=(26, 6), dpi=600)
    # proba_array = proba_array.reshape(-1)
    c_l = [color_l[cluster_idx] for cluster_idx in yhats]
    bar_plots = ax.bar(list(range(len(data_l))), data_l, color=c_l)
    # ax.set_xlim(-2, len(data_l)+1)
    ax.set_xticks(list(range(len(data_l))))
    ax.set_xticklabels(xlabel_l, rotation=90)
    ax.set_title('Cost Plot', fontdict={'fontsize': 30, 'fontweight': 'medium'})
    ax.set_xlabel("label idx", fontdict={'fontsize': 26})
    ax.set_ylabel("Cost", fontdict={'fontsize': 26})
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.tick_params(axis='both', which='minor', labelsize=10)
    ax.bar_label(bar_plots, labels=yhats, fontsize=10)
    # ax.vlines(x=biggest_yhat_idx-0.5, ymin=min(proba_array), ymax=max(proba_array), color='black')
    plt.savefig('./results/figs/sample_'+str(idx)+'.png', bbox_inches='tight')
    plt.close()


def read_pred_output_csoaa_thres():
    with open("/home/cc/Praxi-study/Praxi-Pipeline/prediction_openshift_image/model_testing_scripts/cwd/results/label_table-multi.yaml", "r") as datafile:
        label_table = yaml.load(datafile, Loader=yaml.Loader)
    with open('/home/cc/Praxi-study/Praxi-Pipeline/prediction_openshift_image/model_testing_scripts/cwd/results/pred_true.txt', 'r') as datatile:
        y_true = yaml.safe_load(datatile)
    predfilepath = "/home/cc/Praxi-study/Praxi-Pipeline/prediction_openshift_image/model_testing_scripts/cwd/results/pred_output-multi.txt"
    thr = 0.9

    with open(predfilepath, "r") as datafile:
        for idx, line in enumerate(datafile):
            # pred_l = []
            pred_entry_l = line.split(" ")
            # pred_pair = pred_entry_l.split(":")
            pred_d = {label_table[int(pred_entry.split(":")[0])]: float(pred_entry.split(":")[1].strip("\n")) for pred_entry in pred_entry_l}
            pred_d = {k: v for k, v in sorted(pred_d.items(), key=lambda item: item[1])}
            yhats = [0 if v > thr else 1 for _, v in pred_d.items()]

            pred_key_l, y_true_idx = list(pred_d.keys()), []
            for i, key in enumerate(pred_key_l):
                if key in y_true[idx]:
                    y_true_idx.append(i)
            for i in y_true_idx:
                pred_key_l[i] = pred_key_l[i]+"*"
            plot(idx, list(pred_d.values()), pred_key_l, yhats)

def read_pred_output_csoaa_ntag():
    with open("/home/ubuntu/praxi/demos/ic2e_demo/results/label_table-iterative.yaml", "r") as datafile:
        label_table = yaml.load(datafile)
    with open('./results/pred_true.txt', 'r') as datatile:
        y_true = yaml.safe_load(datatile)
    predfilepath = "/home/ubuntu/praxi/demos/ic2e_demo/results/pred_output-iterative.txt"
    ntagfilepath = "/home/ubuntu/praxi/demos/ic2e_demo/results/pred_ntag-iterative.txt"

    with open(ntagfilepath, "r") as datafile:
        ntag_l = []
        for idx, line in enumerate(datafile):
            ntag_l.append(int(line))
    with open(predfilepath, "r") as datafile:
        for idx, line in enumerate(datafile):
            # pred_l = []
            pred_entry_l = line.split(" ")
            # pred_pair = pred_entry_l.split(":")
            pred_d = {label_table[int(pred_entry.split(":")[0])]: float(pred_entry.split(":")[1].strip("\n")) for pred_entry in pred_entry_l}
            pred_d = {k: v for k, v in sorted(pred_d.items(), key=lambda item: item[1])}
            yhats = [0 for _ in range(len(pred_d))]
            for i in range(ntag_l[idx]):
                yhats[i] = 1
            pred_key_l, y_true_idx = list(pred_d.keys()), []
            for i, key in enumerate(pred_key_l):
                if key in y_true[idx]:
                    y_true_idx.append(i)
            for i in y_true_idx:
                pred_key_l[i] = pred_key_l[i]+"*"
            plot(idx, list(pred_d.values()), pred_key_l, yhats)
    
if __name__ == "__main__":
    # read_pred_output_csoaa_ntag()
    read_pred_output_csoaa_thres()
    

