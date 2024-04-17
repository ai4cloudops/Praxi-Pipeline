import os
import datetime
from collections import Counter
import matplotlib.pyplot as plt
import tarfile

def get_file_mod_times_from_tar(tar_path):
    """ Extract file modification times from a tar.gz archive. """
    file_mod_times = {}
    with tarfile.open(tar_path, "r:gz") as tar:
        for tarinfo in tar:
            if tarinfo.isfile() and 'cache' not in tarinfo.name and "usr/local/lib/python3.9/site-packages" in tarinfo.name:
                # mod_time = datetime.datetime.fromtimestamp(tarinfo.mtime)
                file_mod_times[tarinfo.name] = datetime.datetime.fromtimestamp(tarinfo.mtime)
    return file_mod_times

def get_file_mod_times(directory_path):
    """ Recursively find all files and get their modification times. """
    file_mod_times = {}
    for dirpath, dirnames, filenames in os.walk(directory_path):
        for filename in filenames:
            if "distutils-precedence" not in dirpath and "setuptools" not in dirpath and "pip" not in dirpath and "pkg_resources" not in dirpath and "_distutils_hack" not in dirpath and "distutils-precedence.pth" not in filename:
                file_path = os.path.join(dirpath, filename)
                mod_time = os.path.getmtime(file_path)
                file_mod_times[file_path] = datetime.datetime.fromtimestamp(mod_time)
    return file_mod_times

def plot_modification_times(file_mod_times, filename):
    """ Plot the number of files modified at each exact timestamp. """
    # Convert timestamps to datetime objects for exact times
    # exact_times = [datetime.datetime.fromtimestamp(t) for t in file_mod_times.values()]
    exact_times = list(file_mod_times.values())
    exact_times.sort()  # Sort to ensure the first modification time is the earliest

    # Normalize times by subtracting the first modification time
    first_time = exact_times[0]
    normalized_times = [(t - first_time).total_seconds() for t in exact_times]  # Normalize to minutes

    # Count how many times each exact timestamp appears
    time_counts = Counter(normalized_times)

    # Preparing data for plotting
    times = list(time_counts.keys())
    counts = list(time_counts.values())

    # Sorting the data by time
    times, counts = zip(*sorted(zip(times, counts)))

    # Plotting
    plt.figure(figsize=(12, 6))
    p = plt.bar(times, counts, width=0.7, align='center')  # Using a very small width for precise display
    plt.bar_label(p, label_type='edge', fontsize=18)
    plt.xlabel('Modification Time (Normalized Second)', fontsize=20)
    plt.ylabel('Number of Files Modified', fontsize=20)
    # plt.title('File Modifications Over Time')
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.tight_layout()
    # plt.show()
    plt.savefig(fig_path+filename+'.pdf', bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    fig_path = "/home/cc/Praxi-study/Praxi-Pipeline/prediction_XGBoost_openshift_image/figs/"

    directory_path = "/home/cc/Praxi-study/data_gen_venv/venv/lib/python3.10/site-packages"
    file_mod_times = get_file_mod_times(directory_path)
    plot_modification_times(file_mod_times, "VM_mtime")


    # tar_path = '/home/cc/Praxi-study/Praxi-Pipeline/get_layer_changes/cwd/zongshun96_python3_9-slim-bullseye.plotly_v5_18_0-contourpy_v1_2_0/sha256_eb8028260c64fb58096eb7e2a28b68f5314474551c0cf295341356800bfd5228.tar.gz'
    # file_mod_times = get_file_mod_times_from_tar(tar_path)
    # plot_modification_times(file_mod_times, "container_mtime")



# temp_d = dict(sorted(file_mod_times.items(), key=lambda item: item[1], reverse=False))
# first_time = list(temp_d.values())[0]
# for k in temp_d:
#     temp_d[k] = (temp_d[k] - first_time).total_seconds()
#     if "packaging" in k:
#         print(temp_d[k], k)
# print(temp_d)
# packages_l = dict()
# for file in temp_d.keys():
#     packages_l[file.split("/")[5]] = 0
# print(packages_l)