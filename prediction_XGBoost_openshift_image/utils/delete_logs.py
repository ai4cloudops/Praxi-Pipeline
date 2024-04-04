

import os, shutil, tqdm
import multiprocessing as mp
dirpath = "/home/cc/Praxi-study/Praxi-Pipeline/prediction_XGBoost_openshift_image/model_testing_scripts/"
cwds = [os.path.join(dirpath,file) for file in os.listdir(dirpath) if os.path.isdir(os.path.join(dirpath,file)) and file[:19] == 'cwd_ML_with_data_3_']
logdirs = []
for cwd in cwds:
    logdirs.extend([os.path.join(cwd,file) for file in os.listdir(cwd) if file == 'logs'])

def remove_files(logdir):
        shutil.rmtree(logdir)
        # # print(logdir)
        # for file in os.listdir(logdir):
        #     os.remove(os.path.join(logdir,file))
        # os.rmdir(logdir)


for logdir in tqdm.tqdm(logdirs):
    remove_files(logdir)


# pool = mp.Pool(processes=20)
# rets = [pool.apply_async(remove_files, args=(logdir,)) for logdir in tqdm.tqdm(logdirs)]
# rets = [ret.get() for ret in tqdm.tqdm(rets)]
# pool.close()
# pool.join()