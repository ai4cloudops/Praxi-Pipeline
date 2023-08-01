from sklearn.datasets import load_wine
a = load_wine()

import xgboost as xgb
data = xgb.DMatrix(a['data'][:, 0:6], a['target'])  # different size of features will not automatically work. You need sth like this below
data2 = xgb.DMatrix(a['data'][:, 6:], a['target'])

# ##############################
# 1. train model1 on n packages (feature set 1)
# 2. train model2 on m packages (feature set 2)
# 3. given a system, create tag set and predict results from model1 (pred1) and model2 (pred2) and take result with highest probability above some cutoff
# the tag set will be a union of feature set 1 and feature set 2
# ##############################

param = {"max_depth": 2, "eta": 1, "objective": "multi:softmax", 'num_class': 3}

num_round = 2
bst = xgb.train(param, data, num_boost_round=num_round)

#incremental training
bst2 = xgb.train(param, data2, 1, xgb_model=bst)

df1 = bst.trees_to_dataframe()
df2 = bst2.trees_to_dataframe()