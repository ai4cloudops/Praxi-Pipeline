import xgboost as xgb
import numpy as np
from scipy.sparse import csr_matrix
import timeit

cwd_clf = "/home/cc/test/"

# Generate synthetic data
np.random.seed(42)
data_size = 672
num_features = 1815421
# X_dense = np.random.rand(data_size, num_features)
X_dense = np.zeros((data_size, num_features))
y = np.random.randint(3000, size=data_size)

# Convert dense matrix to sparse CSR format
X_sparse = csr_matrix(X_dense)

# Create an XGBoost model and train it
model = xgb.XGBClassifier(n_estimators=100, max_depth=1, learning_rate=0.1,silent=False, objective='binary:logistic', \
                        booster='gbtree', n_jobs=8, nthread=None, gamma=0, min_child_weight=1, max_delta_step=0, \
                        subsample=1, colsample_bytree=1, colsample_bylevel=1, reg_alpha=0, reg_lambda=1, tree_method="exact", max_bin=6)
# model.fit(X_dense, y)
# model.save_model(cwd+'dummy_model_zeros.json')
model.load_model(cwd_clf+'cwd_ML_with_data_4_1_0_train_0shuffleidx_0testsamplebatchidx_4nsamples_32njobs_100trees_1depth_None-1rawinput_sampling1_exacttreemethod_1maxbin_modize_par_True25removesharedornoisestags_verpak/model_init.json')
model.set_params(n_jobs=1)
# # Define a function to make predictions with dense input
# def predict_dense():
#     model.predict(X_dense)

# Define a function to make predictions with sparse input
def predict_sparse():
    model.predict(X_sparse)

# # Time inference with dense data
# dense_time = timeit.timeit(predict_dense, number=10)
# print(f"Average inference time with dense matrix: {dense_time / 10:.5f} seconds per execution")

# Time inference with sparse data
sparse_time = timeit.timeit(predict_sparse, number=10)
print(f"Average inference time with sparse matrix: {sparse_time / 10:.5f} seconds per execution")
