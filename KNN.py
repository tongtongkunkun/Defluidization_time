#%%
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score, mean_squared_error
from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
#%%

# global model_name
data = pd.read_csv(r"E:\manuscipt\Agglomeration\20220419-不同地區沙子論文資料\ML_Data.csv", header=0,
                     index_col=0)

def normolize(x):
    x_min = x.min(0)
    x_max = x.max(0)
    x = (x - x_min) / (x_max - x_min)
    return x, (x_min, x_max)

x = data.iloc[:, :-1]
#print(x)
x, normolize_args = normolize(x)
y = data.iloc[:, -1]
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.85, random_state=42)
#%%
# 定义KNN模型
def knn_model(params):
    n_neighbors, weights, p, leaf_size, algorithm = params
    model = KNeighborsRegressor(n_neighbors=n_neighbors,
                                 weights=weights,
                                 p=p,
                                 leaf_size=leaf_size,
                                 algorithm=algorithm)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_train)
    r2 = r2_score(y_train, y_pred)
    mse = mean_squared_error(y_train, y_pred)
    return mse

# 定义参数空间
space = [(1, 15),  # n_neighbors范围
         ['uniform', 'distance'],  # weights范围
         [1, 2, 3],  # p范围
         (10, 50),  # leaf_size范围
         ['ball_tree', 'kd_tree', 'brute']]  # algorithm范围

# 使用gp_minimize优化
res = gp_minimize(knn_model, space, n_calls=100)

# 输出最优参数和最小mse
print("Optimal parameters: n_neighbors={}, weights='{}', p={}, leaf_size={}, algorithm='{}'"
      .format(*res.x))
print("Minimum MSE: {:.3f}".format(res.fun))
#%%
# 训练最终模型
final_model = KNeighborsRegressor(n_neighbors=res.x[0],
                                   weights=res.x[1],
                                   p=res.x[2],
                                   leaf_size=res.x[3],
                                   algorithm=res.x[4])
final_model.fit(x_train, y_train)

# 预测训练数据并计算r2和mse
y_train_pred = final_model.predict(x_train)
r2 = r2_score(y_train, y_train_pred)


# 计算R2和MSE
r2_train = r2_score(y_train, y_train_pred)
mse_train = mean_squared_error(y_train, y_train_pred)
y_test_pred = final_model.predict(x_test)

r2_test = r2_score(y_test, y_test_pred)
mse_test = mean_squared_error(y_test, y_test_pred)

# 打印出训练值和预测值
print("Training values: ", y_train)
print("Training predictions: ", y_train_pred)
print("Testing values: ", y_test)
print("Testing predictions: ", y_test_pred)

# 打印出R2和MSE
print("R2 (train): ", r2_train)
print("MSE (train): ", mse_train)
print("R2 (test): ", r2_test)
print("MSE (test): ", mse_test)
#%%
from skopt.plots import plot_objective
# dim_names = ['n_estimators', 'max_depth', 'min_samples_split']

plot_objective(res)
#%%
dim_names = ['n_neighbors', 'weights', 'p', 'leaf_size', 'algorithm']
plot_objective(result=res, dimensions=dim_names)
plt.show()