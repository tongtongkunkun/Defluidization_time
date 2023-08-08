
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from skopt import gp_minimize
from skopt.space import Real, Integer
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score, mean_squared_error
from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical
import matplotlib as mpl
import matplotlib.pyplot as plt

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



# 定义目标函数（即模型的评估函数）
def objective(params):
    n_estimators = params[0]
    max_depth = params[1]
    min_samples_split = params[2]
    min_samples_leaf = params[3]
    max_features = params[4]



    # 创建随机森林回归模型
    rf = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth,
                                min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
                               max_features=max_features)

    # 训练模型
    rf.fit(x_train, y_train)

    # 在训练集上进行预测
    y_train_pred = rf.predict(x_train)

    # 在测试集上进行预测
    y_test_pred = rf.predict(x_test)

    # 计算R2和MSE
    r2 = r2_score(y_test, y_test_pred)
    mse = mean_squared_error(y_test, y_test_pred)

    return mse


#%% md

#%%
# 定义超参数的搜索空间

space = [Integer(10, 200, name='n_estimators'),
         Integer(2, 650, name='max_depth'),
         Integer(2, 10, name='min_samples_split'),
         Integer(1, 10, name='min_samples_leaf'),
         Real(0.1, 1.0, name='max_features'),
        ]

# 进行超参数优化
res = gp_minimize(objective, space, n_calls=50, random_state=0)

# 输出最优参数和最小均方误差
print("Best parameters: ", res.x)
print("Minimum MSE: ", res.fun)

# 创建随机森林回归模型
rf = RandomForestRegressor(n_estimators=res.x[0], max_depth=res.x[1],
                            min_samples_split=res.x[2], min_samples_leaf=res.x[3],
                            max_features=res.x[4])

# 训练模型
rf.fit(x_train, y_train)

# 在训练集上进行预测
y_train_pred = rf.predict(x_train)

# 在测试集上进行预测
y_test_pred = rf.predict(x_test)

# 计算R2和MSE
r2_train = r2_score(y_train, y_train_pred)
mse_train = mean_squared_error(y_train, y_train_pred)
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

#%% md

#%% md

#%%
from skopt.plots import plot_objective


plot_objective(res)
rf = RandomForestRegressor(n_estimators=res.x[0], max_depth=res.x[1],
                            min_samples_split=res.x[2], min_samples_leaf=res.x[3],
                            max_features=res.x[4])
dim_names = ['n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leaf', 'max_features']
plot_objective(result=res, dimensions=dim_names)
plt.show()

