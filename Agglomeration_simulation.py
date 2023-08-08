#%%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsClassifier
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
#from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score, mean_squared_error
from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical
import matplotlib as mpl
#import xlrd
from sklearn.tree import DecisionTreeRegressor


#%%
# global model_name
data = pd.read_csv(r"E:\manuscipt\Agglomeration\20220419-不同地區沙子論文資料\ML_Data.csv", header=0,
                     index_col=0)

# data = data.iloc[:16, :]
# data['Heating rate'] = data['Heating rate'].apply(lambda x: float(x[:2]))
# print(data)

def normolize(x):
    x_min = x.min(0)
    x_max = x.max(0)
    x = (x - x_min) / (x_max - x_min)
    return x, (x_min, x_max)


x = data.iloc[:, :-1]
#print(x)



x, normolize_args = normolize(x)
print(x)


y = data.iloc[:, -1]
print(y)
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.85, random_state=42)



# 定义DT模型
def dt_model(params):
    max_depth, min_samples_split, min_samples_leaf, max_features = params
    model = DecisionTreeRegressor(max_depth=max_depth,
                                   min_samples_split=min_samples_split,
                                   min_samples_leaf=min_samples_leaf,
                                   max_features=max_features,
                                   random_state=42)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_train)
    r2 = r2_score(y_train, y_pred)
    mse = mean_squared_error(y_train, y_pred)
    return mse

# 定义参数空间
space = [(2, 20),  # max_depth范围
         (2, 20),  # min_samples_split范围
         (1, 20),  # min_samples_leaf范围
         (1, x_train.shape[1])]  # max_features范围

# 使用gp_minimize优化
res = gp_minimize(dt_model, space, n_calls=100)
# 输出最优参数和最小mse
print("Optimal parameters: max_depth={}, min_samples_split={}, min_samples_leaf={}, max_features={}"
      .format(*res.x))
print("Minimum MSE: {:.3f}".format(res.fun))


# 训练最终模型
final_model = DecisionTreeRegressor(max_depth=res.x[0],
                                     min_samples_split=res.x[1],
                                     min_samples_leaf=res.x[2],
                                     max_features=res.x[3],
                                     random_state=42)
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
#%% md

#%%
dim_names = ['max_depth', 'min_samples_split', 'min_samples_leaf', 'max_features']
plot_objective(result=res, dimensions=dim_names)
plt.show()
"""
#%%import
import shap
# explainer = shap.Explainer(final_model, x_train)
# shap_values = explainer(x_test)
# shap.summary_plot(shap_values, x_test, check_additivity=False)

explainer = shap.TreeExplainer(final_model)
shap_values = explainer.shap_values(x_test)
shap.summary_plot(shap_values, x_test)
#%%
explainer = shap.TreeExplainer(final_model)
shap_values = explainer.shap_values(data.iloc[:18, 1:-1].values)
shap.summary_plot(shap_values, data.iloc[:18, 1:-1].values)
#%%
shap_values = explainer.shap_values(x)
print(shap_values)
#%%
shap.initjs()
shap.plots.force( explainer.expected_value,shap_values[0])
#%%
explainer_1 = shap.TreeExplainer(final_model)
shap_values = explainer_1.shap_values(data.iloc[:18, 1:-1].values)
# 创建 Explanation 对象
explanation = shap.Explanation(values=shap_values, base_values=explainer_1.expected_value, data=x)
shap.plots.bar(explanation)
#%%
print(shap.__version__)
#%%
import shap

shap_interaction_values = shap.TreeExplainer(final_model).shap_interaction_values(data.iloc[:18, 1:-1].values)
shap.summary_plot(shap_interaction_values, data.iloc[:18, 1:-1].values)
print(shap_interaction_values)
print(shap_interaction_values.shape)
#%%
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
plt.Figure(figsize=(30,20), dpi=300)
# x=pd.DataFrame(x, columns=x.columns)
plot_tree(final_model, filled=True)
plt.show()
#%%
shap.plots.beeswarm(explanation)
#%%
import seaborn as sns
# interaction_matrix = shap.interaction(x, final_model)
explainer_1 = shap.TreeExplainer(final_model)
shap_values = explainer_1.shap_values(data.iloc[:18, 1:-1].values)

# 对于每个输入特征，计算偏差值
feature_names = ['Feature 2', 'Feature 0', 'Feature 4', 'Feature 14']
fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(10, 8))
for i, feature_name in enumerate(feature_names):
    pd_values = shap_values[:, i].sum(axis=0)
    pd_values = np.tile(pd_values, len(data.iloc[:18, 1:-1]))
    ax = axs[int(i/2)][i%2]
    ax.plot(data.iloc[:18, 1:-1].values[:, i], pd_values.flatten(), linewidth=2)
    ax.set_xlabel(feature_name)
    ax.set_ylabel('Partial Dependence Value')
plt.tight_layout()
plt.show()
"""