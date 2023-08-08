
import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score, mean_squared_error
from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical
import matplotlib as mpl


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

# # 定义SVR模型
# def svr_model(params):
#     C, epsilon, gamma, kernel, degree, coef0, shrinking, tol, max_iter = params
#     model = SVR(C=C, epsilon=epsilon, gamma=gamma, kernel=kernel, degree=degree,
#                 coef0=coef0, shrinking=shrinking, tol=tol, max_iter=max_iter)
#     model.fit(x_train, y_train)
#     y_pred = model.predict(x_train)
#     r2 = r2_score(y_train, y_pred)
#     mse = mean_squared_error(y_train, y_pred)
#     return mse
#
# # 定义参数空间
# space = [(1e-3, 1e3, 'log-uniform'),  # C值范围
#          (1e-3, 1e3, 'log-uniform'),  # epsilon范围
#          (1e-6, 1e1, 'log-uniform'),  # gamma范围
#          ('linear', 'poly', 'rbf', 'sigmoid'),  # kernel类型
#          (1, 5),  # degree范围，只有在kernel为poly时才有效
#          (-1.0, 1.0),  # coef0范围，只有在kernel为poly或sigmoid时才有效
#          (True, False),  # 是否使用启发式方法进行支持向量的筛选
#          (1e-6, 1e-2, 'log-uniform'),  # tol范围
#          (100, 10000)]  # max_iter范围
#
# # 使用gp_minimize优化
# res = gp_minimize(svr_model, space, n_calls=100)
#
# # 输出最优参数和最小mse
# print("Optimal parameters: C={}, epsilon={}, gamma={}, kernel={}, degree={}, coef0={}, shrinking={}, tol={}, max_iter={}"
#       .format(*res.x))
# print("Minimum MSE: {:.3f}".format(res.fun))
#
# # 训练最终模型
# final_model = SVR(C=res.x[0], epsilon=res.x[1], gamma=res.x[2], kernel=res.x[3],
#                   degree=res.x[4], coef0=res.x[5], shrinking=res.x[6], tol=res.x[7], max_iter=res.x[8])
# final_model.fit(x_train, y_train)
#
# # 预测训练数据并计算r2和mse
# y_train_pred = final_model.predict(x_train)
# r2 = r2_score(y_train, y_train_pred)
#
#
# # 计算R2和MSE
# r2_train = r2_score(y_train, y_train_pred)
# mse_train = mean_squared_error(y_train, y_train_pred)
# y_test_pred = final_model.predict(x_test)
#
# r2_test = r2_score(y_test, y_test_pred)
# mse_test = mean_squared_error(y_test, y_test_pred)
#
# # 打印出训练值和预测值
# print("Training values: ", y_train)
# print("Training predictions: ", y_train_pred)
# print("Testing values: ", y_test)
# print("Testing predictions: ", y_test_pred)
#
# # 打印出R2和MSE
# print("R2 (train): ", r2_train)
# print("MSE (train): ", mse_train)
# print("R2 (test): ", r2_test)
# print("MSE (test): ", mse_test)
#%%
from skopt.plots import plot_objective
import matplotlib.pyplot as plt
#dim_names = ['C', 'epsilon', 'gamma','degree', 'coef0', 'shrinking','max_iter',]
# plot_objective(res)
# plt.show()
# #%%
# #import matplotlib.pyplot as plt
# plt.savefig(r'C:\Users\xinmatrix\OneDrive - Nanyang Technological University\Desktop\Picture\objective_plot_SVR.png', format='png', dpi=300)
# #%%
# dim_names = ['C', 'epsilon', 'gamma', 'kernel','degree', 'coef0', 'shrinking', 'tol', 'max_iter']
# plot_objective(result=res, dimensions=dim_names)
# plt.show()

# import shap
#
# # 计算 SHAP 值
# # 定义一个可调用函数，用于进行预测
# def predict(X):
#     return final_model.predict(X)
#
# # 创建 SHAP 解释器并计算 SHAP 值
# explainer = shap.Explainer(predict, x_train)
# shap_values = explainer.shap_values(x_test)
#
# # 创建汇总图
# shap.summary_plot(shap_values, x_test)
#
# # 绘制条形图
# shap.plots.bar(shap_values, feature_names=x_train.columns)


import shap
def svr_model(params):
    C, epsilon, gamma, kernel, degree, coef0, shrinking, tol, max_iter = params
    model = SVR(C=C, epsilon=epsilon, gamma=gamma, kernel=kernel, degree=degree,
                coef0=coef0, shrinking=shrinking, tol=tol, max_iter=max_iter)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_train)
    r2 = r2_score(y_train, y_pred)
    mse = mean_squared_error(y_train, y_pred)
    return mse

space = [(1e-3, 1e3, 'log-uniform'),  # C值范围
         (1e-3, 1e3, 'log-uniform'),  # epsilon范围
         (1e-6, 1e1, 'log-uniform'),  # gamma范围
         ('linear', 'poly', 'rbf', 'sigmoid'),  # kernel类型
         (1, 5),  # degree范围，只有在kernel为poly时才有效
         (-1.0, 1.0),  # coef0范围，只有在kernel为poly或sigmoid时才有效
         (True, False),  # 是否使用启发式方法进行支持向量的筛选
         (1e-6, 1e-2, 'log-uniform'),  # tol范围
         (100, 10000)]  # max_iter范围

# 使用gp_minimize优化
res = gp_minimize(svr_model, space, n_calls=100)
#
# # 定义 SVR 模型
# def svr_model(params, x_train, y_train, x_test):
#     C, epsilon, gamma, kernel, degree, coef0, shrinking, tol, max_iter = params
#     model = SVR(C=C, epsilon=epsilon, gamma=gamma, kernel=kernel, degree=degree,
#                 coef0=coef0, shrinking=shrinking, tol=tol, max_iter=max_iter)
#     model.fit(x_train, y_train)
#     y_pred = model.predict(x_test)
#     return model, y_pred
#
# # 使用最优参数训练最终模型
# final_model, y_pred = svr_model(res.x, x_train, y_train, x_test)
#
# # 创建 SHAP 解释器并计算 SHAP 值
# explainer = shap.Explainer(final_model)
# shap_values = explainer.shap_values(x_test)
#
# # 绘制条形图
# shap.plots.bar(shap_values, feature_names=x_test.columns)
import shap

# 定义 SVR 模型
final_model = SVR(C=res.x[0], epsilon=res.x[1], gamma=res.x[2], kernel=res.x[3],
                  degree=res.x[4], coef0=res.x[5], shrinking=res.x[6], tol=res.x[7], max_iter=res.x[8])
final_model.fit(x_train, y_train)

# 创建 SHAP 解释器并计算 SHAP 值
explainer = shap.KernelExplainer(final_model.predict, x_train)
shap_values = explainer.shap_values(x_train)


shap_values = explainer.shap_values(x_train)
# # 创建汇总图
# shap.summary_plot(shap_values, x_train)

feature_names = ['SiO2', 'Al2O3', 'K2O',
'Fe2O3', 'Na2O', 'CaO',
'TiO2', 'MgO', 'BaO',
'P2O5', 'SO3', 'ZrO2',
'SrO', 'MnO', 'CuO',
'Particle size',
                 ]

import shap

# 创建 SHAP 解释器并计算 SHAP 值
explainer = shap.KernelExplainer(final_model.predict, x_train)
shap_values = explainer.shap_values(x_train)

# 获取特征名列表
# feature_names = x_test.columns.tolist()

# 绘制条形图
# shap.plots.bar(shap_values, max_display=10, feature_names=feature_names)
# 绘制条形图
shap.summary_plot(shap_values, x_train, plot_type="bar", max_display=10)
# # 绘制条形图
# shap.plots.bar(shap_values, feature_names=feature_names)



