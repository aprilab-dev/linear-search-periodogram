import numpy as np
import json

import component.utilis as utilis
import time
from joblib import Parallel, delayed


# ================================================================================================
# 关于Linear-Period的参数估计结果的实验
# ================================================================================================
# 实验内容：改变仿真模拟的形变速率v和高度h，观察Linear-Period的参数估计结果的MAE
##——初始参数设置——
##1.读取参数文件：parms_period_ils.json，主要包括仿真模拟的卫星参数、干涉图像的参数、Linear-Period的搜索参数设置
##2.设置模拟形变速率v的范围V_orig和高度h的范围H_orig
##3.设置迭代次数：iterative_times，用于Linear-Period的迭代次数，默认为0，即不迭代。迭代算法详情请见论文章节4.4部分与实验结果


# 实验代码
# 读取参数文件
with open("parms_period_ils.json") as f:
    param_file = json.load(f)

# 设置仿真模拟形变速率V范围和高度h范围
V_orig = np.round(np.arange(-50, 51, 1) * 0.001, 3)
H_orig = np.arange(-20, 21, 1)
# V_orig = [0.005, 0.007]
# H_orig = [10]
# 设置参数估计方法为Linear-Period
param_file["period_est_mode"] = "linear-period"
# 设置Linear-Period迭代次数
param_file["linear-period"]["iterative_times"] = 2
print(param_file)

# 设置重复实验次数
check_time = 500
# 定义存储变量，用于存储实验结果
data_all = {}
T1 = time.time()
# 遍历设置的高度h范围，进行实验，获得不同高度h下、不同v的实验结果
for k, h in enumerate(H_orig):
    t1 = time.time()
    print(f"height={h} start")
    # 设置仿真模拟高度h
    param_file["param_simulation"]["height"] = h
    # 并行计算，计算不同形变速率v下的实验结果
    data = Parallel(n_jobs=30)(
        delayed(utilis.lab_period)(param_file, check_times=check_time, change_name="velocity", value=v, data_id=i, sat_name="sentinel-1") for i, v in enumerate(V_orig)
    )
    # 存储实验结果
    data_all[k] = utilis.dict_collect(data)
    t2 = time.time()
    print(f"height={h} done", f"Time: {t2 - t1} s")

T2 = time.time()
print(f"Time: {T2 - T1} s")
# 整合实验结果，获得MAE等参数估计结果
success_rate, v_est_data, h_est_data, a_est1st, mae_v, mae_h = utilis.data_collect(data_all, len(H_orig), len(V_orig), check_times=check_time, nifg=param_file["Nifg"])
# print(success_rate)
# print(h_est_data, v_est_data)

# 保存实验结果到本地
np.savez(
    "./lab_v_linear_2.npz",
    data_all=data_all,
    success_rate=success_rate,
    v_est_data=v_est_data,
    h_est_data=h_est_data,
    a_est1st=a_est1st,
    mae_v=mae_v,
    mae_h=mae_h,
)
