"""
#################################################################################################################
                                                                                                                
                                  基于Periodogram的时序相位解缠方法函数包                                                 
                                                                                                             
#################################################################################################################
#-------------------------------------Python Version 1.0.0------------------------------------------------------#
# Author           : 王家兴 
# Organization     : Aprilab@NWPU
# Date             : 2024-12-11
# Description      : 本函数包包含了基于Periodogram的InSAR时序相位解缠方法函数包,
#                    包括时序InSAR观测相位仿真模拟、参数估计、实验结果整合等函数。
# --------------------------------------------------------------------------------------------------------------#
#
# 参考文献：
    [1] [](https://www.tudelft.nl/citg/over-faculteit/afdelingen/geoscience-remote-sensing/research/lambda/lambda)下载LAMBDA software package Python implementation,Version 1.0
    [2] Ferretti A, Prati C, Rocca F. Permanent scatterers in sar interferometry[J]. IEEE Transactions on geoscience and remote sensing, 2001, 39(1): 8-20.
    [3] Van Leijen F J. Persistent scatterer interferometry based on geodetic estimation theory[D]. Delft University of Technology, 2014.
    [4] Czikhardt R, Van Der Marel H, Papco J. Gecoris: An open-source toolbox for analyzingtime series of corner reflectors in insar geodesy[J]. Remote Sensing, 2021, 13(5): 926.
#
# ---------------------------------------------------------------------------------------------#
# 声明：本程序包是仅供学术研究使用，未经作者允许，不得用于商业用途。
#----------------------------------------------------------------------------------------------#
"""

import numpy as np
from component.simulated_phase import SimulatedPhase
from component.periodogram import Periodogram

def data_collect(data, changed_param_length, V_orig_length, check_times, nifg):
    """_并行实验数据整理与收集_
    note:
    本函数只是针对使用joblib进行并行实验的数据收集和整理,不是必要函数。
    你可以根据自己的需求编写自己的数据收集函数。

    Parameters
    ----------
    data : _dict_
        _joblib并行处理后返回的数据_
    changed_param_length : _int_
        _改变的参数总长度_
    V_orig_length : _int_
        _改变的v的总长度_
    check_times : _int_
        _重复试验次数_
    nifg : _int_
        _干涉图影像数量_

    Returns
    -------
    success_rate : _ndarray_
        _相位解缠参数估计成功率_
    v_est_data : _ndarray_
        _v的估计值_
    h_est_data : _ndarray_
        _h的估计值_
    a_est_data : _ndarray_
        _整数模糊度解_
    mae_v : _ndarray_
        _v的平均绝对估计误差_
    mae_h : _ndarray_
        _h的平均绝对估计误差_
    """
    success_rate = np.zeros((changed_param_length, V_orig_length))
    mae_v = np.zeros((changed_param_length, V_orig_length))
    mae_h = np.zeros((changed_param_length, V_orig_length))
    v_est_data = []
    h_est_data = []
    a_est_data = []
    for k in range(changed_param_length):
        for i in range(V_orig_length):
            success_rate[k][i] = data[k][i]["success_rate"]
            mae_v[k][i] = data[k][i]["mae_v"]
            mae_h[k][i] = data[k][i]["mae_h"]
            v_est_data.append(data[k][i]["est_data_v"].reshape(1, -1))
            h_est_data.append(data[k][i]["est_data_h"].reshape(1, -1))
            a_est_data.append(data[k][i]["a_data_1st"])
    v_est_data = np.concatenate(v_est_data, axis=0)
    h_est_data = np.concatenate(h_est_data, axis=0)
    a_est_data = np.concatenate(a_est_data, axis=0).reshape(changed_param_length, V_orig_length, check_times, nifg)
    return success_rate, v_est_data, h_est_data, a_est_data, mae_v, mae_h

def dict_collect(data_list):
    """_将多个字典合并为一个字典_

    Parameters
    ----------
    data_list : _type_
        _joblib并行实验后,返回的数据列表_

    Returns
    -------
    data : _dict_
        _合并后的数据_
    """
    # 将列中的字典合并
    data = {}
    for i in range(len(data_list)):
        data.update(data_list[i])
    return data

def lab_period(param_file, check_times, change_name, value, data_id, sat_name="sentinel-1"):
    """_实验室实验:Periodogram相位解缠方法_

    Parameters
    ----------
    param_file : _json_
        _仿真模拟的卫星参数、干涉图像参数、相位解缠算法参数设置_
    check_times : _int_
        _重复试验次数_
    change_name : _str_
        _改变的参数名称_
    value : _type_
        _改变参数对应的值_
    data_id : _int_
        _改变数据对应的索引,用于随机种子_
    sat_name : str, optional
        _模拟的SAR卫星_, by default "sentinel-1"

    Returns
    -------
    data : _dict_
        _实验数据_
    """
    # 初始化数据空间
    est_data_h = np.zeros(check_times)
    est_data_v = np.zeros(check_times)
    a_data_1st = np.zeros((check_times, param_file["Nifg"]))
    # 改变参数
    if change_name in ["velocity", "height"]:
        param_file["param_simulation"][change_name] = value
    else:
        param_file[change_name] = value
    print(param_file["param_simulation"])
    success_time = 0
    data = {}
    # 重复试验
    for i in range(check_times):
        # 生成仿真观测相位数据
        arc_phase, par2ph  = SimulatedPhase(param_file, check_times, i, sat_name).sim_arc_phase()
        # periodgram 估计v,h
        h_est, v_est = Periodogram(param_file, arc_phase, par2ph).periodogram_estimation()
        est_data_h[i] = h_est
        est_data_v[i] = v_est
        # 计算整数模糊度解
        a_data_1st[i, :] = np.round((arc_phase - v_est * par2ph["velocity"] - h_est * par2ph["height"]) / (2 * np.pi))
        # print(f"check_times={i}, h_est={h_est}, v_est={v_est}")
        # 计算成功率,成功率判定标准为估计的v,h与真实值的误差小于一定阈值
        if abs(h_est - param_file["param_simulation"]["height"]) < 0.5 and abs(v_est - param_file["param_simulation"]["velocity"]) < 0.0005:
            success_time += 1
    success_rate = success_time / check_times
    # 计算v,h的平均绝对估计误差
    mae_v = np.mean(np.abs(est_data_v - param_file["param_simulation"]["velocity"]))
    mae_h = np.mean(np.abs(est_data_h - param_file["param_simulation"]["height"]))
    data[data_id] = {"success_rate": success_rate, "est_data_h": est_data_h, "est_data_v": est_data_v, "a_data_1st": a_data_1st, "mae_v": mae_v, "mae_h": mae_h}
    return data
