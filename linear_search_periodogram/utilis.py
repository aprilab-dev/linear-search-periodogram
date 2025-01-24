"""
#################################################################################################################
                                                                                                                
                                  Periodogram based sequential phase unwrapping method function package                                                
                                                                                                             
#################################################################################################################
#-------------------------------------Python Version 1.0.0------------------------------------------------------#
# Author           : Wang Jiaxing 
# Organization     : Aprilab@NWPU
# Date             : 2024-12-11
# Description      : This document contains some experimental functions
# --------------------------------------------------------------------------------------------------------------#
# Disclaimer: This package is intended for academic use only and may not be used commercially without the permission of the author.
#----------------------------------------------------------------------------------------------#
"""

import numpy as np
from linear_search_periodogram.simulated_phase import SimulatedPhase
from linear_search_periodogram.periodogram import Periodogram

def data_collect(data, changed_param_length, V_orig_length, check_times, nifg):
    """_Parallel experimental data collation and collection_
    note:
    This function is only for data collection and collation for parallel experiments 
    using joblib, and is not a necessary function.
    You can write your own data collection functions according to your needs.

    Parameters
    ----------
    data : _dict_
        _joblib returns data after parallel processing_
    changed_param_length : _int_
        _Change the total length of the parameter_
    V_orig_length : _int_
        _Change the total length of v_
    check_times : _int_
        _Number of repetitions_
    nifg : _int_
        _The number of interferogram images_

    Returns
    -------
    success_rate : _ndarray_
        _Estimation success rate of phase unwinding parameters_
    v_est_data : _ndarray_
        _The estimate of v_
    h_est_data : _ndarray_
        _The estimate of h_
    a_est_data : _ndarray_
        _Integer ambiguity solution_
    mae_v : _ndarray_
        _The average absolute estimate error of v_
    mae_h : _ndarray_
        _The average absolute estimate error of h_
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
    """_Merge multiple dictionaries into one dictionary_

    Parameters
    ----------
    data_list : _type_
        _joblib returns a list of data after a parallel experiment_

    Returns
    -------
    data : _dict_
        _The combined data_
    """
    data = {}
    for i in range(len(data_list)):
        data.update(data_list[i])
    return data

def lab_period(param_file, check_times, change_name, value, data_id, sat_name="sentinel-1"):
    """_Laboratory experiment :Periodogram phase unwinding method_

    Parameters
    ----------
    param_file : _json_
        _Simulation satellite parameters, interference image parameters, 
          phase unwrapping algorithm parameters set_
    check_times : _int_
        _Number of repetitions_
    change_name : _str_
        _Changed parameter name_
    value : _type_
        _Change the value of the parameter_
    data_id : _int_
        _Change the index corresponding to the data for random seeds_
    sat_name : str, optional
        _Simulated SAR satellite_, by default "sentinel-1"

    Returns
    -------
    data : _dict_
        _Experimental data_
    """
    # Initializes the data space
    est_data_h = np.zeros(check_times)
    est_data_v = np.zeros(check_times)
    a_data_1st = np.zeros((check_times, param_file["Nifg"]))
    # Change parameter
    if change_name in ["velocity", "height"]:
        param_file["param_simulation"][change_name] = value
    else:
        param_file[change_name] = value

    success_time = 0
    data = {}
    # Repeated test
    for i in range(check_times):
        # Generate simulation observation phase data
        arc_phase, par2ph  = SimulatedPhase(param_file, check_times, i, sat_name).sim_arc_phase()
        # periodgram 估计v,h
        h_est, v_est = Periodogram(param_file, arc_phase, par2ph).periodogram_estimation()
        est_data_h[i] = h_est
        est_data_v[i] = v_est
        # Calculate the integer ambiguity solution
        a_data_1st[i, :] = np.round((arc_phase - v_est * par2ph["velocity"] - h_est * par2ph["height"]) / (2 * np.pi))
        # The success rate is calculated, and the success rate is determined by the error between the estimated v and h and the true value is less than a certain threshold
        if abs(h_est - param_file["param_simulation"]["height"]) < 0.5 and abs(v_est - param_file["param_simulation"]["velocity"]) < 0.0005:
            success_time += 1
    success_rate = success_time / check_times
    # Calculate the average absolute estimation error of v,h
    mae_v = np.mean(np.abs(est_data_v - param_file["param_simulation"]["velocity"]))
    mae_h = np.mean(np.abs(est_data_h - param_file["param_simulation"]["height"]))
    data[data_id] = {"success_rate": success_rate, "est_data_h": est_data_h, "est_data_v": est_data_v, "a_data_1st": a_data_1st, "mae_v": mae_v, "mae_h": mae_h}
    return data
