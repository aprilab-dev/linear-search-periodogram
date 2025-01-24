import numpy as np
import json

import linear_search_periodogram.utilis as utilis
import time
from joblib import Parallel, delayed


# ================================================================================================
# An experiment on the results of parameter estimation for Linear Period
# ================================================================================================
# Experimental content: Change the deformation rate v and height h of the simulation, observe the MAE of the parameter estimation results of Linear Period
##——Initial parameter setting——
##1.Read the parameter file: parms_period_ils.json, which mainly includes the satellite parameters of simulation, the parameters of interference image, and the Linear Period search parameter setting
##2.Set the range V_orig for the analog deformation rate v and H_orig for the height h
##3.Set the number of iterations: iterative_times, the number of iterations used in Linear Period. The default value is 0, that is, no iteration. For details of the iterative algorithm, see Section 4.4 and experimental results


# Experimental code
# Read parameter file
with open("params.json") as f:
    param = json.load(f)

# Set the simulation deformation rate V range and height h range
V_orig = np.round(np.arange(-50, 51, 1) * 0.001, 3)
H_orig = np.arange(-20, 21, 1)

# Set the number of repetitions
check_time = 500
# Defines a storage variable to store the results of an experiment
data_all = {}
T1 = time.time()
# Traverse the set range of height h, conduct experiments, and obtain experimental results at different heights h and different v
for k, h in enumerate(H_orig):
    t1 = time.time()
    print(f"height={h} start")
    # Set the simulation height h
    param["param_simulation"]["height"] = h
    # The experimental results under different deformation rates v were calculated in parallel
    data = Parallel(n_jobs=30)(
        delayed(utilis.lab_period)(param, check_times=check_time, change_name="velocity", value=v, data_id=i, sat_name="sentinel-1") for i, v in enumerate(V_orig)
    )
    # Store experimental results
    data_all[k] = utilis.dict_collect(data)
    t2 = time.time()
    print(f"height={h} done", f"Time: {t2 - t1} s")

T2 = time.time()
print(f"Time: {T2 - T1} s")
# The estimation results of MAE and other parameters are obtained by integrating the experimental results
success_rate, v_est_data, h_est_data, a_est1st, mae_v, mae_h = utilis.data_collect(data_all, len(H_orig), len(V_orig), check_times=check_time, nifg=param["Nifg"])

# Save the experiment results locally
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
