# linear search periodogram method

## 1. Operating environment
Configure the virtual environment using miniconda
[python:3.11 ,numpy:1.24.2, scipy:1.11.2,joblib:1.3.2]


# 2. linear_search_periodogram folder
## 2.1 periodogram.py 
The Periodogram class implements two different periodogram algorithms, linear search method and graph search method, which can be obtained by modifying the params.json.
## 2.2 simulated_phase.json文件
The SimulatedPhase class implements the generation of an observed phase using satellite data.
## 2.3 utilis.py file
utilis contains some functions needed for parallel simulation and a basic Periodogram function.


# 3. simulation.py

This experimental script mainly demonstrates the use of the Periodogram algorithm to change the deformation rates v and h in the simulated observation phase, and repeat the experiment multiple times for each group (v, h) to obtain the results of parameter estimation (success rate, MAE, etc.).  
Parallel computing is used to accelerate the experiment. Users can make reasonable adjustments according to their own experimental parameters (number of images, search parameters, etc.)  

# 4 parms.json file

```python
{
    "sentinel-1":{
        "Bn": 150,
        "revisit_cycle": 12,
        "H": 693000,
        "incidence_angle": 38,
        "wavelength": 0.056
    },
    "Nifg": 30,                # Number of Interference 
    "param_simulation": {
        "height": 10,
        "velocity": 0.01
    },
    "noise_level": 10, 
    "search_range": {
        "h_range": [-20 ,20],
        "v_range": [-1450,1450],
        "h_step": 1,
        "v_step": 0.0001
    },
    "period_est_mode": "linear-period",   # or grid-period
    "iterative_times": 1                  
}
```
# 3.Notes
1. The n_job of joblib parallel computing represents the number of parallel operations, which is related to the number of CPU cores. Please set it reasonably according to the number of CPU cores at hand.

2. The core modules in the project are ```class Periodogram and class SimulatedPhase```, and you can design simulation experiments according to different functions.

References:  
[1] Ferretti A, Prati C, Rocca F. Permanent scatterers in sar interferometry[J]. IEEE Transactions on geoscience and remote sensing, 2001, 39(1): 8-20.  
[2] Van Leijen F J. Persistent scatterer interferometry based on geodetic estimation theory[D]. Delft University of Technology, 2014.  
[3] Czikhardt R, Van Der Marel H, Papco J. Gecoris: An open-source toolbox for analyzingtime series of corner reflectors in insar geodesy[J]. Remote Sensing, 2021, 13(5): 926.