# Periodogram algorithm sequential phase unwinding simulation function package uses reference books

# 1.Operating environment
Configure the virtual environment using miniconda
[python:3.11 ,numpy:1.24.2, scipy:1.11.2,joblib:1.3.2]

# 2. module
## 2.2.1 periodogram.py file
The module encapsulates an object: periodogram, which contains three parts: graph search, linear search, iterative linear search
## 2.2.2 simulated_phase.json文件
Observation phase simulation module: class simulated_phase，Generate simulated phases based on satellite parameters
## 2.1 utilis.py file
utilis里包含两个模块 [peirod_ils模块](./utilis/period_ils.py)。在使用Periodogram算法和ILS、IB算法进行仿真模拟实验时，直接调用[peirod_ils模块](./utilis/period_ils.py)即可。
## 2.2.3 parms_period_ils.json file
```python
{
    "sentinel-1":{
        "Bn":150,
        "revisit_cycle": 12,
        "H":693000,
        "incidence_angle": 38,
        "wavelength":0.056
    },
    "Nifg": 30, 
    "param_simulation": {
        "height":10,
        "velocity":0.01
    },
    "noise_level": 10, 
    "search_range": {
        "h_range":[-20 ,20],
        "v_range":[-1450,1450],
        "h_step":1,
        "v_step":0.0001
    },
    "period_est_mode": "linear-period",
    "iterative_times":2
}
```
### 2.2.2 lab_line_period.py
具体使用过程参考[period_实验](./template/lab_line_period.py)
该实验脚本主要展示了使用Periodogram算法，改变模拟观测相位中的形变速率v和h,并对每组(v,h)进行多次重复实验,获得参数估计的结果(成功率、MAE等)。
**使用了并行计算加速实验,使用者可根据自己的实验改变参数(影像数量、搜索参数等)进行合理调整**
需要注意的是，```parms_period_ils.json```文件中的"period_est_mode"决定了使用的Periodogram方法，目前的```class Period_estimator```模块
包含了3种Perirogram算法,**输入对应名称字符改变使用方法**：
1.linear-period 快速的线性搜索下的Periodogram，可以通过控制"linear-period":{"mode":"iterative","iterative_times":2}来选择是否迭代估计
2.grid-period   传统的网格搜索下的Periodogram
3.grid-period-itr 粗搜索+迭代下的网格搜索Periodogram "grid-period":{"mode":"iterative","iterative_times":2},
针对于该实验脚本和并行运算,自主编写了实验数据整理与保存的函数 **period_ils.data_collect,period_ils.dict_collect**,你可以根据自己的实验要求自行编写，不是必要函数。

# 3.注意事项
1.joblib并行运算的**n_job**代表了并行数量，和cpu核数有关,请按照之际的cpu核数合理设置。

3.真的使用的核心代码模块只有 ```class periodogram, class simulated_phase```,可根据自己的研究内容进行深度定制。

参考文献： 
[1] [](https://www.tudelft.nl/citg/over-faculteit/afdelingen/geoscience-remote-sensing/research/lambda/lambda)下载python版本
[2] Ferretti A, Prati C, Rocca F. Permanent scatterers in sar interferometry[J]. IEEE Transactions on geoscience and remote sensing, 2001, 39(1): 8-20.
[3] Van Leijen F J. Persistent scatterer interferometry based on geodetic estimation theory[D]. Delft University of Technology, 2014.
[4] Czikhardt R, Van Der Marel H, Papco J. Gecoris: An open-source toolbox for analyzingtime series of corner reflectors in insar geodesy[J]. Remote Sensing, 2021, 13(5): 926.