import numpy as np

class SimulatedPhase: 
    """观测相位仿真模拟,包括随机相位噪声、随机垂直基线、参数到相位的转换系数、每arc弧观测相位等,
    采用“线性形变+地形高度残差+随机相位噪声”的模拟方式
    """

    def __init__(self, param_file, check_times, check_num, sat_name="sentinel-1"):
        """仿真模拟参数初始化

        Parameters
        ----------
        param_file : json
            _字典文件,包含有仿真模拟的卫星参数、干涉图像的参数、相位解缠的初始参数设置_
        check_times : _int_
            _仿真模拟次数_
        check_num : _int_
            _模拟参数下标，用于控制随机种子号_
        sat_name : str, optional
            _模拟的卫星名称_, by default "sentinel-1"

        m2ph : _float_
            _形变到相位的转换系数,4pi/lambda_
        incidence_angle : _float_
            _入射角,弧度制_
        R : _float_
            _斜距_
        Bn : _float_
            _垂直基线的标准差_
        revisit_cycle : _float_
            _重访周期_
        """
        self.param_file = param_file
        self.check_times = check_times
        self.check_num = check_num
        self.m2ph = 4 * np.pi / self.param_file[sat_name]["wavelength"]
        self.incidence_angle = param_file[sat_name]["incidence_angle"] * np.pi / 180
        self.R = param_file[sat_name]["H"] / np.cos(self.incidence_angle)
        self.Bn = param_file[sat_name]["Bn"]
        self.revisit_cycle = param_file[sat_name]["revisit_cycle"]

        # print(self.R, self.Bn, self.revisit_cycle, self.incidence_angle)

    def add_radom_noise(self):
        """随即相位噪声
        note:
        通过随机种子生成随机相位噪声,随机噪声符合正态分布,均值为0,标准差为noise_level[deg]
        """
        # 随机种子
        np.random.seed(self.check_num + self.check_times)
        # 随机相位噪声
        self.noise = np.random.normal(0, np.pi * self.param_file["noise_level"] / 180, self.param_file["Nifg"])

    def add_random_baseline(self):
        """垂直基线随机生成
        note:
            通过随机种子生成随机垂直基线,随机垂直基线符合正态分布,均值为0,标准差为Bn[m]
        """
        np.random.seed(self.check_num)
        self.normal_baseline = np.random.normal(0, self.Bn, self.param_file["Nifg"])

    def par2ph(self):
        """_估计参数到相位的转换系数_
        Parameters:
        -----------
        h2ph : _float_
            _高度h到相位的转换系数,和垂直基线有关_
        v2ph : _float_
            _形变速率v到相位的转换系数,和时间基线有关_
        par2ph : _dict_
            _参数到相位的转换系数,便于后续使用_
        note:
            根据观测相位的模型function modeol 来获得参数到相位的转换系数,包括v2ph和h2ph。具体的观测相位模型参考论文第2章节
        """
        self.add_random_baseline()
        par2ph = dict()
        # h2ph
        self.h2ph = self.m2ph * self.normal_baseline / (self.R * np.sin(self.incidence_angle))
        # v2ph
        self.v2ph = self.m2ph * self.revisit_cycle * np.arange(1, self.param_file["Nifg"] + 1, 1) / 365
        par2ph["height"] = self.h2ph
        par2ph["velocity"] = self.v2ph
        self.par2ph = par2ph

    def sim_arc_phase(self):
        """_模拟每arc弧观测相位_

        note:
        通过参数到相位的转换系数和仿真模拟参数v,h生成每arc弧观测相位

        Returns
        -------
        arc_phase : _array_
            _每arc弧观测相位_
        par2ph : _dict_
            _参数到相位的转换系数:字典形式_
        """
        self.add_radom_noise()
        self.par2ph()
        phase_true = self.param_file["param_simulation"]["height"] * self.h2ph + self.param_file["param_simulation"]["velocity"] * self.v2ph + self.noise
        complex_signal = np.exp(1j * phase_true)
        arc_phase = np.angle(complex_signal)
        return arc_phase, self.par2ph

    def sim_arc_phase_a(self):
        # self.add_radom_noise()
        self.par2ph()
        phase_true = self.param_file["param_simulation"]["height"] * self.h2ph + self.param_file["param_simulation"]["velocity"] * self.v2ph
        complex_signal = np.exp(1j * phase_true)
        arc_phase = np.angle(complex_signal)
        a = np.round((arc_phase - phase_true) / (2 * np.pi))
        return arc_phase, self.par2ph, a

