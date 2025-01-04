import numpy as np

class Periodogram:
    """Periodogram method for temporal phase unwrapping 基于谱估计的相位解缠方法"""

    def __init__(self, param_file, phase_obs, par2ph) -> None:
        """初始化参数

        Parameters
        ----------
        param_file : _dict_
            _字典文件,包含有仿真模拟的卫星参数、干涉图像的参数、相位解缠的初始参数设置_
        phase_obs : _ndarray_
            _仿真模拟的观测相位,被缠绕到了[-pi,pi]_
        par2ph : _dict_
            _参数v,h到相位的转换系数,维度由干涉图影像数量决定__
        fine_search : _dict_
            _细搜索参数设置:搜索范围,搜索步长_
        coarse_search : _dict_
            _粗搜索参数设置:搜索范围,搜索步长_
        h_sum_range : _list_
            _初步h搜索的参数设置:搜索范围,搜索步长_
        """
        self.param_file = param_file
        self.arc_phase = phase_obs
        self.par2ph = par2ph
        self.fine_search = self.param_file["fine_search"]
        self.coarse_search = self.param_file["coarse_search"]
        self.h_sum_range = self.param_file["h_sum_range"]

    def construct_searched_space(self):
        """构建参数搜搜空间

        Parameters
        ----------
        W_h_fine : _ndarray_
            _高度h的细搜索空间_
        W_v_fine : _ndarray_
            _形变速率v的细搜索空间_
        h_sum : _ndarray_
            _初步h搜索的参数空间_
        note:
        根据所选择的Periodogram方法,构建相应的搜索空间
        """
        # Fine search space精细搜索空间
        self.W_h_fine = np.arange(self.fine_search["h_range"][0], self.fine_search["h_range"][1] + 1, 1) * self.fine_search["step_size"][0]
        self.W_v_fine = np.arange(self.fine_search["v_range"][0], self.fine_search["v_range"][1] + 1, 1) * self.fine_search["step_size"][1]
        # 根据所选择的Periodogram方法,构建相应的搜索空间
        if self.param_file["period_est_mode"] == "linear-period":
            self.h_sum = np.arange(self.h_sum_range[0], self.h_sum_range[1] + 1, 1) * self.fine_search["step_size"][0]
        elif self.param_file["period_est_mode"] == "grid-period":
            # Coarse search space粗搜索空间
            self.W_h_coarse = np.arange(self.coarse_search["h_range"][0], self.coarse_search["h_range"][1] + 1, 1) * self.coarse_search["step_size"][0]
            self.W_v_coarse = np.arange(self.coarse_search["v_range"][0], self.coarse_search["v_range"][1] + 1, 1) * self.coarse_search["step_size"][1]
            # print(self.W_h_coarse, self.W_v_coarse)
            # print(self.W_h_fine, self.W_v_fine)

    @staticmethod
    def construct_phase_space(W_h, W_v, h2ph, v2ph):
        """构建搜索相位空间:

        Parameters
        ----------
        W_h : _type_
            _h的搜索空间_
        W_v : _type_
            _v的搜索空间_
        h2ph : _type_
            _h到相位的转换系数_
        v2ph : _type_
            _v到相位的转换系数_

        Returns
        -------
        search_phase_space : _ndarray_
            _搜索相位空间_
        note:
        grid-period方法会使用到该相位搜索空间
        """
        v2ph = v2ph.reshape(-1, 1)
        h2ph = h2ph.reshape(-1, 1)
        search_phase_space = np.kron(W_v * v2ph, np.ones((1, len(W_h)))) + np.kron(np.ones((1, len(W_v))), W_h * h2ph)
        return search_phase_space

    @staticmethod
    def Periodogram_1D_DFT(e_phase, p2ph, W):
        """1D_DFT谱估计

        Parameters
        ----------
        e_phase : _type_
            _观测相位的复数信号形式_
        p2ph : _type_
            _参数到相位的转换系数_
        W : _type_
            _参数搜索空间_

        Returns
        -------
        _type_
            _目标函数值/频谱值_
        note:
        针对于相位模型只包含1个参数的情况,根据参数搜索空间W,计算目标函数值/频谱值
        """
        # print(e_phase.shape)
        sub_e = np.exp(-1j * (p2ph.reshape(-1, 1) * W.reshape(1, -1)))  ##shape:(W_shape,Nifg))
        # print(sub_e.shape)
        xjw_dft_v = np.mean(e_phase.reshape(-1, 1) * sub_e, axis=0)  ##shape:(W_shape)
        # print(xjw_dft_v.shape)
        coh = np.abs(xjw_dft_v)
        # print(coh.shape)
        return coh

    def linear_periodogram(self):
        """Linear-Periodogram方法
        分为两步分:1.模糊估计 2.迭代精细估计
        """
        # Mean Estimation 1D_V
        # 根据h_sum来获得I_sum
        self.Periodogram_1D_h()
        # 基于I_sum进行参数v的1D-Periodogram估计
        DFT_V = self.Periodogram_1D_DFT(self.e_phase_sub_h, self.par2ph["velocity"], self.W_v_fine)
        v_est = self.W_v_fine[np.argmax(DFT_V)]
        # Fine Esitimation 1D_H
        # 根据v_est去除v的影响，获得新的相位信息
        phase_sub_v = self.arc_phase - v_est * self.par2ph["velocity"]
        e_phase_sub_v = np.exp(1j * phase_sub_v)
        # 进行参数h的1D-Periodogram估计
        DFT_H = self.Periodogram_1D_DFT(e_phase_sub_v, self.par2ph["height"], self.W_h_fine)
        h_est = self.W_h_fine[np.argmax(DFT_H)]

        # Iterative estimation
        if self.param_file["linear-period"]["mode"] == "iterative":
            for i in range(self.param_file["linear-period"]["iterative_times"]):
                # Fine Esitimation 1D_V
                # 根据h_est去除h的影响，获得新的相位信息
                phase_sub_h = self.arc_phase - h_est * self.par2ph["height"]
                e_phase_sub_h = np.exp(1j * phase_sub_h)
                # 进行参数v的1D-Periodogram高精度估计
                DFT_V = self.Periodogram_1D_DFT(e_phase_sub_h, self.par2ph["velocity"], self.W_v_fine)
                v_est = self.W_v_fine[np.argmax(DFT_V)]
                # Fine Esitimation 1D_H
                # 根据v_est去除v的影响，获得新的相位信息
                phase_sub_v = self.arc_phase - v_est * self.par2ph["velocity"]
                e_phase_sub_v = np.exp(1j * phase_sub_v)
                # 进行参数h的1D-Periodogram高精度估计
                DFT_H = self.Periodogram_1D_DFT(e_phase_sub_v, self.par2ph["height"], self.W_h_fine)
                h_est = self.W_h_fine[np.argmax(DFT_H)]
        self.param = {"height": h_est, "velocity": v_est}

    def Periodogram_1D_h(self):
        """降低h再观测相位中的影响
        note:
        根据h_sum来获得I_sum

        """
        # 从观测相位中减去h_sum对应的相位
        self.sub_phase_h = self.arc_phase - self.h_sum.reshape(-1, 1) * self.par2ph["height"].reshape(1, -1)
        # print(self.sub_phase_h)
        # 计算减去h_sum后的复数信号并在h维度上求和
        self.e_phase_sub_h = np.sum(np.exp(1j * self.sub_phase_h), axis=0)
        # print(self.arc_phase)
        # print(np.angle(self.e_phase_sub_h))
        # print(self.e_phase_sub_h.shape)

    def grid_periodogram(self, W_v, W_h):
        """Grid-Periodogram方法,基于网格搜索的Periodogram相位解缠方法

        Parameters
        ----------
        W_v : _type_
            _参数v搜索空间_
        W_h : _type_
            _参数h搜索空间_
        """
        # 根据搜索空间构建搜索相位空间
        search_phase_space = self.construct_phase_space(W_h, W_v, self.par2ph["height"], self.par2ph["velocity"])
        # 观测相位与搜索相位空间的差值
        sub_phase = self.arc_phase.reshape(-1, 1) - search_phase_space
        # 计算sub_phase并在影像数量维度上求和，获得目标函数值
        gamma_temporal = np.sum(np.exp(1j * sub_phase), axis=0, keepdims=True) / self.param_file["Nifg"]
        # 获得目标函数值最大的索引
        max_index = np.argmax(np.abs(gamma_temporal))
        # print(max_index)
        # 根据索引获得估计的参数v,h
        param_index = np.unravel_index(max_index, [len(W_v), len(W_h)], order="C")
        h_est = W_h[param_index[1]]
        v_est = W_v[param_index[0]]
        self.param = {"height": h_est, "velocity": v_est}

    def grid_periodogram_itr(self):
        """基于迭代估计的Grid-Periodogram方法
        note:
        1.粗搜索 2.细搜索 3.迭代估计
        """
        # coarse search
        self.grid_periodogram(self.W_v_coarse, self.W_h_coarse)
        h_est = self.param["height"]
        v_est = self.param["velocity"]
        # iterative fine search
        if self.param_file["grid-period"]["mode"] == "iterative":
            for i in range(self.param_file["grid-period"]["iterative_times"]):
                # Fine Esitimation 1D_V
                phase_sub_h = self.arc_phase - h_est * self.par2ph["height"]
                e_phase_sub_h = np.exp(1j * phase_sub_h)
                DFT_V = self.Periodogram_1D_DFT(e_phase_sub_h, self.par2ph["velocity"], self.W_v_fine)
                v_est = self.W_v_fine[np.argmax(DFT_V)]
                # Fine Esitimation 1D_H
                phase_sub_v = self.arc_phase - v_est * self.par2ph["velocity"]
                e_phase_sub_v = np.exp(1j * phase_sub_v)
                DFT_H = self.Periodogram_1D_DFT(e_phase_sub_v, self.par2ph["height"], self.W_h_fine)
                h_est = self.W_h_fine[np.argmax(DFT_H)]
        self.param = {"height": h_est, "velocity": v_est}

    def periodogram_estimation(self):
        self.construct_searched_space()
        if self.param_file["period_est_mode"] == "linear-period":
            self.linear_periodogram()
        elif self.param_file["period_est_mode"] == "grid-period":
            self.grid_periodogram(self.W_v_fine, self.W_h_fine)
        elif self.param_file["period_est_mode"] == "grid-period-itr":
            self.grid_periodogram_itr()
        return self.param["height"], self.param["velocity"]

