import numpy as np

class Periodogram:
    """
    Periodogram method for temporal phase unwrapping"""
    def __init__(self, param, phase_obs, par2ph) -> None:
        """
        Parameters
        ----------
        param : dict
            Dictionary file containing all parameters from the meta (.json) file.
        phase_obs : ndarray
            Simulated observed phase wrapped to [-pi, pi].
        par2ph : dict
            Conversion coefficient from parameters v and h to phase.
        """
        self.params = param
        self.arc_phase = phase_obs
        self.par2ph = par2ph
        self.result = {"height": 0, "velocity": 0}

        # Construct the search spaces for height and velocity
        create_search_space = lambda bounds, step: np.arange(bounds[0], bounds[1] + 1, 1) * step

        self.height_search_space = create_search_space(
            self.params["search_range"]["h_range"],
            self.params["search_range"]["h_step"]
        )

        self.velocity_search_space = create_search_space(
            self.params["search_range"]["v_range"],
            self.params["search_range"]["v_step"]
        )


    @staticmethod
    def linear_search(phase, par2ph, search_space):
        """Linear search to find parameter value that maximizes objective function.

        Parameters
        ----------
        phase : numpy.ndarray
            Complex signal form of the observed phase.
        par2ph : numpy.ndarray
            Conversion coefficient from parameters to phase.
        search_space : numpy.ndarray
            Parameter search space.

        Returns
        -------
        float
            Search value that maximizes the objective function.
        """
        # Calculate residual phase for each value in search space
        residue_phase = phase - np.outer(par2ph, search_space)

        # Calculate coherence (objective function) for each residual
        complex_signal = np.exp(1j * residue_phase)
        coherence = np.abs(np.sum(complex_signal, axis=0))

        # Find value that maximizes coherence
        max_idx = np.argmax(coherence)
        estimated_value = search_space[max_idx]

        return estimated_value


    def linear_periodogram(self):
        """
            Linear-Periodogram method
        """

        #  --------------------------------------------
        #  Step1: calculate the height phase
        #  --------------------------------------------

        # Calculate height-phase from all height values within the search space
        phase_hght = np.outer(self.height_search_space, self.par2ph["height"])
        # remove height phase from delta_phase
        hght_removed = self.arc_phase - phase_hght

        #  --------------------------------------------
        #  Step2: sum over different height values
        #  --------------------------------------------
        # Convert height-removed phase to complex exponential and sum over heights
        hght_removed_sum = np.sum(np.exp(1j * hght_removed), axis=0)

        #  --------------------------------------------
        #  Step3: linear search for delta_v
        #  --------------------------------------------
        estimated_velocity = self.linear_search(
            np.angle(hght_removed_sum),
            self.par2ph["velocity"],
            self.velocity_search_space
        )

        #  --------------------------------------------
        #  Step4: linear search for height estimate
        #  --------------------------------------------
        vel_removed = self.arc_phase - estimated_velocity * self.par2ph["velocity"]
        estimated_height = self.linear_search(
            vel_removed,
            self.par2ph["height"],
            self.height_search_space
        )

        self.result = {"height": estimated_height, "velocity": estimated_velocity}

        if self.params["iterative_times"] != 0:
            self.iteration()

    def iteration(self):
        iterative_times = self.params["iterative_times"]
        h_est = self.result["height"]
        v_est = self.result["velocity"]
        for i in range(iterative_times):
            # remove height term from delta_phase
            phase_sub_h = self.arc_phase - h_est * self.par2ph["height"]

            # linear search for param_v
            e_phase_sub_h = np.exp(1j * phase_sub_h)
            v_est = self.linear_search(e_phase_sub_h, self.par2ph["velocity"], self.velocity_search_space)

            # remove velocity term from delta_phase
            phase_sub_v = self.arc_phase - v_est * self.par2ph["velocity"]

            # linear search for param_h
            e_phase_sub_v = np.exp(1j * phase_sub_v)
            h_est = self.linear_search(e_phase_sub_v, self.par2ph["height"], self.height_search_space)

        self.result = {"height": h_est, "velocity": v_est}


    def grid_periodogram(self):
        """
            grid periodogram method
        """
        # Construct the search space for grid periodogram method
        v2ph = self.par2ph["velocity"].reshape(-1, 1)
        h2ph = self.par2ph["height"].reshape(-1, 1)
        search_phase_space = np.kron(self.velocity_search_space * v2ph, np.ones((1, len(self.height_search_space)))) + \
                             np.kron(np.ones((1, len(self.velocity_search_space))), self.height_search_space * h2ph)

        # Calculate the difference between the observed phase and the search phase space.
        sub_phase = self.arc_phase.reshape(-1, 1) - search_phase_space

        # Calculate sub_phase and sum along the image count dimension to obtain the objective function value.
        gamma_temporal = np.sum(np.exp(1j * sub_phase), axis=0, keepdims=True) / self.params["Nifg"]

        # Obtain the index of the maximum objective function value.
        max_index = np.argmax(np.abs(gamma_temporal))

        # Obtain the estimated parameters v and h based on the index.
        param_index = np.unravel_index(max_index, [len(self.velocity_search_space), len(self.height_search_space)], order="C")

        h_est = self.height_search_space[param_index[1]]
        v_est = self.velocity_search_space[param_index[0]]

        self.result = {"height": h_est, "velocity": v_est}

    def periodogram_estimation(self):
        if self.params["period_est_mode"] == "linear-period":
            self.linear_periodogram()
        elif self.params["period_est_mode"] == "grid-period":
            self.grid_periodogram()
        return self.result["height"], self.result["velocity"]

