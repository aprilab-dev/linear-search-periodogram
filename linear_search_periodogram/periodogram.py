import numpy as np

class Periodogram:
    """Periodogram method for temporal phase unwrapping"""

    def __init__(self, param, phase_obs, par2ph) -> None:
        """Initialize parameters

        Parameters
        ----------
        param : _dict_
            _Dictionary file, containing all the parameters in the JSON file._
        phase_obs : _ndarray_
            _The simulated observed phase is wrapped to the range [-pi, pi]._
        par2ph : _dict_
            _The conversion coefficient from parameters v and h to phase__
        """
        self.params      = param
        self.arc_phase  = phase_obs
        self.par2ph     = par2ph
        self.result     = {"height": 0, "velocity": 0}

        # Construct the search space.
        h_start = self.params["search_range"]["h_range"][0]
        h_end   = self.params["search_range"]["h_range"][1]
        h_step  = self.params["search_range"]["h_step"]

        v_start = self.params["search_range"]["h_range"][0]
        v_end   = self.params["search_range"]["h_range"][1]
        v_step  = self.params["search_range"]["v_step"]

        self.height_search_space   = np.arange(h_start, h_end + 1, 1) * h_step
        self.velocity_search_space = np.arange(v_start, v_end + 1, 1) * v_step

        

    @staticmethod
    def linear_search(exp_phase, p2ph, search_space):
        """linear_search
        Parameters
        ----------
        exp_phase : The complex signal form of the observed phase.
        p2ph : Conversion coefficient from parameters to phase.
        search_spaceW : Parameter search space.

        Returns
        -------
        The search value that maximizes the objective function.
        """
        sub_e = np.exp(-1j * (p2ph.reshape(-1, 1) * search_space.reshape(1, -1))) 
        objective_function = np.mean(exp_phase.reshape(-1, 1) * sub_e, axis=0)  
        result = search_space[np.argmax(np.abs(objective_function))]
        return result


    def linear_periodogram(self):
        """
            Linear-Periodogram method
        """
        # step1: 
        # calculate the height phase
        delta_phase_h = self.height_search_space.reshape(-1, 1) * self.par2ph["height"].reshape(1, -1)
        # remove height term from delta_phase
        sub_phase_h = self.arc_phase - delta_phase_h

        # step2: 
        # sum over different height values
        e_phase_sub_h = np.sum(np.exp(1j * sub_phase_h), axis=0)

        # step3: 
        # linear search for delta_v
        v_est = self.linear_search(e_phase_sub_h, self.par2ph["velocity"], self.velocity_search_space)
   
        # step4:
        # remove velocity term 
        phase_sub_v = self.arc_phase - v_est * self.par2ph["velocity"]
        # linear search to determine delta_h
        e_phase_sub_v = np.exp(1j * phase_sub_v)
        h_est = self.linear_search(e_phase_sub_v, self.par2ph["height"], self.height_search_space)

        self.result = {"height": h_est, "velocity": v_est}

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

