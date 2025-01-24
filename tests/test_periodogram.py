import numpy as np
import pytest
from linear_search_periodogram.periodogram import Periodogram
from linear_search_periodogram.simulated_phase import SimulatedPhase

@pytest.fixture
def sample_params():
    """Create sample parameters for testing"""
    return {
        "Nifg": 20,
        "noise_level": 10,
        "search_range": {
            "h_range": [-50, 50],
            "h_step": 1,
            "v_range": [-10, 10],
            "v_step": 0.1
        },
        "iterative_times": 2,
        "period_est_mode": "linear-period",
        "param_simulation": {
            "height": 10,
            "velocity": 0.5
        },
        "sentinel-1": {
            "wavelength": 0.0555,
            "incidence_angle": 34,
            "H": 693000,
            "Bn": 150,
            "revisit_cycle": 6
        }
    }

@pytest.fixture
def simulated_data(sample_params):
    """Create simulated phase data for testing"""
    sim = SimulatedPhase(sample_params, check_times=1, check_num=0)
    arc_phase, par2ph = sim.sim_arc_phase()
    return arc_phase, par2ph

def test_periodogram_initialization(sample_params, simulated_data):
    """Test Periodogram class initialization"""
    arc_phase, par2ph = simulated_data
    periodogram = Periodogram(sample_params, arc_phase, par2ph)

    assert isinstance(periodogram.height_search_space, np.ndarray)
    assert isinstance(periodogram.velocity_search_space, np.ndarray)
    assert periodogram.result == {"height": 0, "velocity": 0}

def test_linear_search():
    """Test the static linear_search method"""
    # Create test data
    phase = np.array([0, np.pi/2, np.pi])
    par2ph = np.array([1, 1, 1])
    search_space = np.array([0, 0.5, 1.0])

    result = Periodogram.linear_search(phase, par2ph, search_space)
    assert isinstance(result, (float, np.float64))

def test_periodogram_estimation(sample_params, simulated_data):
    """Test the full periodogram estimation process"""
    arc_phase, par2ph = simulated_data
    periodogram = Periodogram(sample_params, arc_phase, par2ph)

    # Test linear periodogram
    h_est, v_est = periodogram.periodogram_estimation()
    assert isinstance(h_est, (float, np.float64))
    assert isinstance(v_est, (float, np.float64))

    # Test grid periodogram
    sample_params["period_est_mode"] = "grid-period"
    periodogram = Periodogram(sample_params, arc_phase, par2ph)
    h_est, v_est = periodogram.periodogram_estimation()
    assert isinstance(h_est, (float, np.float64))
    assert isinstance(v_est, (float, np.float64))

def test_simulated_phase(sample_params):
    """Test SimulatedPhase class"""
    sim = SimulatedPhase(sample_params, check_times=1, check_num=0)

    # Test noise generation
    sim.add_radom_noise()
    assert len(sim.noise) == sample_params["Nifg"]

    # Test baseline generation
    sim.add_random_baseline()
    assert len(sim.normal_baseline) == sample_params["Nifg"]

    # Test parameter to phase conversion
    sim.par2ph()
    assert "height" in sim.par2ph
    assert "velocity" in sim.par2ph

    # Test full simulation
    arc_phase, par2ph = sim.sim_arc_phase()
    assert len(arc_phase) == sample_params["Nifg"]
    assert isinstance(par2ph, dict)

def test_error_handling(sample_params, simulated_data):
    """Test error handling"""
    arc_phase, par2ph = simulated_data

    # Test with invalid estimation mode
    invalid_params = sample_params.copy()
    invalid_params["period_est_mode"] = "invalid-mode"
    periodogram = Periodogram(invalid_params, arc_phase, par2ph)

    with pytest.raises(Exception):
        periodogram.periodogram_estimation()