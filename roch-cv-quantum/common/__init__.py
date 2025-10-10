"""
Common utilities for CV quantum optics simulations.

Provides Gaussian state tooling, noise models, Fisher information calculations,
and plotting utilities shared across all projects.
"""

__version__ = "0.1.0"

from .constants import *
from .gaussian import *
from .noise_models import *
from .fisher_crb import *
from .plotting import *
from .utils import *

__all__ = [
    "HBAR",
    "C_LIGHT",
    "LAMBDA_TELECOM",
    "SHOT_NOISE_VARIANCE",
    "beam_splitter",
    "phase_shifter",
    "squeezer",
    "loss_channel",
    "vacuum_state",
    "coherent_state",
    "squeezed_state",
    "cov_after_channel",
    "purity",
    "log_negativity",
    "shot_noise",
    "thermal_noise",
    "detector_noise",
    "balanced_homodyne",
    "fisher_information_phase",
    "cramer_rao_bound",
    "setup_plot_style",
    "save_figure",
    "set_seed",
    "save_results",
    "load_results",
]
