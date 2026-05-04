# surrogate_config_11d.py
# ============================================================
#  SINGLE SOURCE OF TRUTH for bounds, DNN architecture,
#  true parameter vector, and output settings.
#  Import from here in ALL scripts — never redefine locally.
# ============================================================

import os
import numpy as np
import random

# --------------------------
#   Reproducibility
# --------------------------
SEED = 1234

def set_global_seed(seed: int = SEED):
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass
    np.random.seed(seed)
    random.seed(seed)


# --------------------------
#   Parameter space (11-D)
# --------------------------
XNAMES = ['as', 'bs', 'ag', 'bg', 'N0r', 'N0s', 'N0g', 'rhos', 'rhog', 'qc0', 'qi0']

X_TRUE_11D = np.array(
    [200.0, 0.3, 400.0, 0.4, 0.5, 0.5, 0.5, 0.2, 0.4, 1.0e-3, 6.0e-4],
    dtype=float
)

PMIN_11 = np.array(
    [50.0, 0.10, 50.0, 0.10, 0.05, 0.05, 0.05, 0.05, 0.05, 1.0e-4, 2.0e-6],
    dtype=float
)
PMAX_11 = np.array(
    [1000.0, 1.0, 1200.0, 0.90, 5.0, 2.5, 2.5, 1.0, 1.0, 2.0e-3, 1.0e-3],
    dtype=float
)

pmin = PMIN_11
pmax = PMAX_11

IDX_AG = XNAMES.index('ag')   # 2
IDX_BG = XNAMES.index('bg')   # 3

# --------------------------
#   Outputs @ t = 120 min
# --------------------------
YNAMES_SIX = ['PCP', 'ACC', 'LWP', 'IWP', 'OLR', 'OSR']
K_OUTPUTS   = 6

Y_SIG_SIX = np.array([2.0, 5.0, 0.5, 1.0, 5.0, 5.0], dtype=float)

YMASK_FIG7 = np.array([1, 0, 1, 1, 0, 0], dtype=int)   # no radiation: PCP, LWP, IWP
YMASK_FIG9 = np.array([0, 0, 0, 0, 1, 1], dtype=int)   # radiation:    OLR, OSR

CRM_OUTPUT_COL_START = 18
CRM_OUTPUT_COL_END   = 24


# --------------------------
#   DNN hyperparameters (AUTHORITATIVE)
# --------------------------
DNN_LAYERS   = 8
DNN_NODES    = 50
DNN_LR       = 1e-3
DNN_BATCH    = 32
DNN_EPOCHS   = 20000
DNN_LR_FINAL = 5e-7


# --------------------------
#   Training sizes
# --------------------------
TRAIN_SIZE       = 5000
TRAIN_SIZES_LIST = list(range(100, 5001, 100))


# --------------------------
#   CS / PDR hyperparameters
# --------------------------
PDR_POLY_LEVEL        = 20
PDR_TARGET_STEPS      = 2000
PDR_MAX_INNER_ITERS   = 2000
PDR_EPS0              = 1e-6
PD_EARLY_STOP         = False
PD_EARLY_STOP_TOL     = 1e-10

# MCMC settings — shared by both DNN and PDR MCMC pipelines
MCMC_NWALK = 32
MCMC_NBURN = 40000
MCMC_NMCMC = 400000

SG_LEVEL = 4
SG_RULE  = "clenshaw-curtis"

NUM_TRIALS = 20


# --------------------------
#   Cloud MCMC "true" posterior chain files
#   NOTE: verify these paths on your system — update here and
#   both DNN and PDR pipelines will stay in sync automatically.
# --------------------------
TRUE_CHAIN_FILE_NORAD = (
    "/home/ss24ce/last_time-2/FINAL_CRM11D_20260216_140339/output/"
    "mcmc_crm1d__EXP_3_full11_ag400.00000_bg0.40000_norad.nc"
)
TRUE_CHAIN_FILE_RAD = (
    "/home/ss24ce/last_time-2/FINAL_CRM11D_20260222_044235/output/"
    "mcmc_crm1d__EXP_3_full11_ag400.00000_bg0.40000_rad.nc"
)


# --------------------------
#   CRM file locations
# --------------------------
EXPDIR   = './cloud_column_model/'
RUN_FILE = 'run_one_crm1d.txt'
OUT_FILE = 'crm1d_output.txt'
NAMELIST = 'namelist_3h_t30-180.f90'

INPUT_FILE_FULL  = os.path.join(EXPDIR, RUN_FILE)
OUTPUT_FILE_FULL = os.path.join(EXPDIR, OUT_FILE)
NAMELIST_FULL    = os.path.join(EXPDIR, NAMELIST)


# --------------------------
#   Output directories
# --------------------------
os.makedirs('models_11d',    exist_ok=True)
os.makedirs('output',        exist_ok=True)
os.makedirs('plots',         exist_ok=True)
os.makedirs('trials_results', exist_ok=True)


# --------------------------
#   Convenience helpers
# --------------------------
def map_to_canonical_11d(X: np.ndarray) -> np.ndarray:
    """Affine map from physical [PMIN_11, PMAX_11] to [-1,1]^11."""
    X = np.asarray(X, dtype=float)
    return 2.0 * (X - PMIN_11) / (PMAX_11 - PMIN_11) - 1.0


def map_to_physical_11d(Xi: np.ndarray) -> np.ndarray:
    """Inverse: [-1,1]^11 → physical domain."""
    Xi = np.asarray(Xi, dtype=float)
    return 0.5 * (Xi + 1.0) * (PMAX_11 - PMIN_11) + PMIN_11


def ensure_within_bounds(X: np.ndarray) -> np.ndarray:
    """Clip to [PMIN_11, PMAX_11]."""
    return np.clip(X, PMIN_11, PMAX_11)
