#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Scalability Benchmark: PDR (GPU-batched) vs DNN (GPU)
======================================================
Dimensions  : d ∈ {2, 4, 6, 8, 11}
              - First d parameters vary in [pmin[:d], pmax[:d]]
              - Remaining 11-d parameters fixed at x_true
Training sizes : m = 100 → 5000, step 100
Data           : CRM called via parmap (parallel, fastest)
Methods timed  : PDR (GPU-batched, Table 5.1) and DNN (GPU, fixed arch)

Incremental saving
------------------
  After EVERY (d, m) pair completes, one row is immediately written and
  flushed to scalability_results.csv — so a mid-run crash loses nothing.

  CSV columns: d, m, N_basis, t_pdr_s, t_dnn_s

Theoretical complexity
----------------------
  PDR : O(m × N(d))               N(d) = hyperbolic cross basis size
  DNN : O(m × (d×H + L×H²))      H=50 nodes, L=8 hidden layers
        L×H² = 20,000 >> d×H, so DNN time is nearly flat across d

Output
------
  scalability_results.csv         — incremental row per (d, m)
  scalability_pdr_dnn_bydim.png   — combined plot after all runs finish
  scalability_times.npz           — full timing arrays for reuse
"""

import csv
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.special import legendre
import sys
import time
import warnings
warnings.filterwarnings("ignore")

sys.path.append('/home/ss24ce/.local/lib/python3.10/site-packages')

from parmap_framework import parmap
from module_runcrm import runcrm
from gpu_batched_sr_lasso import solve_all_outputs_gpu_batch

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim

# =============================================================================
# CONFIGURATION
# =============================================================================

RNG_SEED = 42

DIMS        = [2, 4, 6, 8, 11]
TRAIN_SIZES = list(range(100, 5001, 100))   # 100, 200, ..., 5000

# 11D physical parameter setup
pmin   = np.array([50.0,   0.10, 50.0,   0.1,  0.05,  0.05,  0.05,  0.05,  0.05,  1.e-4, 2.e-6])
pmax   = np.array([1000.0, 1.0,  1200.0, 0.90, 5.0,   2.5,   2.5,   1.0,   1.0,   2.e-3, 1.e-3])
x_true = np.array([200.0,  0.3,  400.0,  0.4,  0.5,   0.5,   0.5,   0.2,   0.4,   1.e-3, 6.e-4])
xnames = ['as','bs','ag','bg','N0r','N0s','N0g','rhos','rhog','qc0','qi0']

# CRM outputs: columns 18..23 → 6 quantities
OUTPUT_NAMES = ['PCP','ACC','LWP','IWP','OLR','OSR']
K = 6

# CRM file paths
INPUT_FILE  = './cloud_column_model/run_one_crm1d.txt'
OUTPUT_FILE = './cloud_column_model/crm1d_output.txt'
NAMELIST    = './cloud_column_model/namelist_3h_t30-180.f90'

# Parmap (fastest parallel CRM execution)
PARMASTER   = 'scispark6.jpl.nasa.gov:8786'
PARNWORKERS = 12
PARMODE     = 'par'

# PDR
TARGET_TOTAL_STEPS = 2000
HC_POLY_LEVEL      = 20     # hyperbolic cross polynomial level

# DNN (architecture fixed across all d; only input layer width = d changes)
DNN_EPOCHS   = 20000
DNN_BATCH    = 128
DNN_LR       = 1e-3
DNN_LAYERS   = 8            # number of hidden layers
DNN_HIDDEN   = 50           # nodes per hidden layer
DNN_FINAL_LR = 5e-7         # end LR for exponential decay scheduler

# Output CSV
CSV_FILE = 'scalability_results.csv'
CSV_HEADER = ['d', 'm', 'N_basis', 't_pdr_s', 't_dnn_s']

# =============================================================================
# CSV helper — open once, write row-by-row, flush immediately
# =============================================================================

def open_csv(filepath):
    """
    Open CSV for writing (append mode so re-runs add rows, not overwrite).
    Returns (file_handle, csv.writer).
    Write header only if file is new/empty.
    """
    file_is_new = (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0)
    fh = open(filepath, 'a', newline='')
    writer = csv.writer(fh)
    if file_is_new:
        writer.writerow(CSV_HEADER)
        fh.flush()
    return fh, writer


def write_row(fh, writer, d, m, N_basis, t_pdr, t_dnn):
    """Write one result row and flush immediately to disk."""
    writer.writerow([d, m, N_basis, f'{t_pdr:.6f}', f'{t_dnn:.6f}'])
    fh.flush()          # guarantee data hits disk even if job is killed

# =============================================================================
# Basis utilities
# =============================================================================

def multiidx_gen(N, rule, w, base=0, multiidx=np.array([]), MULTI_IDX=np.array([])):
    """Recursive hyperbolic-cross multi-index generation."""
    if len(multiidx) != N:
        i = base
        while rule(np.append(multiidx, i)) <= w:
            MULTI_IDX = multiidx_gen(N, rule, w, base, np.append(multiidx, i), MULTI_IDX)
            i += 1
    else:
        MULTI_IDX = np.vstack([MULTI_IDX, multiidx]) if MULTI_IDX.size else multiidx
    return MULTI_IDX


def build_basis(d, poly_level=HC_POLY_LEVEL):
    """Return (Lambda, weights) for hyperbolic cross Legendre basis in d dims."""
    HCfunc  = lambda x: np.prod(x + 1) - 1
    Lambda  = multiidx_gen(d, HCfunc, poly_level).astype(int)
    N_basis = Lambda.shape[0]
    weights = np.ones(N_basis, dtype=float)
    for n in range(N_basis):
        for k in range(d):
            weights[n] *= np.sqrt(2 * Lambda[n, k] + 1)
    return Lambda, weights


def build_design_matrix(xi_train, Lambda):
    """Build (m, N_basis) normalised Legendre design matrix."""
    m, d    = xi_train.shape
    N_basis = Lambda.shape[0]
    A = np.ones((m, N_basis), dtype=float)
    for n in range(N_basis):
        for kdim in range(d):
            Pk   = legendre(Lambda[n, kdim])
            norm = np.sqrt(2 * Lambda[n, kdim] + 1)
            A[:, n] *= norm * Pk(xi_train[:, kdim])
    return A


def map_to_canonical(X, pmin_d, pmax_d):
    return 2.0 * (X - pmin_d) / (pmax_d - pmin_d) - 1.0

# =============================================================================
# PDR hyperparameters  (Table 5.1)
# =============================================================================

def compute_table51_hparams(A_scaled, m):
    normA2  = np.linalg.norm(A_scaled, ord=2)
    tau     = 1.0 / normA2
    sigma   = 1.0 / normA2
    r       = np.exp(-1.0)
    T_inner = max(int(np.ceil((2.0 * normA2) / (r * np.sqrt(tau * sigma)))), 1)
    s       = T_inner / (2.0 * normA2)
    lam     = (np.sqrt(25.0 * m)) ** -1
    R       = max(int(np.ceil(TARGET_TOTAL_STEPS / T_inner)), 1)
    T_pd    = R * T_inner
    return dict(lam=lam, tau=tau, sigma=sigma, r=r,
                T_inner=T_inner, s=s, R=R, T_pd=T_pd)

# =============================================================================
# DNN definition
# =============================================================================

class DNN(nn.Module):
    """
    Fixed architecture: input_dim=d → [50]×8 hidden (Tanh) → K outputs.
    Only the first linear layer changes width with d.
    """
    def __init__(self, input_dim, output_dim=K,
                 n_layers=DNN_LAYERS, hidden=DNN_HIDDEN):
        super().__init__()
        seq = [nn.Linear(input_dim, hidden), nn.Tanh()]
        for _ in range(n_layers):
            seq += [nn.Linear(hidden, hidden), nn.Tanh()]
        seq += [nn.Linear(hidden, output_dim)]
        self.net = nn.Sequential(*seq)

    def forward(self, x):
        return self.net(x)


def init_weights(module):
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        nn.init.zeros_(module.bias)


def train_dnn_timed(X_np, Y_np, device):
    """
    Train DNN and return wall-clock time (seconds).
    Uses torch.cuda.synchronize() before/after so GPU truly finishes
    before the clock stops — without this, async GPU ops cause underestimation.

    X_np : (m, d)  canonical float32 inputs
    Y_np : (m, K)  min-max scaled float32 outputs in [-1, 1]
    """
    torch.manual_seed(RNG_SEED)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(RNG_SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark     = False

    model = DNN(X_np.shape[1]).to(device)
    model.apply(init_weights)

    crit  = nn.MSELoss()
    opt   = optim.Adam(model.parameters(), lr=DNN_LR)
    gamma = (DNN_FINAL_LR / DNN_LR) ** (1.0 / max(DNN_EPOCHS, 1))
    sched = optim.lr_scheduler.ExponentialLR(opt, gamma=gamma)

    ds = TensorDataset(torch.from_numpy(X_np), torch.from_numpy(Y_np))
    dl = DataLoader(ds, batch_size=DNN_BATCH, shuffle=True, drop_last=False)

    # Wait for GPU to be fully idle before starting the clock
    if device.type == 'cuda':
        torch.cuda.synchronize()
    t0 = time.time()

    model.train()
    for _ in range(DNN_EPOCHS):
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()
        sched.step()

    # Wait for GPU to truly finish before stopping the clock
    if device.type == 'cuda':
        torch.cuda.synchronize()
    return time.time() - t0

# =============================================================================
# Main
# =============================================================================

def main():
    np.random.seed(RNG_SEED)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"PyTorch device : {device}")
    if device.type != 'cuda':
        print("WARNING: GPU not found — DNN timings will be very slow on CPU.")

    pmap_exec = parmap.Parmap(master=PARMASTER, mode=PARMODE, numWorkers=PARNWORKERS)

    # Pre-compute basis for every dimension
    basis_cache = {}
    print("\nHyperbolic cross basis sizes:")
    for d in DIMS:
        Lambda, weights = build_basis(d)
        N_basis = Lambda.shape[0]
        basis_cache[d] = dict(Lambda=Lambda, weights=weights, N_basis=N_basis)
        print(f"  d={d:2d}  →  N_basis = {N_basis:,}")

    # Timing storage
    times_pdr = {d: np.full(len(TRAIN_SIZES), np.nan) for d in DIMS}
    times_dnn = {d: np.full(len(TRAIN_SIZES), np.nan) for d in DIMS}

    # Open CSV once — rows are appended as results arrive
    csv_fh, csv_writer = open_csv(CSV_FILE)
    print(f"\nResults will be saved incrementally to: {CSV_FILE}")

    try:
        # =====================================================================
        # Outer loop: dimension
        # =====================================================================
        for d in DIMS:
            print(f"\n{'='*70}")
            print(f"  Dimension d = {d}  |  N_basis = {basis_cache[d]['N_basis']:,}")
            print(f"  Varying : {xnames[:d]}")
            if d < 11:
                print(f"  Fixed   : {dict(zip(xnames[d:], x_true[d:]))}")
            print(f"{'='*70}")

            pmin_d  = pmin[:d]
            pmax_d  = pmax[:d]
            Lambda  = basis_cache[d]['Lambda']
            weights = basis_cache[d]['weights']
            N_basis = basis_cache[d]['N_basis']

            # Generate full training pool once (m_max samples) for this d
            m_max = max(TRAIN_SIZES)
            np.random.seed(RNG_SEED * 100 + d)

            x_vary       = np.random.uniform(pmin_d, pmax_d, size=(m_max, d))
            x_full_train = np.tile(x_true, (m_max, 1))   # all fixed at x_true
            x_full_train[:, :d] = x_vary                  # overwrite first d cols

            print(f"  Calling CRM for {m_max} samples via parmap …")
            runs_train = [
                [INPUT_FILE, OUTPUT_FILE, NAMELIST, i + 1, x_full_train[i].tolist()]
                for i in range(m_max)
            ]
            F_full      = np.array(pmap_exec(runcrm, runs_train))
            F_train_max = F_full[:, 18:24]                # (m_max, K)
            print(f"  CRM done. Output shape: {F_train_max.shape}")

            # Global output scaling for DNN (computed on full pool)
            F_min = F_train_max.min(axis=0)
            F_rng = F_train_max.max(axis=0) - F_min
            F_rng = np.where(F_rng == 0, 1.0, F_rng)

            # Canonical inputs for the full pool
            xi_train_max = map_to_canonical(x_vary, pmin_d, pmax_d)

            # -----------------------------------------------------------------
            # Inner loop: training size m
            # -----------------------------------------------------------------
            for idx_m, m in enumerate(TRAIN_SIZES):
                print(f"  m = {m:5d}", end="  ", flush=True)

                xi_m = xi_train_max[:m]    # (m, d)
                F_m  = F_train_max[:m]     # (m, K)

                # ---- PDR ----------------------------------------------------
                A     = build_design_matrix(xi_m, Lambda)
                scale = 1.0 / np.sqrt(m)
                A_sc  = A   * scale
                B_sc  = F_m * scale

                hp_raw = compute_table51_hparams(A_sc, m)
                hp = {
                    'lambda' : hp_raw['lam'],
                    'tau'    : hp_raw['tau'],
                    'sigma'  : hp_raw['sigma'],
                    'r'      : hp_raw['r'],
                    'T_inner': hp_raw['T_inner'],
                    's'      : hp_raw['s'],
                    'R'      : hp_raw['R'],
                    'T_pd'   : hp_raw['T_pd'],
                    'run_on_gpu': True,
                }

                t0 = time.time()
                _ = solve_all_outputs_gpu_batch(A_sc, B_sc, weights, hp, method='pdr')
                t_pdr = time.time() - t0
                times_pdr[d][idx_m] = t_pdr

                # ---- DNN ----------------------------------------------------
                Y_sc  = (2.0 * (F_m - F_min) / F_rng - 1.0).astype(np.float32)
                X_dnn = xi_m.astype(np.float32)

                t_dnn = train_dnn_timed(X_dnn, Y_sc, device)
                times_dnn[d][idx_m] = t_dnn

                # ---- Save row immediately to CSV ----------------------------
                write_row(csv_fh, csv_writer, d, m, N_basis, t_pdr, t_dnn)

                print(f"PDR = {t_pdr:7.2f}s   DNN = {t_dnn:7.2f}s   → saved to CSV")

    finally:
        # Always close the CSV cleanly, even if an exception occurs
        csv_fh.close()
        print(f"\nCSV closed: {CSV_FILE}")

    # =========================================================================
    # Save full timing arrays
    # =========================================================================
    np.savez(
        'scalability_times.npz',
        train_sizes = np.array(TRAIN_SIZES),
        dims        = np.array(DIMS),
        **{f'pdr_d{d}'    : times_pdr[d]              for d in DIMS},
        **{f'dnn_d{d}'    : times_dnn[d]              for d in DIMS},
        **{f'Nbasis_d{d}' : basis_cache[d]['N_basis'] for d in DIMS},
    )
    print("Saved: scalability_times.npz")

    # =========================================================================
    # PLOT
    # =========================================================================
    sizes  = np.array(TRAIN_SIZES, dtype=float)
    H, L   = DNN_HIDDEN, DNN_LAYERS    # 50, 8

    cmap   = plt.cm.tab10
    colors = {d: cmap(i / 9) for i, d in enumerate(DIMS)}

    fig, ax = plt.subplots(figsize=(13, 8))

    for d in DIMS:
        c     = colors[d]
        N_d   = basis_cache[d]['N_basis']
        t_pdr = times_pdr[d]
        t_dnn = times_dnn[d]

        # Actual runtimes
        ax.plot(sizes, t_pdr, '-o',
                color=c, markersize=4, linewidth=2.0,
                label=f'PDR  d={d}  (N={N_d:,})')
        ax.plot(sizes, t_dnn, '--s',
                color=c, markersize=4, linewidth=2.0, alpha=0.80,
                label=f'DNN  d={d}')

        # Theoretical curves (normalised to first data point at m=100)
        theory_pdr  = sizes * float(N_d)
        theory_pdr *= t_pdr[0] / theory_pdr[0]

        step_cost   = d * H + L * H ** 2       # d×50 + 8×2500
        theory_dnn  = sizes * float(step_cost)
        theory_dnn *= t_dnn[0] / theory_dnn[0]

        ax.plot(sizes, theory_pdr, ':',  color=c, linewidth=1.4, alpha=0.55)
        ax.plot(sizes, theory_dnn, '-.', color=c, linewidth=1.1, alpha=0.45)

    # Dual legend
    style_handles = [
        Line2D([0],[0], color='k', ls='-',  marker='o', ms=5, lw=2.0,
               label='PDR — actual'),
        Line2D([0],[0], color='k', ls='--', marker='s', ms=5, lw=2.0, alpha=0.8,
               label='DNN — actual'),
        Line2D([0],[0], color='k', ls=':',  lw=1.6, alpha=0.7,
               label=r'PDR theory  $\mathcal{O}(m \cdot N(d))$'),
        Line2D([0],[0], color='k', ls='-.', lw=1.3, alpha=0.7,
               label=r'DNN theory  $\mathcal{O}(m \cdot (dH + LH^2))$'),
    ]
    dim_handles = [
        Line2D([0],[0], color=colors[d], lw=3,
               label=f'd = {d}   [N = {basis_cache[d]["N_basis"]:,}]')
        for d in DIMS
    ]

    leg1 = ax.legend(handles=style_handles, loc='upper left',
                     fontsize=10, title='Line style', title_fontsize=10,
                     framealpha=0.92)
    ax.add_artist(leg1)
    ax.legend(handles=dim_handles, loc='center left',
              fontsize=10, title='Dimension  (basis size)',
              title_fontsize=10, framealpha=0.92)

    ax.set_xlabel('Training Sample Size  $m$', fontsize=13)
    ax.set_ylabel('Wall-clock Training Time  (seconds)', fontsize=13)
    ax.set_title(
        'Scalability: PDR vs DNN (GPU)  —  Varying Dimension and Training Size\n'
        r'Dotted: PDR $\mathcal{O}(m \cdot N(d))$  |  '
        r'Dash-dot: DNN $\mathcal{O}(m \cdot (dH + LH^2))$',
        fontsize=12
    )
    ax.grid(True, which='both', alpha=0.3)
    ax.set_xlim(sizes[0] - 50, sizes[-1] + 100)

    plt.tight_layout()
    plt.savefig('scalability_pdr_dnn_bydim.png', dpi=150)
    plt.close()
    print("Saved: scalability_pdr_dnn_bydim.png")

    # =========================================================================
    # Console summary at m = 5000
    # =========================================================================
    print(f"\n{'='*60}")
    print(f"Summary at m = {TRAIN_SIZES[-1]} (wall-clock seconds):")
    print(f"{'d':>4}  {'N(d)':>8}  {'PDR (s)':>10}  {'DNN (s)':>10}  {'Speedup':>10}")
    print("-" * 60)
    for d in DIMS:
        t_p = times_pdr[d][-1]
        t_n = times_dnn[d][-1]
        su  = t_n / t_p if (not np.isnan(t_p) and t_p > 0) else float('inf')
        print(f"{d:>4}  {basis_cache[d]['N_basis']:>8,}  "
              f"{t_p:>10.2f}  {t_n:>10.2f}  {su:>9.1f}×")
    print(f"{'='*60}")
    print(f"\nAll done. Results in: {CSV_FILE}")


if __name__ == "__main__":
    main()
