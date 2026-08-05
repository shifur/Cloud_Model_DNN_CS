#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Inference Cost Benchmark: PDR vs DNN (GPU) — MCMC timing across dimensions
===========================================================================
Loads saved models from train_save_models_5000.py and runs MCMC for each
dimension d ∈ {2, 4, 6, 8, 11} using the m=5000 trained model.

PDREmulator — fully vectorised basis evaluation
------------------------------------------------
Uses numpy.polynomial.legendre.legval to evaluate ALL basis functions at
once with no Python loops over N_basis. This removes the Python loop
overhead that caused PDR to be slow at d=4+ despite having fewer
mathematical operations than DNN.

Speed comparison (d=4, N_basis=308):
  Version 1 (original, legendre() per step) : ~20+ hours
  Version 2 (precomputed polys, list comp)  : ~669 seconds
  Version 3 (fully vectorised, this file)   : ~few seconds

Mathematical result: IDENTICAL across all versions.

MCMC settings (same for both methods)
--------------------------------------
  nwalk = 64, nburn = 1000, nmcmc = 10000

Output
------
  inference_results.csv
  inference_cost_pdr_dnn_bydim.png
"""

import csv
import os
import sys
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
from numpy.polynomial.legendre import legval
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import torch
from torch import nn
import emcee
from tqdm import tqdm

sys.path.append('/home/ss24ce/.local/lib/python3.10/site-packages')

from surrogate_config_11d import Y_SIG_SIX, YMASK_FIG7
from crm_eval_11d_six import run_cloud_11d_six

# =============================================================================
# CONFIGURATION
# =============================================================================

RNG_SEED  = 42
DIMS      = [2, 4, 6, 8, 11]
M_SAVE    = 5000
MODEL_DIR = 'saved_models'

# MCMC settings — same for both PDR and DNN
NWALK = 64
NBURN = 1000
NMCMC = 10000

# 11D physical parameter setup
pmin   = np.array([50.0,   0.10, 50.0,   0.1,  0.05,  0.05,  0.05,  0.05,  0.05,  1.e-4, 2.e-6])
pmax   = np.array([1000.0, 1.0,  1200.0, 0.90, 5.0,   2.5,   2.5,   1.0,   1.0,   2.e-3, 1.e-3])
x_true = np.array([200.0,  0.3,  400.0,  0.4,  0.5,   0.5,   0.5,   0.2,   0.4,   1.e-3, 6.e-4])
x_sig  = np.array([20.0,   0.05, 20.0,   0.05, 0.05,  0.05,  0.05,  0.05,  0.05,  1.e-4, 1.e-5])
xnames = ['as','bs','ag','bg','N0r','N0s','N0g','rhos','rhog','qc0','qi0']

# DNN architecture (must match training script)
DNN_LAYERS = 8
DNN_HIDDEN = 50
K          = 6

# Output folder — timestamped so every run gets its own folder
RUN_TAG  = time.strftime("%Y%m%d_%H%M%S")
OUT_DIR  = os.path.join("inference_results", f"run_{RUN_TAG}")
os.makedirs(OUT_DIR, exist_ok=True)

# Output file names — inside OUT_DIR
CSV_FILE   = os.path.join(OUT_DIR, f'inference_results_{RUN_TAG}.csv')
PLOT_FILE  = os.path.join(OUT_DIR, f'inference_cost_pdr_dnn_bydim_{RUN_TAG}.png')
CSV_HEADER = ['d', 'N_basis', 'method', 't_mcmc_s',
              'acceptance_rate', 'nburn', 'nmcmc', 'nwalk']

# =============================================================================
# CSV helpers
# =============================================================================

def open_csv(filepath):
    file_is_new = (not os.path.exists(filepath)) or \
                  (os.path.getsize(filepath) == 0)
    fh     = open(filepath, 'a', newline='')
    writer = csv.writer(fh)
    if file_is_new:
        writer.writerow(CSV_HEADER)
        fh.flush()
    return fh, writer


def write_row(fh, writer, d, N_basis, method, t_mcmc, acceptance_rate):
    writer.writerow([d, N_basis, method,
                     f'{t_mcmc:.6f}',
                     f'{acceptance_rate:.4f}',
                     NBURN, NMCMC, NWALK])
    fh.flush()

# =============================================================================
# DNN architecture (must match training script exactly)
# =============================================================================

class DNN(nn.Module):
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

# =============================================================================
# PDR emulator — FULLY VECTORISED version
# =============================================================================

class PDREmulator:
    """
    Loads PDR model from .npz file.

    Basis evaluation strategy (fully vectorised)
    ---------------------------------------------
    For each dimension k, we precompute a coefficient matrix at __init__:

        legendre_coeffs[k][n] = coefficients of P_{Lambda[n,k]}

    At each MCMC step, legval(xk, coeffs) evaluates ALL N_basis Legendre
    polynomials at xk simultaneously using numpy — zero Python loops over
    N_basis, zero legendre() object creations during MCMC.

    Speed (d=4, N_basis=308, 704,000 MCMC calls):
        Version 1 (original)   : ~20+ hours
        Version 2 (precomputed): ~669 seconds
        Version 3 (this)       : ~few seconds  ← fully vectorised
    """

    def __init__(self, npz_path):
        D            = np.load(npz_path, allow_pickle=True)
        self.Lambda  = D['Lambda']      # (N_basis, d)
        self.C_six   = D['C_six']      # (N_basis, K)
        self.pmin_d  = D['pmin_d']     # (d,)
        self.pmax_d  = D['pmax_d']     # (d,)
        self.d       = int(D['d'])
        self.m       = int(D['m'])
        self.N_basis = int(D['N_basis'])

        # Precompute normalisation constants — shape (N_basis, d)
        # norms[n, k] = sqrt(2 * Lambda[n,k] + 1)
        self.norms = np.sqrt(2.0 * self.Lambda + 1.0).astype(float)

        # Precompute Legendre coefficient vectors for ALL (n, k) pairs
        # leg_coeffs[k] : list of N_basis arrays, one per basis function
        #   leg_coeffs[k][n] = coefficient array for P_{Lambda[n,k]}
        #                      e.g. P_2 → [0, 0, 1] in Legendre basis
        #   legval(x, leg_coeffs[k][n]) evaluates P_{Lambda[n,k]}(x)
        #
        # Stored as (N_basis, max_deg+2) padded array for fast numpy ops
        # leg_mat[k] shape: (N_basis, max_deg_k+2)
        print(f"    Precomputing Legendre coefficient matrices "
              f"for d={self.d} … ", end="", flush=True)
        t0 = time.time()

        self.leg_mat = []   # leg_mat[k] : (N_basis, max_deg_k+2)
        for k in range(self.d):
            degs_k   = self.Lambda[:, k].astype(int)   # (N_basis,)
            max_deg  = int(degs_k.max())
            # Each row is the Legendre coefficient vector for one basis fn
            # P_n in Legendre basis = unit vector e_n
            mat = np.zeros((self.N_basis, max_deg + 2), dtype=float)
            for n in range(self.N_basis):
                deg = degs_k[n]
                mat[n, deg] = 1.0    # e_deg in Legendre basis → P_deg
            self.leg_mat.append(mat)  # (N_basis, max_deg+2)

        print(f"done in {time.time()-t0:.3f}s")

    def _map_canonical(self, x_d):
        return 2.0 * (x_d - self.pmin_d) / (self.pmax_d - self.pmin_d) - 1.0

    def _design_row_vectorised(self, xi_row):
        """
        Fully vectorised basis evaluation — no Python loops over N_basis.

        For each dimension k:
          legval(xk, mat[n]) = P_{Lambda[n,k]}(xk)  for ALL n at once

        xi_row : (d,) canonical input in [-1, 1]^d
        returns : (N_basis,) basis vector φ(xi_row)
        """
        phi = np.ones(self.N_basis, dtype=float)

        for k in range(self.d):          # loop over d only (2 to 11 iters)
            xk  = float(xi_row[k])
            mat = self.leg_mat[k]        # (N_basis, max_deg+2)

            # Evaluate ALL N_basis Legendre polynomials at xk simultaneously
            # legval(x, c) where c is (N_basis, max_deg+2):
            #   returns (N_basis,) — one value per row
            vals_k = legval(xk, mat.T)   # legval expects coeffs as columns
                                         # mat.T shape: (max_deg+2, N_basis)
                                         # returns: (N_basis,)

            phi *= self.norms[:, k] * vals_k

        return phi                       # (N_basis,)

    def predict(self, x_full):
        """
        x_full : (11,) full parameter vector
        Returns : (K,) predicted outputs — mathematically identical to
                  all previous versions, just computed faster.
        """
        x_d = x_full[:self.d]
        xi  = self._map_canonical(x_d)
        phi = self._design_row_vectorised(xi)   # (N_basis,)
        y   = phi @ self.C_six                  # (K,)
        return y

# =============================================================================
# DNN emulator — loads from .pt, predicts for d-dimensional input
# =============================================================================

class DNNEmulator:
    """
    Loads DNN model from .pt checkpoint.
    predict() accepts full 11D input — uses only first d dimensions.
    """
    def __init__(self, pt_path, device=None):
        self.device = device or torch.device('cpu')
        ckpt        = torch.load(pt_path, map_location=self.device)

        self.d      = int(ckpt['d'])
        self.Ymin   = ckpt['Ymin']
        self.Yrange = ckpt['Yrange']
        self.pmin_d = ckpt['pmin_d']
        self.pmax_d = ckpt['pmax_d']

        self.model  = DNN(self.d,
                          n_layers=int(ckpt['layers']),
                          hidden=int(ckpt['hidden'])).to(self.device)
        self.model.load_state_dict(ckpt['state_dict'])
        self.model.eval()

        self.Yrange = np.where(self.Yrange == 0, 1.0, self.Yrange)

    def _map_canonical(self, x_d):
        return 2.0 * (x_d - self.pmin_d) / (self.pmax_d - self.pmin_d) - 1.0

    def predict(self, x_full):
        x_d  = x_full[:self.d].astype(np.float32)
        xi   = self._map_canonical(x_d).astype(np.float32)
        xi_t = torch.from_numpy(xi).unsqueeze(0).to(self.device)
        with torch.no_grad():
            y_sc = self.model(xi_t).cpu().numpy()[0]
        return (y_sc + 1.0) * 0.5 * self.Yrange + self.Ymin

# =============================================================================
# Log-probability functions for emcee
# =============================================================================

def log_prob_pdr(xp, emul, x_true_full, L1_six,
                 y_sig_six, pmin_full, pmax_full, PMask, y_mask):
    x       = x_true_full.copy()
    pidx    = np.nonzero(PMask)[0]
    x[pidx] = xp

    if np.any(x < pmin_full) or np.any(x > pmax_full):
        return -np.inf

    y_pred   = emul.predict(x)
    sel      = (np.asarray(y_mask, dtype=int) == 1)
    obs      = y_pred[sel]
    mu       = np.asarray(L1_six, dtype=float)[sel]
    sig      = np.asarray(y_sig_six, dtype=float)[sel]
    residual = obs - mu
    log_like = -0.5 * np.sum((residual / sig) ** 2 +
                              np.log(2.0 * np.pi * sig ** 2))
    return float(log_like)


def log_prob_dnn(xp, emul, x_true_full, L1_six,
                 y_sig_six, pmin_full, pmax_full, PMask, y_mask):
    x       = x_true_full.copy()
    pidx    = np.nonzero(PMask)[0]
    x[pidx] = xp

    if np.any(x < pmin_full) or np.any(x > pmax_full):
        return -np.inf

    y_pred   = emul.predict(x)
    sel      = (np.asarray(y_mask, dtype=int) == 1)
    obs      = y_pred[sel]
    mu       = np.asarray(L1_six, dtype=float)[sel]
    sig      = np.asarray(y_sig_six, dtype=float)[sel]
    residual = obs - mu
    log_like = -0.5 * np.sum((residual / sig) ** 2 +
                              np.log(2.0 * np.pi * sig ** 2))
    return float(log_like)

# =============================================================================
# MCMC runner with tqdm progress display
# =============================================================================

def run_mcmc_timed(log_prob_fn, emul, d, L1_six,
                   y_sig_six, y_mask, method_name=''):
    """
    Run emcee with tqdm progress bars for burn-in and production.
    Returns (wall_clock_seconds, mean_acceptance_rate).
    """
    np.random.seed(RNG_SEED)

    # PMask: first d parameters active, rest fixed at x_true
    PMask      = np.zeros(11, dtype=float)
    PMask[:d]  = 1.0

    # Initial walker positions
    p0 = np.tile(x_true, (NWALK, 1))
    for j in range(d):
        lo, hi    = pmin[j], pmax[j]
        p0[:, j] += np.random.normal(0.0, x_sig[j], NWALK)
        p0[:, j]  = np.clip(p0[:, j], lo, hi)
    p0_active = p0[:, :d]              # (nwalk, d)

    sampler = emcee.EnsembleSampler(
        NWALK, d, log_prob_fn,
        args=[emul, x_true, L1_six, y_sig_six,
              pmin, pmax, PMask, y_mask]
    )

    # ---- Burn-in with tqdm progress bar ------------------------------------
    print(f"\n  [{method_name}] "
          f"Burn-in ({NBURN} steps × {NWALK} walkers):")
    t0 = time.time()

    with tqdm(total=NBURN,
              desc=f"    burn-in",
              unit="step", ncols=75,
              bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} '
                         '[{elapsed}<{remaining}, {rate_fmt}]') as pbar:
        for _ in sampler.sample(p0_active, iterations=NBURN,
                                progress=False, store=True):
            pbar.update(1)

    state = sampler.get_last_sample()
    sampler.reset()

    # ---- Production with tqdm progress bar ---------------------------------
    print(f"  [{method_name}] "
          f"Production ({NMCMC} steps × {NWALK} walkers):")

    with tqdm(total=NMCMC,
              desc=f"    production",
              unit="step", ncols=75,
              bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} '
                         '[{elapsed}<{remaining}, {rate_fmt}]') as pbar:
        for _ in sampler.sample(state, iterations=NMCMC,
                                progress=False, store=True):
            pbar.update(1)

    t_mcmc = time.time() - t0
    acc    = float(np.mean(sampler.acceptance_fraction))

    print(f"  [{method_name}] Done — "
          f"total={t_mcmc:.2f}s  acceptance={acc:.3f}")
    return t_mcmc, acc

# =============================================================================
# Main
# =============================================================================

def main():
    np.random.seed(RNG_SEED)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"PyTorch device : {device}")

    # Compute CRM truth observation once
    print("\nComputing CRM truth observation at x_true …")
    L1_six    = run_cloud_11d_six(x_true[None, :])[0]
    y_sig_six = Y_SIG_SIX
    y_mask    = YMASK_FIG7
    print(f"L1_six = {L1_six}")

    # Storage
    times_pdr = {}
    times_dnn = {}
    acc_pdr   = {}
    acc_dnn   = {}
    N_basis_d = {}

    csv_fh, csv_writer = open_csv(CSV_FILE)
    print(f"\nOutput folder  : {os.path.abspath(OUT_DIR)}")
    print(f"CSV file       : {CSV_FILE}")
    print(f"Plot file      : {PLOT_FILE}")
    print(f"MCMC settings  : nwalk={NWALK}, nburn={NBURN}, nmcmc={NMCMC}")

    try:
        for d in DIMS:
            print(f"\n{'='*60}")
            print(f"  Dimension d = {d}")
            print(f"{'='*60}")

            # ---- Load PDR model --------------------------------------------
            pdr_path = os.path.join(MODEL_DIR, f'd{d}_m{M_SAVE}_pdr.npz')
            if not os.path.exists(pdr_path):
                print(f"  [SKIP] PDR model not found: {pdr_path}")
                continue
            pdr_emul     = PDREmulator(pdr_path)
            N_basis      = pdr_emul.N_basis
            N_basis_d[d] = N_basis
            print(f"  PDR model loaded | N_basis = {N_basis:,}")

            # ---- Load DNN model --------------------------------------------
            dnn_path = os.path.join(MODEL_DIR, f'd{d}_m{M_SAVE}_dnn.pt')
            if not os.path.exists(dnn_path):
                print(f"  [SKIP] DNN model not found: {dnn_path}")
                continue
            dnn_emul = DNNEmulator(dnn_path, device=device)
            print(f"  DNN model loaded | d={dnn_emul.d}, "
                  f"layers={DNN_LAYERS}, hidden={DNN_HIDDEN}")

            # ---- Run PDR MCMC ----------------------------------------------
            t_pdr, a_pdr = run_mcmc_timed(
                log_prob_pdr, pdr_emul, d,
                L1_six, y_sig_six, y_mask,
                method_name=f'PDR d={d}')
            times_pdr[d] = t_pdr
            acc_pdr[d]   = a_pdr

            # ---- Run DNN MCMC ----------------------------------------------
            t_dnn, a_dnn = run_mcmc_timed(
                log_prob_dnn, dnn_emul, d,
                L1_six, y_sig_six, y_mask,
                method_name=f'DNN d={d}')
            times_dnn[d] = t_dnn
            acc_dnn[d]   = a_dnn

            # ---- Save to CSV immediately -----------------------------------
            write_row(csv_fh, csv_writer,
                      d, N_basis, 'PDR', t_pdr, a_pdr)
            write_row(csv_fh, csv_writer,
                      d, N_basis, 'DNN', t_dnn, a_dnn)
            print(f"  → d={d} results saved to CSV")

    finally:
        csv_fh.close()
        print(f"\nCSV closed : {CSV_FILE}")

    # =========================================================================
    # PLOT
    # =========================================================================
    dims_done = [d for d in DIMS if d in times_pdr and d in times_dnn]
    if len(dims_done) == 0:
        print("No results to plot.")
        return

    d_arr     = np.array(dims_done, dtype=float)
    t_pdr_arr = np.array([times_pdr[d] for d in dims_done])
    t_dnn_arr = np.array([times_dnn[d] for d in dims_done])

    H, L        = DNN_HIDDEN, DNN_LAYERS
    total_steps = (NBURN + NMCMC) * NWALK

    # PDR theory : total_steps × N(d) × (d + K)
    theory_pdr  = np.array([
        total_steps * N_basis_d[d] * (d + K)
        for d in dims_done], dtype=float)
    theory_pdr *= t_pdr_arr[0] / theory_pdr[0]

    # DNN theory : total_steps × (d×H + L×H² + H×K)
    theory_dnn  = np.array([
        total_steps * (d * H + L * H ** 2 + H * K)
        for d in dims_done], dtype=float)
    theory_dnn *= t_dnn_arr[0] / theory_dnn[0]

    fig, ax = plt.subplots(figsize=(10, 7))

    ax.plot(d_arr, t_pdr_arr, '-o',
            color='steelblue', markersize=8, linewidth=2.5,
            label='PDR — actual')
    ax.plot(d_arr, t_dnn_arr, '--s',
            color='darkorange', markersize=8, linewidth=2.5,
            label='DNN — actual')
    ax.plot(d_arr, theory_pdr, ':',
            color='steelblue', linewidth=1.8, alpha=0.65,
            label=r'PDR theory  $\mathcal{O}(N(d)\cdot(d+K))$')
    ax.plot(d_arr, theory_dnn, '-.',
            color='darkorange', linewidth=1.5, alpha=0.65,
            label=r'DNN theory  $\mathcal{O}(dH + LH^2 + HK)$')

    # Annotate N(d) on PDR data points
    for i, d in enumerate(dims_done):
        ax.annotate(
            f'N={N_basis_d[d]:,}',
            xy=(d, t_pdr_arr[i]),
            xytext=(5, 8), textcoords='offset points',
            fontsize=8, color='steelblue'
        )

    ax.set_xticks(dims_done)
    ax.set_xticklabels([f'd={d}' for d in dims_done], fontsize=11)
    ax.set_xlabel('Dimension  $d$', fontsize=13)
    ax.set_ylabel('MCMC Wall-clock Time  (seconds)', fontsize=13)
    ax.set_title(
        f'Inference Cost: PDR vs DNN (GPU)\n'
        f'emcee: nwalk={NWALK}, nburn={NBURN}, nmcmc={NMCMC}  |  '
        f'model trained on m={M_SAVE}',
        fontsize=12
    )
    ax.legend(fontsize=10, framealpha=0.92)
    ax.grid(True, which='both', alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOT_FILE, dpi=150)
    plt.close()
    print(f"Saved: {PLOT_FILE}")

    # =========================================================================
    # Console summary
    # =========================================================================
    print(f"\n{'='*65}")
    print(f"Inference Cost Summary  "
          f"(nwalk={NWALK}, nburn={NBURN}, nmcmc={NMCMC})")
    print(f"{'d':>4}  {'N(d)':>8}  {'PDR (s)':>10}  "
          f"{'DNN (s)':>10}  {'PDR acc':>8}  {'DNN acc':>8}")
    print("-" * 65)
    for d in dims_done:
        print(f"{d:>4}  {N_basis_d[d]:>8,}  "
              f"{times_pdr[d]:>10.2f}  {times_dnn[d]:>10.2f}  "
              f"{acc_pdr[d]:>8.3f}  {acc_dnn[d]:>8.3f}")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
