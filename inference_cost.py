"""
Inference Cost: Theoretical per-output (left) + Empirical timing d=11 (right)

UPDATED VERSION:
- SIX outputs for BOTH PDR and DNN timing
- PDR full inference      = recurrence + basis construction + contraction
- PDR precomputed         = Basis @ C only
- DNN inference           = six-output forward pass
- DNN weights loaded directly from the 6-output checkpoint
- FAIR comparison: both PDR and DNN run on GPU when CuPy is available

N_TEST=1000, K_exp=6, K_theory=1, d=11
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import os
import warnings
warnings.filterwarnings("ignore")

import torch
from torch import nn

# ── CuPy for PDR ─────────────────────────────────────────────────────────────
try:
    import cupy as cp
    _HAS_CUPY = True
    xp = cp
    print("CuPy found → PDR will run on GPU")
except ImportError:
    _HAS_CUPY = False
    xp = np
    print("CuPy NOT found → PDR falls back to CPU numpy (comparison not GPU-vs-GPU)")

# ── Parameters ────────────────────────────────────────────────────────────────
d = 11
H = 50
L = 8
K_exp = 6             # SIX outputs for experimental timed forward pass
K_theory = 1          # ONE output for theoretical per-output cost
N_d = 4291

N_TEST = 1000
N_RUNS = 5
RNG_SEED = 99

MODEL_DIR = 'saved_models'
M_SAVE = 5000

pmin = np.array([50., 0.10, 50., 0.1, 0.05, 0.05, 0.05, 0.05, 0.05, 1.e-4, 2.e-6])
pmax = np.array([1000., 1., 1200., 0.90, 5., 2.5, 2.5, 1., 1., 2.e-3, 1.e-3])

# ── Theoretical ops (per-output: K_theory=1) ───────────────────────────────
# DNN:
# input -> Linear(d,H) -> tanh
# then L times: Linear(H,H) -> tanh
# final: Linear(H,K_theory)
W_fwd = d * H + L * H * H + H * K_theory
b_fwd = (L + 1) * H + K_theory
s_fwd = (L + 1) * H
C_FWD_DNN = W_fwd + b_fwd + s_fwd

# PDR contraction-only theory (precomputed basis)
C_PDR_PRECOMP = N_d * K_theory

# PDR raw theory
C_PDR_RAW = N_d * (d + K_theory)

T_MCMC = 704_000
m_vals = np.arange(100, 5001, 100)

print(f"\nDNN  C_fwd (K_theory={K_theory}) = {W_fwd:,} + {b_fwd:,} + {s_fwd:,} = {C_FWD_DNN:,} ops per forward pass")
print(f"PDR  precomp (K_theory={K_theory}) = {C_PDR_PRECOMP:,} ops/sample  |  raw = {C_PDR_RAW:,} ops per forward pass")
print(f"Theory: DNN / PDR_precomp = {C_FWD_DNN / C_PDR_PRECOMP:.2f}×")

# ── Sync helpers ──────────────────────────────────────────────────────────────
def sync_cupy():
    if _HAS_CUPY:
        cp.cuda.Stream.null.synchronize()

def sync_torch(device):
    if device.type == 'cuda':
        torch.cuda.synchronize()

# ── DNN (six-output runtime model) ───────────────────────────────────────────
class DNN(nn.Module):
    def __init__(self, inp, out=K_exp, nl=L, hid=H):
        super().__init__()
        layers = [nn.Linear(inp, hid), nn.Tanh()]
        for _ in range(nl):
            layers += [nn.Linear(hid, hid), nn.Tanh()]
        layers += [nn.Linear(hid, out)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# ── PDR helpers ───────────────────────────────────────────────────────────────
def canon(X, lo, hi):
    return 2.0 * (X - lo) / (hi - lo) - 1.0

def leg_rec_gpu(x_dev, max_deg, dtype):
    """
    3-term Legendre recurrence on GPU (CuPy) or CPU (numpy).
    x_dev   : (N_TEST,)
    returns : (max_deg+1, N_TEST)
    """
    P = xp.empty((max_deg + 1, x_dev.shape[0]), dtype=dtype)
    P[0] = 1.0
    if max_deg >= 1:
        P[1] = x_dev
    for k in range(2, max_deg + 1):
        P[k] = ((2 * k - 1) * x_dev * P[k - 1] - (k - 1) * P[k - 2]) / k
    return P

def build_basis_gpu(xi_dev, Lambda_dev):
    """
    Build the full orthonormal basis matrix.

    xi_dev     : (N_TEST, d)
    Lambda_dev : (N_basis, d)

    Returns
    -------
    Basis_dev  : (N_TEST, N_basis)
    """
    n, dd = xi_dev.shape
    N_basis = Lambda_dev.shape[0]
    dtype = xi_dev.dtype

    Pdim = []
    for kdim in range(dd):
        max_deg = int(Lambda_dev[:, kdim].max())
        Pdim.append(leg_rec_gpu(xi_dev[:, kdim], max_deg, dtype))

    Basis_dev = xp.ones((n, N_basis), dtype=dtype)

    for kdim in range(dd):
        degs = Lambda_dev[:, kdim]                     # (N_basis,)
        vals = Pdim[kdim][degs, :].T                   # (N_TEST, N_basis)
        norms = xp.sqrt((2 * degs + 1).astype(dtype))  # (N_basis,)
        Basis_dev *= vals * norms[xp.newaxis, :]

    return Basis_dev

def pdr_predict_full_gpu(xi_dev, Lambda_dev, C_dev):
    """
    Full PDR inference:
    recurrence + basis construction + contraction
    """
    Basis_dev = build_basis_gpu(xi_dev, Lambda_dev)
    return Basis_dev @ C_dev

def pdr_predict_precomputed_gpu(Basis_dev, C_dev):
    """
    Basis-precomputed PDR inference:
    ONLY Basis @ C
    """
    return Basis_dev @ C_dev

# ── Load models ───────────────────────────────────────────────────────────────
device   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
pdr_path = os.path.join(MODEL_DIR, f'd{d}_m{M_SAVE}_pdr.npz')
dnn_path = os.path.join(MODEL_DIR, f'd{d}_m{M_SAVE}_dnn.pt')

print(f"\nPyTorch device : {device}")
print(f"PDR device     : {'GPU (CuPy)' if _HAS_CUPY else 'CPU (numpy)'}")
print(f"N_TEST         : {N_TEST}")
print(f"N_RUNS         : {N_RUNS}")

# Fixed test set
np.random.seed(RNG_SEED)
x_test = np.random.uniform(pmin, pmax, size=(N_TEST, d)).astype(np.float32)

# ── PDR load ──────────────────────────────────────────────────────────────────
data      = np.load(pdr_path, allow_pickle=True)
Lambda_np = data['Lambda'].astype(np.int32)
C_np      = data['C_six'].astype(np.float32)                 # use all SIX outputs
pmin_pd   = data['pmin_d']
pmax_pd   = data['pmax_d']
N_basis   = int(data['N_basis'])

xi_np = canon(x_test, pmin_pd, pmax_pd).astype(np.float32)

Lambda_dev = xp.asarray(Lambda_np)
C_dev      = xp.asarray(C_np)
xi_dev     = xp.asarray(xi_np)

print(f"\nPDR: N_basis={N_basis:,} | using SIX output columns: K_exp={K_exp}")
print(f"PDR: Lambda + C + xi pre-loaded to {'GPU (CuPy)' if _HAS_CUPY else 'CPU (numpy)'}")

# Build basis ONCE outside timing
Basis_dev = build_basis_gpu(xi_dev, Lambda_dev)
sync_cupy()
print(f"PDR: full basis precomputed once with shape {Basis_dev.shape}")

# Warm-up PDR
_ = pdr_predict_full_gpu(xi_dev, Lambda_dev, C_dev)
_ = pdr_predict_precomputed_gpu(Basis_dev, C_dev)
sync_cupy()

# Timed runs — PDR full
pdr_full_runs = []
for _ in range(N_RUNS):
    sync_cupy()
    t0 = time.perf_counter()
    _ = pdr_predict_full_gpu(xi_dev, Lambda_dev, C_dev)
    sync_cupy()
    pdr_full_runs.append(time.perf_counter() - t0)

t_pdr_full_mean = np.mean(pdr_full_runs)
t_pdr_full_min  = np.min(pdr_full_runs)

print(f"PDR full        mean={t_pdr_full_mean:.6f}s  min={t_pdr_full_min:.6f}s  "
      f"({t_pdr_full_mean / N_TEST * 1e6:.2f} µs/point)")

# Timed runs — PDR precomputed
pdr_pre_runs = []
for _ in range(N_RUNS):
    sync_cupy()
    t0 = time.perf_counter()
    _ = pdr_predict_precomputed_gpu(Basis_dev, C_dev)
    sync_cupy()
    pdr_pre_runs.append(time.perf_counter() - t0)

t_pdr_pre_mean = np.mean(pdr_pre_runs)
t_pdr_pre_min  = np.min(pdr_pre_runs)

print(f"PDR precomputed mean={t_pdr_pre_mean:.6f}s  min={t_pdr_pre_min:.6f}s  "
      f"({t_pdr_pre_mean / N_TEST * 1e6:.2f} µs/point)")

# ── DNN load: use full 6-output checkpoint ──────────────────────────────────
ckpt    = torch.load(dnn_path, map_location=device)
pmin_dn = ckpt['pmin_d']
pmax_dn = ckpt['pmax_d']

model = DNN(d, out=K_exp).to(device)
model.load_state_dict(ckpt['state_dict'], strict=True)
model.eval()

xi_dnn   = canon(x_test, pmin_dn, pmax_dn).astype(np.float32)
X_tensor = torch.from_numpy(xi_dnn).to(device)

print(f"DNN: weights + inputs pre-loaded to {device}")
print(f"DNN: using full SIX-output checkpoint: K_exp={K_exp}")

# Warm-up DNN
with torch.no_grad():
    _ = model(X_tensor[:5])
sync_torch(device)

# Timed runs — DNN six-output
dnn_runs = []
for _ in range(N_RUNS):
    sync_torch(device)
    t0 = time.perf_counter()
    with torch.inference_mode():
        _ = model(X_tensor)   # (N_TEST, K_exp)
    sync_torch(device)
    dnn_runs.append(time.perf_counter() - t0)

t_dnn_mean = np.mean(dnn_runs)
t_dnn_min  = np.min(dnn_runs)

print(f"DNN six-output  mean={t_dnn_mean:.6f}s  min={t_dnn_min:.6f}s  "
      f"({t_dnn_mean / N_TEST * 1e6:.2f} µs/point)")

# ── Comparison summary ────────────────────────────────────────────────────────
ratio_theory = C_FWD_DNN / C_PDR_PRECOMP
ratio_full   = t_dnn_mean / t_pdr_full_mean
ratio_pre    = t_dnn_mean / t_pdr_pre_mean

print(f"\n{'='*74}")
print(f"  GPU COMPARISON  (d={d}; experiment K_exp={K_exp}, theory K_theory={K_theory})")
print(f"{'='*74}")
print(f"  {'':24} {'PDR full':>12} {'PDR precomp':>14} {'DNN':>12}")
print(f"  {'-'*70}")
print(f"  {'Ops/point (theory)':24} {'—':>12} {C_PDR_PRECOMP:>14,} {C_FWD_DNN:>12,}")
print(f"  {'Wall-clock (s)':24} {t_pdr_full_mean:>12.6f} {t_pdr_pre_mean:>14.6f} {t_dnn_mean:>12.6f}")
print(f"  {'µs/point':24} {t_pdr_full_mean/N_TEST*1e6:>12.2f} "
      f"{t_pdr_pre_mean/N_TEST*1e6:>14.2f} {t_dnn_mean/N_TEST*1e6:>12.2f}")
print(f"  {'-'*70}")
print(f"  Theory (DNN / PDR_precomp) = {ratio_theory:.2f}×")
print(f"  Wall   (DNN / PDR_full)    = {ratio_full:.2f}×")
print(f"  Wall   (DNN / PDR_precomp) = {ratio_pre:.2f}×")
print(f"{'='*74}")

# =============================================================================
# PLOT
# =============================================================================
# PLOT (DNN vs CS ONLY)
# =============================================================================
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 13,
    'axes.linewidth': 1.0,
    'grid.linewidth': 0.5,
    'grid.alpha': 0.4,
    'grid.linestyle': '--'
})

COL_DNN = '#1f77b4'
COL_CS  = '#2ca02c'

fig, axes = plt.subplots(1, 2, figsize=(13, 6))

fig.suptitle(
    r'Inference Cost: DNN vs CS  ($d=11$, Experimental $K=6$, Theory $K=1$ per output)',
    fontsize=18, fontweight='bold', y=1.03
)

# ─────────────────────────────────────────────────────────────
# LEFT: EXPERIMENTAL RESULTS (DNN vs CS only)
# ─────────────────────────────────────────────────────────────
ax = axes[0]

cs_device_str = 'GPU (CuPy)' if _HAS_CUPY else 'CPU (numpy)'
dnn_device_str = 'GPU (PyTorch CUDA)' if device.type == 'cuda' else 'CPU (PyTorch)'

labels_emp = [
    f'DNN\n{dnn_device_str}',
    f'CS\n{cs_device_str}',
]

times_emp = [
    t_dnn_mean,
    t_pdr_pre_mean,   # renamed as CS
]

colors_emp = [COL_DNN, COL_CS]

bars = ax.bar(
    labels_emp,
    times_emp,
    color=colors_emp,
    width=0.55,
    edgecolor='black',
    linewidth=1.0
)

for bar, t in zip(bars, times_emp):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() * 1.15,
        f'{t:.5f}s\n({t / N_TEST * 1e6:.2f} µs/forward pass)',
        ha='center',
        va='bottom',
        fontsize=11,
        fontweight='bold'
    )

ax.set_yscale('log')
ax.set_ylabel('Wall-clock time (seconds, log scale)')
ax.set_title(
    f'Experimental Inference Time\n'
    f'$d={d}$, $K={K_exp}$, N_TEST={N_TEST}, {N_RUNS} runs (mean)',
    fontweight='bold'
)

ax.grid(axis='y', which='both')
ax.set_ylim(min(times_emp) / 3, max(times_emp) * 6)

# ─────────────────────────────────────────────────────────────
# RIGHT: THEORETICAL COST (DNN vs CS only)
# ─────────────────────────────────────────────────────────────
ax = axes[1]

labels_theory = ['DNN', 'CS']
ops_theory = [C_FWD_DNN, C_PDR_PRECOMP]
colors_theory = [COL_DNN, COL_CS]

bars = ax.bar(
    labels_theory,
    ops_theory,
    color=colors_theory,
    width=0.55,
    edgecolor='black',
    linewidth=1.0
)

for bar, val in zip(bars, ops_theory):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() * 1.12,
        f'{val:,}\nops per forward pass',
        ha='center',
        va='bottom',
        fontsize=11,
        fontweight='bold'
    )

ax.set_yscale('log')
ax.set_ylabel('Operations per forward pass (log scale)')
ax.set_title(
    'Theoretical Inference Cost: ops per forward pass (K=1 per output)\n'
    r'DNN: $C_{\mathrm{fwd}} = dH + LH^2 + HK_{\mathrm{theory}} + ((L+1)H + K_{\mathrm{theory}}) + (L+1)H$'
    '\n'
    r'CS: $C_{\mathrm{CS}} = N(d)\times K_{\mathrm{theory}}$',
    fontweight='bold'
)

ax.grid(axis='y', which='both')
ax.set_ylim(min(ops_theory) / 3, max(ops_theory) * 4)

# ─────────────────────────────────────────────────────────────
# FINALIZE
# ─────────────────────────────────────────────────────────────
plt.tight_layout()

out = 'inference_cost_dnn_vs_cs_six_outputs.png'
plt.savefig(out, dpi=200, bbox_inches='tight')
plt.close()

print(f"\nPlot saved → {out}")
