"""
interpobality_4d_updated_v2.py
==============================
4x4 pairwise coefficient interaction maps (as, bs, ag, bg) -- PCP and OLR.

ONE global 11-D SR-LASSO surrogate (full hyperbolic cross, p_poly=20 ->
4291 basis) is fit for the two target outputs. For each parameter pair
(i, j) among the four fall-speed parameters we extract the coefficients of
basis functions that are nonzero ONLY in dims i and j (all seven other
dimensions at degree zero); those give the (nu_i, nu_j) interaction tile.
The diagonal shows each parameter's solo 1-D decay strip.

Rendering:
  - Every panel shares one LogNorm + colormap, so colors are directly,
    pixel-for-pixel comparable between PCP and OLR.
  - Cells present in the hyperbolic-cross basis but with negligible
    coefficients are shown at the colormap floor (VMIN), not left blank.
  - (degree_i, degree_j) combinations that were never part of the
    truncated basis at all (pruned by the p_poly hyperbolic-cross rule,
    not by SR-LASSO) are NaN-masked and drawn in neutral light gray, so
    "structurally excluded" is distinguishable from "included but small."

White-cell-border styling is controlled by EDGE_CASES (section 4b). The
default run produces BOTH the hairline ("thin") and the no-line ("noline")
versions. The "noline" version -- edgecolors="face", linewidth=0.0 -- is
the one with no white line anywhere; "face" (rather than "none") avoids
faint seams on PDF export.

  files written:  pairwise_corner_<NAME>_asbsagbg_<TAG>.png
                  e.g. pairwise_corner_OLR_asbsagbg_noline.png

Parameter layout (index : name):
  0:as 1:bs 2:ag 3:bg 4:N0r 5:N0s 6:N0g 7:rhos 8:rhog 9:qc0 10:qi0
Output slice F[:, 18:24] = [PCP, ACC, LWP, IWP, OLR, OSR]; we fit PCP and OLR.

NOTE: this file was reconstructed to match the repo's conventions
(multiidx_gen / legendre norm / cvxpy-CLARABEL / parmap+runcrm). Verify the
knobs below against your last run before trusting the numbers.
"""

import os, sys, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.cm import ScalarMappable
from matplotlib import colormaps
from scipy.special import legendre
import cvxpy as cp

# custom cluster packages (same as coefficient_decay)
sys.path.append('/home/ss24ce/.local/lib/python3.10/site-packages')
from parmap_framework import parmap
from module_runcrm import runcrm

OUTDIR = "interpretability"
os.makedirs(OUTDIR, exist_ok=True)

# ----- knobs ----------------------------------------------------------------
p_poly  = 20         # 11-D hyperbolic-cross degree (4291 basis at p=20)
m_train = 2000       # CRM training samples (THE EXPENSIVE PART)
mu_val  = 1e-5       # SR-LASSO regularization (larger -> sparser)
VMIN    = 1e-3       # color/display floor (relative coefficient)
THUMB   = 10         # crop each pair thumbnail to degrees 0..THUMB
TARGET_OUTPUTS = {'PCP': 0, 'OLR': 4}      # name -> column in the 6-output slice
SUBSET  = [0, 1, 2, 3]                      # as, bs, ag, bg
# ----------------------------------------------------------------------------

D = 11
pnames = ['as', 'bs', 'ag', 'bg', 'N0r', 'N0s', 'N0g', 'rs', 'rg', 'qc0', 'qi0']
pmin = np.array([50, 0.10, 50, 0.10, 0.05, 0.05, 0.05, 0.05, 0.05, 1e-4, 2e-6])
pmax = np.array([1000, 1.00, 1200, 0.90, 5.00, 2.50, 2.50, 1.00, 1.00, 2e-3, 1e-3])

def to_canonical(X):                       # (m,11) physical -> (m,11) in [-1,1]
    return 2.0 * (X - pmin) / (pmax - pmin) - 1.0

# =============================================================================
# 1. 11-D hyperbolic-cross multi-index set  (same rule as coefficient_decay)
# =============================================================================
np.random.seed(42)
HCfunc = lambda idx: np.prod(idx + 1) - 1

def multiidx_gen(N, rule, w, base=0, multiidx=np.array([]), MULTI_IDX=np.array([])):
    if len(multiidx) != N:
        i = base
        while rule(np.append(multiidx, i)) <= w:
            MULTI_IDX = multiidx_gen(N, rule, w, base,
                                     np.append(multiidx, i), MULTI_IDX)
            i += 1
    else:
        MULTI_IDX = np.vstack([MULTI_IDX, multiidx]) if MULTI_IDX.size else multiidx
    return MULTI_IDX

Lambda = multiidx_gen(D, HCfunc, p_poly).astype(int)     # (N_basis, 11)
N_basis = Lambda.shape[0]
print(f"11-D hyperbolic-cross basis size: {N_basis}")

# tuple -> column index, for O(1) lookup of a specific multi-index
idx_of = {tuple(row): n for n, row in enumerate(Lambda)}

def legvec(deg, x):                        # orthonormal 1-D Legendre on [-1,1]
    return np.sqrt((2 * deg + 1) / 2.0) * legendre(int(deg))(x)

def design_matrix(Xi):                      # Xi canonical (m,11) -> (m,N_basis)
    m = Xi.shape[0]
    Psi = np.ones((m, N_basis))
    for n in range(N_basis):
        col = np.ones(m)
        for dim in range(D):
            deg = Lambda[n, dim]
            if deg > 0:
                col *= legvec(deg, Xi[:, dim])
        Psi[:, n] = col
    return Psi

# =============================================================================
# 2. Draw training points and run the CRM  (parmap + runcrm, as in the repo)
# =============================================================================
X_phys = pmin + np.random.rand(m_train, D) * (pmax - pmin)
Xi_tr  = to_canonical(X_phys)

runs_train = [[
    './cloud_column_model/run_one_crm1d.txt',
    './cloud_column_model/crm1d_output.txt',
    './cloud_column_model/namelist_3h_t30-180.f90',
    i + 1,
    X_phys[i].tolist()
] for i in range(m_train)]

print("Running CRM for CS training (11-D)…")
pmap = parmap.Parmap(master='scispark6.jpl.nasa.gov:8786', mode='par', numWorkers=12)
F_full  = np.array(pmap(runcrm, runs_train))
F_train = F_full[:, 18:24]                 # [PCP, ACC, LWP, IWP, OLR, OSR]

# =============================================================================
# 3. Global SR-LASSO fit (one solve per target output)
# =============================================================================
Psi = design_matrix(Xi_tr)
A = Psi / np.sqrt(m_train)
b = F_train / np.sqrt(m_train)

coeffs = {}
t0 = time.time()
for name, k in TARGET_OUTPUTS.items():
    xvar = cp.Variable(N_basis)
    obj  = mu_val * cp.norm1(xvar) + cp.norm(A @ xvar - b[:, k], 2)
    cp.Problem(cp.Minimize(obj)).solve(solver=cp.CLARABEL)
    coeffs[name] = np.asarray(xvar.value)
    print(f"  {name}: nonzeros = {np.count_nonzero(np.abs(coeffs[name]) > 1e-12)}")
print(f"SR-LASSO solved in {time.time() - t0:.1f}s")

# =============================================================================
# 4. Extract per-pair tiles / per-parameter strips from the global coefficients
# =============================================================================
def solo_strip(c, i):
    """1-D decay for parameter i: |coeff| of the pure-degree-d index in dim i."""
    v = np.full(THUMB + 1, np.nan)
    for d in range(THUMB + 1):
        key = [0] * D; key[i] = d
        n = idx_of.get(tuple(key))
        if n is not None:
            v[d] = abs(c[n])
    return v

def pair_tile(c, i, j):
    """(deg_i, deg_j) interaction map for the pair (i, j).
    NaN where the combination is not in the hyperbolic-cross basis at all."""
    M = np.full((THUMB + 1, THUMB + 1), np.nan)
    for di in range(THUMB + 1):
        for dj in range(THUMB + 1):
            key = [0] * D; key[i] = di; key[j] = dj
            n = idx_of.get(tuple(key))
            if n is not None:
                M[di, dj] = abs(c[n])
    return M

# =============================================================================
# 4b. White-gridline styling cases
#     "noline" is the no-white-line version (edgecolors follow the cell face).
# =============================================================================
EDGE_CASES = {
    "thin":   dict(edgecolors="white", linewidth=0.3),   # hairline white grout
    "noline": dict(edgecolors="face",  linewidth=0.0),   # no white line anywhere
}

cmap = colormaps["inferno"].copy()
cmap.set_bad("0.85")                       # NaN / structurally-absent -> light gray
norm = LogNorm(vmin=VMIN, vmax=1.0)

def pcolor_tile(ax, M, edge_kw):
    n = M.shape[0]
    edges = np.arange(-0.5, n + 0.5, 1.0)
    ax.pcolormesh(edges, edges, np.ma.masked_invalid(M),
                  cmap=cmap, norm=norm, **edge_kw)
    ax.set_xlim(-0.5, n - 0.5); ax.set_ylim(-0.5, n - 0.5)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])

def pcolor_strip(ax, v, edge_kw):
    n = len(v)
    xe = np.arange(-0.5, n + 0.5, 1.0); ye = np.array([-0.5, 0.5])
    ax.pcolormesh(xe, ye, np.ma.masked_invalid(v)[None, :],
                  cmap=cmap, norm=norm, **edge_kw)
    ax.set_xlim(-0.5, n - 0.5); ax.set_ylim(-0.5, 0.5)
    ax.set_xticks([]); ax.set_yticks([])

# =============================================================================
# 5. Assemble the 4x4 corner figure for one output / one styling case
# =============================================================================
def subset_corner_plot(c, name, edge_kw, tag):
    # normalize to the output's largest coefficient so the LogNorm floor
    # (VMIN) and vmax=1.0 are meaningful "relative coefficient" values
    cmax = np.max(np.abs(c)) if np.max(np.abs(c)) > 0 else 1.0
    cn = c / cmax

    P = SUBSET
    npar = len(P)
    fig, axs = plt.subplots(npar, npar, figsize=(6, 6))

    for ri in range(npar):
        for ci in range(npar):
            ax = axs[ri, ci]
            if ci < ri:                     # hide lower triangle
                ax.axis("off"); continue
            if ri == ci:                    # diagonal: solo strip
                v = np.clip(solo_strip(cn, P[ri]), VMIN, 1.0)
                pcolor_strip(ax, v, edge_kw)
                for s in ax.spines.values():
                    s.set_color("navy"); s.set_linewidth(1.3)
            else:                           # off-diagonal: interaction tile
                M = pair_tile(cn, P[ri], P[ci])
                M = np.where(np.isnan(M), np.nan, np.clip(M, VMIN, 1.0))
                pcolor_tile(ax, M, edge_kw)
            if ri == 0:
                ax.set_title(pnames[P[ci]], fontsize=11)
            if ci == ri:
                ax.set_ylabel(pnames[P[ri]], fontsize=11,
                              rotation=0, ha="right", va="center")

    fig.suptitle(f"CS coefficient interaction maps — {name}", fontsize=12, y=0.98)

    # one shared colorbar
    sm = ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
    cbar = fig.colorbar(sm, ax=axs, fraction=0.046, pad=0.04)
    cbar.set_label("relative |coefficient|")

    fpath = os.path.join(OUTDIR, f"pairwise_corner_{name}_asbsagbg_{tag}.png")
    fig.savefig(fpath, dpi=130, bbox_inches="tight"); plt.close(fig)
    print(f"  saved {fpath}")

# =============================================================================
# 6. Run: every target output x every styling case
# =============================================================================
for name in TARGET_OUTPUTS:
    for tag, edge_kw in EDGE_CASES.items():
        subset_corner_plot(coeffs[name], name, edge_kw, tag)

print(f"\nDone. 4-parameter (as/bs/ag/bg) corner plots "
      f"({'/'.join(EDGE_CASES)}) in ./{OUTDIR}/")
