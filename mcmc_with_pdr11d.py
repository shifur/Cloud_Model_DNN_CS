# mcmc_with_pdr11d.py
# CHANGES from previous version (all marked ### CHANGE N):
#
#   CHANGE 1: nwalk, nburn, nmcmc imported from surrogate_config_11d
#             (was 64/500/3000 — far too small for 11D convergence)
#             Now 32/40000/400000 — identical to mcmc_with_dnn_updated_new.py
#
#   CHANGE 2: Added production-run timing block (total_prod_time_s,
#             per_step_time_ms) — identical structure to DNN MCMC
#
#   CHANGE 3: Added gelman_rubin_rhat() — same function as DNN MCMC
#
#   CHANGE 4: Added plot_traces() — same function as DNN MCMC
#
#   CHANGE 5: Added timing plot + txt output (plot_mcmc_timing, save_timing_txt)
#             — same functions as DNN MCMC
#
#   CHANGE 6: Timing stored as NetCDF global attributes
#             — same keys as DNN MCMC output file
#
#   CHANGE 7: _safe_float() helper added for NetCDF safety
#             — same as DNN MCMC
#
#   CHANGE 8: autocorr + R-hat diagnostics added post-chain
#             — same as DNN MCMC
#
#   Everything that was correct before is UNCHANGED:
#   simps wrapper, kde helpers, ED/JS helpers, plotting helpers,
#   log_prob_pdr11d import, multiprocessing.Pool parallelism,
#   NetCDF variable names/structure, ED/JS metric computation.

import os, time
import numpy as np
import xarray as xr
import emcee
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from multiprocessing import get_context

from surrogate_config_11d import (
    set_global_seed, SEED,
    XNAMES, X_TRUE_11D, PMIN_11, PMAX_11,
    Y_SIG_SIX, YMASK_FIG7, YMASK_FIG9,
    MCMC_NWALK, MCMC_NBURN, MCMC_NMCMC,   # ### CHANGE 1: imported from config
)
from crm_eval_11d_six import run_cloud_11d_six
import log_prob_pdr11d

# ------------------------------------------------------------------
#  PATHS TO CLOUD MCMC "TRUE" POSTERIOR CHAINS (11D full CRM)
# ------------------------------------------------------------------
TRUE_CHAIN_FILE_NORAD = "/home/ss24ce/last_time-2/plots/output/mcmc_crm1d__EXP_3_full11_ag400.00000_bg0.40000_norad.nc"
TRUE_CHAIN_FILE_RAD   = "/home/ss24ce/last_time-2/plots/output/mcmc_crm1d__EXP_3_full11_ag400.00000_bg0.40000_rad.nc"

# ------------------------------------------------------------------
#  OUTPUT FOLDER
# ------------------------------------------------------------------
RUN_TAG           = time.strftime("%Y%m%d_%H%M%S")
FINAL_DIR         = os.path.abspath(f"FINAL_PDR11D_{RUN_TAG}")
FINAL_OUTPUT_DIR  = os.path.join(FINAL_DIR, "output")
FINAL_PLOTS_DIR   = os.path.join(FINAL_DIR, "plots")
FINAL_METRICS_DIR = os.path.join(FINAL_DIR, "mcmc__PDR")
for _d in [FINAL_DIR, FINAL_OUTPUT_DIR, FINAL_PLOTS_DIR, FINAL_METRICS_DIR]:
    os.makedirs(_d, exist_ok=True)
print(f"[INFO] Final outputs will be saved under: {FINAL_DIR}")

# ==============================================================
#   simps fallback (SciPy old/new compatible)
# ==============================================================
try:
    from scipy.integrate import simps as _scipy_simps
    def simps(y, x=None, axis=-1):
        return _scipy_simps(y, x=x, axis=axis)
except Exception:
    try:
        from scipy.integrate import simpson as _scipy_simpson
        def simps(y, x=None, axis=-1):
            return _scipy_simpson(y, x=x, axis=axis)
    except Exception:
        import numpy as _np
        def simps(y, x=None, axis=-1):
            return _np.trapz(y, x=x, axis=axis)

# ==============================================================
#   NetCDF safety helper   ### CHANGE 7 (added — same as DNN MCMC)
# ==============================================================
def _safe_float(arr, dtype=np.float32, nan_fill=-1e30):
    """Force numeric dtype and remove NaN/Inf — matches DNN MCMC."""
    a = np.asarray(arr)
    if a.dtype == object:
        a = np.array(a.tolist(), dtype=np.float64)
    a = a.astype(dtype, copy=False)
    a = np.nan_to_num(a, nan=nan_fill, posinf=1e30, neginf=-1e30)
    return a

# ==============================================================
#   KDE HELPERS
# ==============================================================
def kde1d(x, grid, bw=0.15):
    kde = gaussian_kde(x, bw_method=bw)
    z = kde(grid)
    z /= simps(z, x=grid)
    return z

def kde2d(x, y, xgrid, ygrid, bw=0.15):
    kde = gaussian_kde(np.vstack([x, y]), bw_method=bw)
    Xg, Yg = np.meshgrid(xgrid, ygrid)
    Z = kde(np.vstack([Xg.ravel(), Yg.ravel()])).reshape(Xg.shape)
    Z /= simps(simps(Z, x=ygrid, axis=0), x=xgrid)
    return Xg, Yg, Z

def _grid_limits(samples, lo, hi, qlo=0.5, qhi=99.5, n=200):
    a = max(lo, np.percentile(samples, qlo))
    b = min(hi, np.percentile(samples, qhi))
    if (not np.isfinite(a)) or (not np.isfinite(b)) or (a >= b):
        a, b = lo, hi
    return np.linspace(a, b, n)

# ==============================================================
#   JS DIVERGENCE HELPERS
# ==============================================================
def kullback_leibler_divergence(P, Q):
    eps = 1e-12
    P2 = P + eps; Q2 = Q + eps
    return np.sum(P2 * np.log(P2 / Q2))

def jensen_shannon_divergence(P, Q):
    M = 0.5 * (P + Q)
    return 0.5 * kullback_leibler_divergence(P, M) + \
           0.5 * kullback_leibler_divergence(Q, M)

# ==============================================================
#   ENERGY DISTANCE HELPERS
# ==============================================================
def energy_distance(X, Y, max_n=2000, seed=None):
    rng = np.random.default_rng(seed)
    X = np.asarray(X, dtype=float); Y = np.asarray(Y, dtype=float)
    n, d = X.shape; m, d2 = Y.shape
    assert d == d2
    if n > max_n:
        X = X[rng.choice(n, size=max_n, replace=False)]; n = max_n
    if m > max_n:
        Y = Y[rng.choice(m, size=max_n, replace=False)]; m = max_n
    term_xy = (2.0/(n*m))   * np.sum(np.linalg.norm(X[:,None,:]-Y[None,:,:], axis=-1))
    term_xx = (1.0/(n*n))   * np.sum(np.linalg.norm(X[:,None,:]-X[None,:,:], axis=-1))
    term_yy = (1.0/(m*m))   * np.sum(np.linalg.norm(Y[:,None,:]-Y[None,:,:], axis=-1))
    D2 = max(term_xy - term_xx - term_yy, 0.0)
    return np.sqrt(D2), D2

def compute_pairwise_ed_matrix(X_pdr, X_true, xnames):
    X_pdr = np.asarray(X_pdr, dtype=float)
    X_true = np.asarray(X_true, dtype=float)
    d = len(xnames)
    ED_mat = np.zeros((d, d), dtype=float)
    for i in range(d):
        for j in range(d):
            if i == j: continue
            ED_mat[i,j], _ = energy_distance(X_pdr[:,[i,j]], X_true[:,[i,j]], max_n=2000)
    return ED_mat

def compute_pairwise_js_matrix(X_pdr, X_true, xnames, pmin, pmax, n_grid=80, bw_2d=0.2):
    X_pdr = np.asarray(X_pdr, dtype=float)
    X_true = np.asarray(X_true, dtype=float)
    d = len(xnames)
    JS_mat = np.zeros((d, d), dtype=float)
    for i in range(d):
        for j in range(d):
            if i == j: continue
            Xp = X_pdr[:,[i,j]]; Xt = X_true[:,[i,j]]
            x_grid = _grid_limits(np.concatenate([Xp[:,0],Xt[:,0]]), pmin[i], pmax[i], n=n_grid)
            y_grid = _grid_limits(np.concatenate([Xp[:,1],Xt[:,1]]), pmin[j], pmax[j], n=n_grid)
            _, _, P = kde2d(Xp[:,0], Xp[:,1], x_grid, y_grid, bw=bw_2d)
            _, _, Q = kde2d(Xt[:,0], Xt[:,1], x_grid, y_grid, bw=bw_2d)
            P_flat = P.ravel(); P_flat /= np.sum(P_flat)
            Q_flat = Q.ravel(); Q_flat /= np.sum(Q_flat)
            JS_mat[i,j] = jensen_shannon_divergence(P_flat, Q_flat)
    return JS_mat

# ==============================================================
#   HEATMAP PLOTTING
# ==============================================================
def plot_matrix_heatmap(M, labels, title, out_png, cmap='Blues', fmt="{:.2f}"):
    d = len(labels)
    fig, ax = plt.subplots(figsize=(1.2*d, 1.2*d), facecolor="white")
    im = ax.imshow(M, origin='lower', cmap=cmap,
                   vmin=np.nanmin(M), vmax=np.nanmax(M))
    ax.set_xticks(range(d)); ax.set_xticklabels(labels, rotation=90)
    ax.set_yticks(range(d)); ax.set_yticklabels(labels)
    for i in range(d):
        for j in range(d):
            ax.text(j, i, fmt.format(M[i,j]),
                    ha='center', va='center', fontsize=6, color='black')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.title(title); plt.tight_layout()
    os.makedirs(os.path.dirname(out_png) or '.', exist_ok=True)
    plt.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.close(fig)

# ==============================================================
#   EXISTING PLOTTING: ag-bg + 11x11 pairwise grid
# ==============================================================
def plot_ag_bg_marginal(Xa_plot, xnames, pmin, pmax, x_true, out_png, title_note=""):
    ag_i, bg_i = xnames.index('ag'), xnames.index('bg')
    ag, bg = Xa_plot[:, ag_i], Xa_plot[:, bg_i]
    Xg, Yg, Z = kde2d(ag, bg,
                      _grid_limits(ag, pmin[ag_i], pmax[ag_i]),
                      _grid_limits(bg, pmin[bg_i], pmax[bg_i]), bw=0.15)
    fig = plt.figure(figsize=(5.8, 5.2), facecolor='white')
    gs  = fig.add_gridspec(2, 2, width_ratios=[4,1.2], height_ratios=[1.2,4],
                           hspace=0.05, wspace=0.05)
    ax  = fig.add_subplot(gs[1,0])
    axx = fig.add_subplot(gs[0,0])
    axy = fig.add_subplot(gs[1,1])
    ax.contourf(Xg, Yg, Z, levels=60, alpha=0.85, cmap='rainbow')
    ax.plot([x_true[ag_i]], [x_true[bg_i]], 'kx', ms=9, mew=2)
    ax.set_xlabel('ag'); ax.set_ylabel('bg'); ax.grid(alpha=0.25)
    axx.hist(ag, bins=60, density=True, histtype='step', lw=1.0)
    axy.hist(bg, bins=60, density=True, histtype='step', lw=1.0, orientation='horizontal')
    axx.tick_params(labelbottom=False); axy.tick_params(labelleft=False)
    plt.suptitle(f'ag-bg marginal posterior {title_note}', y=0.98, fontsize=12)
    os.makedirs(os.path.dirname(out_png) or '.', exist_ok=True)
    plt.savefig(out_png, dpi=130, bbox_inches='tight')
    plt.close(fig)

def plot_pairwise_grid(Xa_plot, xnames, pmin, pmax, x_true,
                       subset=None, out_png='plots/fig_pairgrid_pdr11d.png',
                       title="", bw_1d=0.15, bw_2d=0.15, n_grid=160,
                       figsize_per_cell=1.6, dpi=150):
    if subset is None: subset = xnames[:]
    idxs = [xnames.index(n) for n in subset]; d = len(idxs)
    S = Xa_plot[:, idxs]
    g1 = [_grid_limits(S[:,j], pmin[idxs[j]], pmax[idxs[j]], n=n_grid) for j in range(d)]
    k1 = [kde1d(S[:,j], g1[j], bw=bw_1d) for j in range(d)]
    fig, axes = plt.subplots(d, d, figsize=(figsize_per_cell*d, figsize_per_cell*d),
                             facecolor='white', squeeze=False)
    plt.subplots_adjust(hspace=0.05, wspace=0.05)
    for r in range(d):
        for c in range(d):
            ax = axes[r, c]
            if r == c:
                ax.plot(g1[r], k1[r], lw=1.0); ax.set_yticks([])
                if x_true is not None:
                    ax.axvline(x_true[idxs[r]], ls='--', lw=1.0, color='k', alpha=0.7)
                ax.set_xlabel(subset[r] if r == d-1 else "")
                if r < d-1: ax.set_xticklabels([])
                ax.grid(alpha=0.12)
            elif r > c:
                x = S[:,c]; y = S[:,r]
                gx = _grid_limits(x, pmin[idxs[c]], pmax[idxs[c]], n=n_grid)
                gy = _grid_limits(y, pmin[idxs[r]], pmax[idxs[r]], n=n_grid)
                Xg, Yg, Z = kde2d(x, y, gx, gy, bw=bw_2d)
                ax.contourf(Xg, Yg, Z, levels=40, cmap='rainbow', alpha=0.9)
                if x_true is not None:
                    ax.plot([x_true[idxs[c]]], [x_true[idxs[r]]], 'kx', ms=7, mew=1.8)
                if r != d-1: ax.set_xticklabels([])
                else: ax.set_xlabel(subset[c])
                if c != 0: ax.set_yticklabels([])
                else: ax.set_ylabel(subset[r])
                ax.grid(alpha=0.08)
            else:
                ax.set_visible(False)
    if title: plt.suptitle(title, y=0.995, fontsize=12)
    os.makedirs(os.path.dirname(out_png) or '.', exist_ok=True)
    plt.savefig(out_png, dpi=dpi, bbox_inches='tight')
    plt.close(fig)

# ==============================================================
#   DIAGNOSTIC HELPERS   ### CHANGE 3 + 4 (added — same as DNN MCMC)
# ==============================================================
def gelman_rubin_rhat(chains_3d: np.ndarray):
    """Gelman-Rubin R-hat — identical to DNN MCMC version."""
    nsteps, nwalkers, nparams = chains_3d.shape
    rhat = np.empty(nparams, dtype=float)
    for k in range(nparams):
        x = chains_3d[:, :, k]
        n, m = float(nsteps), float(nwalkers)
        chain_means = np.mean(x, axis=0)
        chain_vars  = np.var(x, axis=0, ddof=1)
        B = n * np.var(chain_means, ddof=1)
        W = np.mean(chain_vars)
        Var_hat = ((n-1.0)/n) * W + (B/n)
        rhat[k] = np.sqrt(Var_hat / W)
    return rhat

def plot_traces(Xa, param_names, out_png):
    """Trace plots — identical to DNN MCMC version."""
    nsteps, nwalkers, nparams = Xa.shape
    fig, axes = plt.subplots(nparams, 1, figsize=(10, 2.0*nparams), sharex=True)
    if nparams == 1: axes = [axes]
    for k in range(nparams):
        axes[k].plot(Xa[:,:,k], alpha=0.3)
        axes[k].set_ylabel(param_names[k])
        axes[k].grid(alpha=0.2)
    axes[-1].set_xlabel("MCMC step")
    fig.suptitle("Trace plots (PDR 11D, all walkers)", y=0.99, fontsize=14)
    plt.tight_layout(rect=[0,0,1,0.97])
    os.makedirs(os.path.dirname(out_png) or '.', exist_ok=True)
    plt.savefig(out_png, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("Saved trace plots:", os.path.abspath(out_png))

# ==============================================================
#   TIMING HELPERS   ### CHANGE 5 (added — same as DNN MCMC)
# ==============================================================
def plot_mcmc_timing(timing_dict: dict, out_png: str):
    """Two-panel timing bar chart — identical structure to DNN MCMC."""
    label       = timing_dict['run_note'].replace('_','').upper()
    total_s     = timing_dict['total_time_s']
    total_h     = total_s / 3600.0
    per_step_ms = timing_dict['per_step_time_ms']
    nwalk       = timing_dict['nwalk']
    nmcmc       = timing_dict['nmcmc']

    fig, axes = plt.subplots(1, 2, figsize=(10, 5), facecolor='white')
    ax = axes[0]
    bar = ax.bar(['Total\n400K steps'], [total_s], color='steelblue',
                 width=0.4, edgecolor='black', linewidth=0.8)
    ax.bar_label(bar, labels=[f"{total_s:.1f} s\n({total_h:.2f} h)"],
                 padding=6, fontsize=11, fontweight='bold')
    ax.set_ylabel("Wall-clock time (seconds)")
    ax.set_title(f"Total MCMC time\n({nmcmc:,} steps, {nwalk} walkers, {label})")
    ax.set_ylim(0, total_s * 1.25); ax.yaxis.grid(True, alpha=0.3); ax.set_axisbelow(True)

    ax2 = axes[1]
    bar2 = ax2.bar(['Per MCMC\nstep'], [per_step_ms], color='darkorange',
                   width=0.4, edgecolor='black', linewidth=0.8)
    ax2.bar_label(bar2, labels=[f"{per_step_ms:.3f} ms"],
                  padding=6, fontsize=11, fontweight='bold')
    ax2.set_ylabel("Wall-clock time (milliseconds)")
    ax2.set_title(f"Per-step MCMC time\n(1 step = 1 ensemble proposal, {label})")
    ax2.set_ylim(0, per_step_ms * 1.25); ax2.yaxis.grid(True, alpha=0.3); ax2.set_axisbelow(True)

    fig.suptitle("PDR Surrogate MCMC - Computational Timing", fontsize=14,
                 fontweight='bold', y=1.01)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_png) or '.', exist_ok=True)
    plt.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved timing plot: {os.path.abspath(out_png)}")

def save_timing_txt(timing_dict: dict, out_txt: str):
    """Plain-text timing summary — identical to DNN MCMC."""
    os.makedirs(os.path.dirname(out_txt) or '.', exist_ok=True)
    with open(out_txt, 'w') as f:
        f.write("=== PDR MCMC Timing Summary ===\n")
        f.write(f"Case                  : {timing_dict['run_note']}\n")
        f.write(f"Walkers               : {timing_dict['nwalk']}\n")
        f.write(f"Production steps      : {timing_dict['nmcmc']:,}\n")
        f.write(f"Total production time : {timing_dict['total_time_s']:.2f} s  "
                f"({timing_dict['total_time_s']/3600:.4f} h)\n")
        f.write(f"Per-step time         : {timing_dict['per_step_time_ms']:.4f} ms\n")
    print(f"Saved timing summary : {os.path.abspath(out_txt)}")

# ==============================================================
#   TRUE POSTERIOR SAMPLING
# ==============================================================
def sample_true_from_cloud_chain(nc_path, nsamples=4000):
    if not os.path.exists(nc_path):
        raise FileNotFoundError(f"Cloud MCMC file not found: {nc_path}")
    ds = xr.open_dataset(nc_path)
    if "Xa" not in ds:
        raise RuntimeError("Cloud MCMC file missing 'Xa' variable")
    Xa = ds["Xa"].values
    if Xa.ndim != 3:
        raise RuntimeError(f"Expected Xa with 3 dims, got shape {Xa.shape}")
    nsamp, nch, npar = Xa.shape
    pts = Xa.reshape(-1, npar)
    ns  = min(nsamples, pts.shape[0])
    rng = np.random.default_rng(SEED + 999)
    return pts[rng.choice(pts.shape[0], size=ns, replace=False), :]

def subsample_cloud(X, max_n=4000, seed=None):
    X = np.asarray(X, dtype=float); n = X.shape[0]
    if n <= max_n: return X
    return X[np.random.default_rng(seed).choice(n, size=max_n, replace=False), :]

# ==============================================================
#   MAIN PDR MCMC
# ==============================================================
def run_case(USE_RADIATION: bool):
    set_global_seed(SEED)
    x_true = X_TRUE_11D.copy()
    PMask  = np.ones_like(x_true)

    y_mask   = YMASK_FIG9 if USE_RADIATION else YMASK_FIG7
    run_note = "_rad" if USE_RADIATION else "_norad"

    L1_six = run_cloud_11d_six(x_true[None, :])[0]

    # ### CHANGE 1: nwalk/nburn/nmcmc from config (was 64/500/3000)
    nwalk = MCMC_NWALK   # 32  — matches DNN MCMC
    nburn = MCMC_NBURN   # 40000  — matches DNN MCMC
    nmcmc = MCMC_NMCMC   # 400000 — matches DNN MCMC

    x_sig = np.array([20.0, 0.05, 20.0, 0.05,
                      0.05, 0.05, 0.05, 0.05, 0.05,
                      1.e-4, 1.e-5])

    p0 = np.tile(x_true, nwalk).reshape((nwalk, 11)) + \
         np.random.normal(0.0, 1.0, (nwalk, 11)) * x_sig
    for j in range(11):
        lo, hi = PMIN_11[j], PMAX_11[j]
        bad = (p0[:,j] < lo) | (p0[:,j] > hi)
        while np.any(bad):
            p0[bad, j] = x_true[j] + np.random.normal(0.0, x_sig[j], bad.sum())
            bad = (p0[:,j] < lo) | (p0[:,j] > hi)

    lp0, _ = log_prob_pdr11d.log_prob_pdr11d(
        p0[0], x_true, L1_six, Y_SIG_SIX, PMIN_11, PMAX_11, PMask, y_mask
    )
    if not np.isfinite(lp0):
        raise RuntimeError("Initial PDR log_prob is -inf; train PDR first.")

    nprocs = int(os.environ.get("MCMC_NPROCS", "8"))
    print(f"[INFO] emcee parallel pool: nprocs={nprocs}")

    dtype   = [("Hx", float, (1+6,))]
    ctx     = get_context("spawn")
    t0      = time.time()

    with ctx.Pool(processes=nprocs) as pool:
        sampler = emcee.EnsembleSampler(
            nwalk, 11, log_prob_pdr11d.log_prob_pdr11d,
            blobs_dtype=dtype,
            args=[x_true, L1_six, Y_SIG_SIX, PMIN_11, PMAX_11, PMask, y_mask, 1],
            pool=pool
        )
        state = sampler.run_mcmc(p0, nburn, progress=True)
        sampler.reset()

        # ### CHANGE 2: time the production run separately (same as DNN MCMC)
        t_prod_start = time.time()
        sampler.run_mcmc(state, nmcmc, progress=True)
        t_prod_end   = time.time()

    total_prod_time_s  = t_prod_end - t_prod_start
    per_step_time_ms   = (total_prod_time_s / nmcmc) * 1e3

    print(f"PDR MCMC {run_note} took {time.time()-t0:.1f}s")
    print(f"\n=== MCMC TIMING ({run_note}) ===")
    print(f"  Production steps      : {nmcmc:,}")
    print(f"  Total production time : {total_prod_time_s:.2f} s  "
          f"({total_prod_time_s/3600:.4f} h)")
    print(f"  Per-step time         : {per_step_time_ms:.4f} ms")
    print("================================\n")

    # ### CHANGE 5: save timing plot + txt (same as DNN MCMC)
    timing_dict = {
        'run_note'         : run_note,
        'nwalk'            : nwalk,
        'nmcmc'            : nmcmc,
        'total_time_s'     : total_prod_time_s,
        'per_step_time_ms' : per_step_time_ms,
    }
    plot_mcmc_timing(timing_dict,
                     os.path.join(FINAL_METRICS_DIR, f"fig_timing_pdr11d{run_note}.png"))
    save_timing_txt(timing_dict,
                    os.path.join(FINAL_METRICS_DIR, f"timing_pdr11d{run_note}.txt"))

    Xa      = sampler.get_chain()
    log_pxy = sampler.get_log_prob()
    blobs   = sampler.get_blobs()
    try:   blobs = blobs["Hx"]
    except Exception: pass
    log_pyx = blobs[:,:,0]
    HXa     = blobs[:,:,1:]

    acc = float(np.mean(sampler.acceptance_fraction))
    print(f"Mean acceptance (PDR{run_note}): {acc:.3f}")

    # ### CHANGE 8: autocorr + R-hat (same as DNN MCMC)
    try:
        tau_arr = sampler.get_autocorr_time(quiet=True)
        print("Autocorr times (PDR):", tau_arr)
    except Exception as e:
        print("Autocorr time failed (PDR):", repr(e)); tau_arr = None

    try:
        rhat = gelman_rubin_rhat(Xa)
        print("Gelman-Rubin R-hat (PDR):", rhat)
    except Exception as e:
        print("R-hat failed (PDR):", repr(e)); rhat = None

    # ### CHANGE 4: trace plots (same as DNN MCMC)
    plot_traces(Xa, XNAMES,
                os.path.join(FINAL_PLOTS_DIR, f"trace_pdr11d{run_note}.png"))

    # ### CHANGE 7: _safe_float for NetCDF safety (same as DNN MCMC)
    Xa_s      = _safe_float(Xa,      dtype=np.float32)
    HXa_s     = _safe_float(HXa,     dtype=np.float32)
    log_pxy_s = _safe_float(log_pxy, dtype=np.float32)
    log_pyx_s = _safe_float(log_pyx, dtype=np.float32)

    base   = f'_EXP_3_pdr11d{run_note}.nc'
    out_nc = os.path.join(FINAL_OUTPUT_DIR, base)

    ds = xr.Dataset(
        {"x_true": (["nx"], _safe_float(x_true, dtype=np.float32)),
         "PMask":  (["nx"], _safe_float(PMask.astype(float), dtype=np.float32)),
         "L1_six": (["ny"], _safe_float(L1_six, dtype=np.float32)),
         "L2_six": (["ny"], _safe_float(Y_SIG_SIX**2, dtype=np.float32)),
         "y_mask": (["ny"], _safe_float(y_mask.astype(float), dtype=np.float32)),
         "Xa":     (["nsamples","nchains","nx"], Xa_s),
         "HXa":    (["nsamples","nchains","ny"], HXa_s),
         "p_yx":   (["nsamples","nchains"], log_pyx_s),
         "p_xy":   (["nsamples","nchains"], log_pxy_s)},
        coords={"nsamples": np.arange(Xa_s.shape[0]),
                "nchains":  np.arange(Xa_s.shape[1]),
                "nx":       np.arange(11),
                "ny":       np.arange(6)},
        attrs={"xnames":              ",".join(XNAMES),
               "ynames":              "PCP@120,ACC@120,LWP@120,IWP@120,OLR@120,OSR@120",
               "Acceptance fraction": float(np.mean(sampler.acceptance_fraction)),
               # ### CHANGE 6: timing stored as NetCDF attrs (same as DNN MCMC)
               "MCMC_total_prod_time_s" : float(total_prod_time_s),
               "MCMC_per_step_time_ms"  : float(per_step_time_ms),
               "MCMC_nmcmc_steps"       : int(nmcmc)}
    )
    if tau_arr is not None:
        ds["tau"]  = (["nx"], _safe_float(tau_arr, dtype=np.float32))
        ds.tau.attrs["long_name"] = "Autocorrelation time per parameter (PDR 11D)"
    if rhat is not None:
        ds["rhat"] = (["nx"], _safe_float(rhat, dtype=np.float32))
        ds.rhat.attrs["long_name"] = "Gelman-Rubin R-hat per parameter (PDR 11D)"

    ds.to_netcdf(out_nc)
    print("Saved:", out_nc)

    # Plots
    Xa_plot = Xa_s.reshape(-1, Xa_s.shape[-1])
    plot_ag_bg_marginal(
        Xa_plot, XNAMES, PMIN_11, PMAX_11, x_true,
        out_png=os.path.join(FINAL_PLOTS_DIR, f'fig_agbg_marginal{base.replace(".nc",".png")}'),
        title_note=f"(PDR 11D; {'OLR/OSR' if y_mask[-2:].sum() else 'PCP/LWP/IWP'})"
    )
    plot_pairwise_grid(
        Xa_plot, XNAMES, PMIN_11, PMAX_11, x_true,
        out_png=os.path.join(FINAL_PLOTS_DIR, f'fig_pairgrid{base.replace(".nc",".png")}'),
        title=f"PDR 11D pairwise marginals {'(with rad)' if y_mask[-2:].sum() else '(no rad)'}"
    )

    # ED / JS metrics
    try:
        X_pdr_sub  = subsample_cloud(Xa_plot, max_n=4000, seed=SEED+101)
        X_true_sub = sample_true_from_cloud_chain(
            TRUE_CHAIN_FILE_RAD if USE_RADIATION else TRUE_CHAIN_FILE_NORAD,
            nsamples=4000)

        ED_self_pdr, _  = energy_distance(X_pdr_sub,  X_pdr_sub,  max_n=1500, seed=SEED+201)
        ED_self_true, _ = energy_distance(X_true_sub, X_true_sub, max_n=1500, seed=SEED+202)
        ED_pdr_true, _  = energy_distance(X_pdr_sub,  X_true_sub, max_n=2000, seed=SEED+203)
        ED_true_pdr, _  = energy_distance(X_true_sub, X_pdr_sub,  max_n=2000, seed=SEED+204)

        print("\n=== ENERGY DISTANCE VALIDATION (PDR vs Cloud MCMC) ===")
        print(f"Self ED(PDR):       {ED_self_pdr:.6f}")
        print(f"Self ED(Cloud):     {ED_self_true:.6f}")
        print(f"ED(PDR → Cloud):    {ED_pdr_true:.6f}")
        print(f"ED(Cloud → PDR):    {ED_true_pdr:.6f}")
        print(f"Difference:         {abs(ED_pdr_true - ED_true_pdr):.6e}")
        print("=== END ===\n")

        ED_mat = compute_pairwise_ed_matrix(X_pdr_sub, X_true_sub, XNAMES)
        plot_matrix_heatmap(ED_mat, XNAMES,
            title=f"Energy distance (pairwise) PDR vs Cloud {run_note}",
            out_png=os.path.join(FINAL_METRICS_DIR, f"fig_EDmatrix_pdr11d{run_note}.png"),
            fmt="{:.2f}")

        JS_mat = compute_pairwise_js_matrix(
            X_pdr_sub, X_true_sub, XNAMES, PMIN_11, PMAX_11, n_grid=80, bw_2d=0.2)
        plot_matrix_heatmap(JS_mat, XNAMES,
            title=f"JS divergence (pairwise) PDR vs Cloud {run_note}",
            out_png=os.path.join(FINAL_METRICS_DIR, f"fig_JSmatrix_pdr11d{run_note}.png"),
            fmt="{:.3f}")

        summary_path = os.path.join(FINAL_METRICS_DIR, f'summary_pdr11d{run_note}.txt')
        with open(summary_path, 'w') as f:
            f.write("=== PDR vs Cloud MCMC Posterior Metrics ===\n")
            f.write(f"Case: {'WITH radiation (OLR/OSR)' if USE_RADIATION else 'NO radiation (PCP/LWP/IWP)'}\n\n")
            f.write(f"  Self ED(PDR):   {ED_self_pdr:.6f}\n")
            f.write(f"  Self ED(Cloud): {ED_self_true:.6f}\n")
            f.write(f"  ED(PDR,Cloud):  {ED_pdr_true:.6f}\n")
            f.write(f"  ED(Cloud,PDR):  {ED_true_pdr:.6f}\n")
            f.write(f"  |DeltaED|:      {abs(ED_pdr_true-ED_true_pdr):.6e}\n\n")
            f.write("Pairwise ED matrix:\n"); np.savetxt(f, ED_mat, fmt="%.4f")
            f.write("\nPairwise JS matrix:\n"); np.savetxt(f, JS_mat, fmt="%.6f")
        print("Saved summary:", summary_path)

    except Exception as e:
        print("[WARN] Skipping ED/JS metrics:", repr(e))

def main():
    print("\n=== CASE A (Fig.7): NO RADIATION ===")
    run_case(False)
    print("\n=== CASE B (Fig.9): WITH RADIATION ===")
    run_case(True)

if __name__ == "__main__":
    main()
