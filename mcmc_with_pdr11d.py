# mcmc_with_pdr11d.py
import os, time
import numpy as np
import xarray as xr
import emcee
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from scipy.spatial.distance import cdist   # for memory-safe energy distance

from surrogate_config_11d import (
    set_global_seed, SEED,
    XNAMES, X_TRUE_11D, PMIN_11, PMAX_11,
    Y_SIG_SIX, YMASK_FIG7, YMASK_FIG9,
    TRUE_CHAIN_FILE_NORAD, TRUE_CHAIN_FILE_RAD,
    MCMC_NWALK, MCMC_NBURN, MCMC_NMCMC,
)

from crm_eval_11d_six import run_cloud_11d_six
import log_prob_pdr11d

# ------------------------------------------------------------------
#  Cloud MCMC true posterior paths are imported from surrogate_config_11d.py
# ------------------------------------------------------------------

# KDE figures (ag-bg marginal + 11x11 pairwise grid) go here, mirroring the
# DNN script's DNN_updated_figures_mcmc/ folder.
PDR_FIG_DIR = "PDR_updated_figures_mcmc"
os.makedirs(PDR_FIG_DIR, exist_ok=True)
print(f"[INFO] Saving KDE figures under: {os.path.abspath(PDR_FIG_DIR)}")

# Subsample cap for the KDE figures only (match the standalone Cloud script).
# The full flattened chain is nsteps*nwalkers (e.g. 400000*32 ~ 12.8M),
# far too large for gaussian_kde across 55 panels; cap to MAX_SAMPLES first.
# (ED/JS pools below are still drawn from the full chain.)
MAX_SAMPLES = 400000
SUBSAMPLE_SEED = 0

# simps fallback -- SciPy old/new compatible, always use x= keyword
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
#   KDE PLOTTING  (identical body to the Cloud and DNN scripts)
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

def _grid_limits(samples, lo, hi, qlo=0.5, qhi=99.5, n=160):
    a = max(lo, np.percentile(samples, qlo))
    b = min(hi, np.percentile(samples, qhi))
    if (not np.isfinite(a)) or (not np.isfinite(b)) or (a >= b):
        a, b = lo, hi
    return np.linspace(a, b, n)

def plot_ag_bg_marginal(Xa_plot, xnames, pmin, pmax, x_true, out_png, title_note=""):
    # Guard against silent column/name misalignment (e.g. if some params are fixed)
    assert Xa_plot.shape[1] == len(xnames), (
        f"Xa_plot has {Xa_plot.shape[1]} columns but len(xnames)={len(xnames)}; "
        "chain columns are not aligned with parameter names (some params fixed?)")

    ag_idx = xnames.index('ag')
    bg_idx = xnames.index('bg')
    ag = Xa_plot[:, ag_idx]
    bg = Xa_plot[:, bg_idx]

    ag_min = max(pmin[ag_idx], np.percentile(ag, 0.5))
    ag_max = min(pmax[ag_idx], np.percentile(ag, 99.5))
    bg_min = max(pmin[bg_idx], np.percentile(bg, 0.5))
    bg_max = min(pmax[bg_idx], np.percentile(bg, 99.5))
    ag_grid = np.linspace(ag_min, ag_max, 200)
    bg_grid = np.linspace(bg_min, bg_max, 200)

    Xg, Yg, Z = kde2d(ag, bg, ag_grid, bg_grid, bw=0.15)

    fig = plt.figure(figsize=(5.8, 5.2), facecolor='white')
    gs  = fig.add_gridspec(2, 2, width_ratios=[4, 1.2], height_ratios=[1.2, 4],
                           hspace=0.05, wspace=0.05)
    ax_histx = fig.add_subplot(gs[0, 0])
    ax_histy = fig.add_subplot(gs[1, 1])
    ax       = fig.add_subplot(gs[1, 0])

    ax.contour(Xg, Yg, Z, levels=10, linewidths=1.2, colors='crimson')
    ax.contourf(Xg, Yg, Z, levels=50, alpha=0.6, cmap='rainbow')
    ax.plot([x_true[ag_idx]], [x_true[bg_idx]], 'kx', ms=10, mew=2)
    ax.set_xlabel('ag'); ax.set_ylabel('bg'); ax.grid(alpha=0.25)

    ax_histx.hist(ag, bins=60, density=True, histtype='step', lw=1.2)
    ax_histy.hist(bg, bins=60, density=True, histtype='step', lw=1.2,
                  orientation='horizontal')
    ax_histx.tick_params(labelbottom=False)
    ax_histy.tick_params(labelleft=False)

    plt.suptitle(f'ag-bg marginal posterior {title_note}', y=0.98, fontsize=12)
    os.makedirs(os.path.dirname(out_png) or '.', exist_ok=True)
    plt.savefig(out_png, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print("Saved:", os.path.abspath(out_png))

def plot_pairwise_grid(Xa_plot, xnames, pmin, pmax, x_true,
                       subset=None, out_png='fig_pairgrid.png',
                       title="", bw_1d=0.15, bw_2d=0.15, n_grid=160,
                       figsize_per_cell=1.6, dpi=150):
    assert Xa_plot.shape[1] == len(xnames), (
        f"Xa_plot has {Xa_plot.shape[1]} columns but len(xnames)={len(xnames)}; "
        "chain columns are not aligned with parameter names (some params fixed?)")

    if subset is None:
        subset = xnames[:]
    idxs = [xnames.index(n) for n in subset]
    d = len(idxs)
    S = Xa_plot[:, idxs]

    g1, k1 = [], []
    for j, jj in enumerate(idxs):
        g = _grid_limits(S[:, j], pmin[jj], pmax[jj], n=n_grid)
        g1.append(g)
        k1.append(kde1d(S[:, j], g, bw=bw_1d))

    fig, axes = plt.subplots(
        d, d,
        figsize=(figsize_per_cell * d, figsize_per_cell * d),
        facecolor='white',
        squeeze=False
    )
    plt.subplots_adjust(hspace=0.05, wspace=0.05)

    for r in range(d):
        for c in range(d):
            ax = axes[r, c]
            if r == c:
                ax.plot(g1[r], k1[r], lw=1.0)
                ax.set_yticks([])
                if x_true is not None:
                    ax.axvline(x_true[idxs[r]], ls='--', lw=1.0, color='k', alpha=0.7)
                ax.set_xlabel(subset[r] if r == d - 1 else "")
                if r < d - 1:
                    ax.set_xticklabels([])
                ax.grid(alpha=0.12)

            elif r > c:
                x = S[:, c]
                y = S[:, r]
                gx = _grid_limits(x, pmin[idxs[c]], pmax[idxs[c]], n=n_grid)
                gy = _grid_limits(y, pmin[idxs[r]], pmax[idxs[r]], n=n_grid)
                Xg, Yg, Z = kde2d(x, y, gx, gy, bw=bw_2d)
                ax.contourf(Xg, Yg, Z, levels=40, cmap='rainbow', alpha=0.9)
                if x_true is not None:
                    ax.plot([x_true[idxs[c]]], [x_true[idxs[r]]], 'kx', ms=7, mew=1.8)

                if r != d - 1:
                    ax.set_xticklabels([])
                else:
                    ax.set_xlabel(subset[c])

                if c != 0:
                    ax.set_yticklabels([])
                else:
                    ax.set_ylabel(subset[r])

                ax.grid(alpha=0.08)

            else:
                ax.set_visible(False)

    if title:
        plt.suptitle(title, y=0.995, fontsize=12)
    os.makedirs(os.path.dirname(out_png) or '.', exist_ok=True)
    plt.savefig(out_png, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print("Saved:", os.path.abspath(out_png))


# ==============================================================
#   JS DIVERGENCE HELPERS
# ==============================================================
def kullback_leibler_divergence(P, Q):
    eps = 1e-12
    P2 = P + eps
    Q2 = Q + eps
    return np.sum(P2 * np.log(P2 / Q2))

def jensen_shannon_divergence(P, Q):
    M = 0.5 * (P + Q)
    return 0.5 * kullback_leibler_divergence(P, M) + 0.5 * kullback_leibler_divergence(Q, M)


# ==============================================================
#   ENERGY DISTANCE HELPERS
# ==============================================================
def energy_distance(X, Y, max_n=10000, seed=None):
    rng = np.random.default_rng(seed)
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    n, d = X.shape
    m, d2 = Y.shape
    assert d == d2

    if n > max_n:
        idx = rng.choice(n, size=max_n, replace=False)
        X = X[idx]; n = max_n
    if m > max_n:
        idx = rng.choice(m, size=max_n, replace=False)
        Y = Y[idx]; m = max_n

    # FIX: use cdist -> (n,m) distance matrices directly, instead of building the
    # (n,m,d) broadcast tensor (which was ~2.2 GB at n=m=5000, d=11 and could OOM).
    dist_xy = cdist(X, Y)   # euclidean == previous np.linalg.norm over last axis
    dist_xx = cdist(X, X)
    dist_yy = cdist(Y, Y)

    term_xy = (2.0 / (n * m)) * np.sum(dist_xy)
    term_xx = (1.0 / (n * n)) * np.sum(dist_xx)
    term_yy = (1.0 / (m * m)) * np.sum(dist_yy)

    D2 = term_xy - term_xx - term_yy
    D2 = max(D2, 0.0)
    return np.sqrt(D2), D2

def fixed_subsample(X, max_n=5000, seed=None):
    """Return one fixed subset so exact self-distance and ED symmetry are meaningful."""
    X = np.asarray(X, dtype=float)
    n = X.shape[0]
    if n <= max_n:
        return X.copy()
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=max_n, replace=False)
    return X[idx].copy()

def compute_pairwise_ed_matrix(X_pdr, X_true, xnames, max_n=2000, seed_base=None):
    """
    Pairwise 2D Energy Distance for every unique parameter pair.
      - lower triangle only, then mirrored (ED is symmetric)
      - deterministic per-pair seeds when seed_base is provided
    """
    X_pdr = np.asarray(X_pdr, dtype=float)
    X_true = np.asarray(X_true, dtype=float)

    d = len(xnames)
    ED_mat = np.zeros((d, d), dtype=float)

    for i in range(d):
        for j in range(i):
            Xp = X_pdr[:, [j, i]]
            Xt = X_true[:, [j, i]]

            seed_ij = None if seed_base is None else seed_base + i * d + j
            ED_ij, _ = energy_distance(Xp, Xt, max_n=max_n, seed=seed_ij)

            ED_mat[i, j] = ED_ij
            ED_mat[j, i] = ED_ij

    return ED_mat

def extract_pairwise_ranking(ED_mat, labels):
    """Convert a symmetric ED/JS matrix into a ranked list of unique pairs (desc)."""
    M = np.asarray(ED_mat, dtype=float)
    d = len(labels)
    pairs = []
    for i in range(d):
        for j in range(i):
            pairs.append((f"{labels[j]}-{labels[i]}", M[i, j]))
    pairs.sort(key=lambda x: x[1], reverse=True)
    return pairs

def plot_pairwise_ed_heatmap(ED_mat, labels, out_png, title,
                             cmap='Blues', fmt="{:.2f}"):
    M = np.asarray(ED_mat, dtype=float)
    d = len(labels)

    fig, ax = plt.subplots(figsize=(1.25 * d, 1.05 * d), facecolor='white')

    vmin = np.nanmin(M)
    vmax = np.nanmax(M)
    im = ax.imshow(M, origin='lower', cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')

    ax.set_xticks(range(d))
    ax.set_xticklabels(labels, rotation=90, fontsize=11)
    ax.set_yticks(range(d))
    ax.set_yticklabels(labels, fontsize=11)

    for i in range(d):
        for j in range(d):
            ax.text(j, i, fmt.format(M[i, j]),
                    ha='center', va='center', fontsize=8, color='black')

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Energy distance")

    ax.set_title(title, fontsize=16, fontweight='bold', pad=10)
    plt.tight_layout()

    os.makedirs(os.path.dirname(out_png) or '.', exist_ok=True)
    plt.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("Saved ED heatmap:", os.path.abspath(out_png))

def plot_pairwise_ed_ranking(ED_mat, labels, out_png,
                             title="Ranked pairwise energy distance",
                             top_k=None):
    pairs = extract_pairwise_ranking(ED_mat, labels)
    if top_k is not None:
        pairs = pairs[:top_k]

    pair_names = [p[0] for p in pairs][::-1]
    pair_vals = [p[1] for p in pairs][::-1]

    fig_h = max(6, 0.28 * len(pair_names))
    fig, ax = plt.subplots(figsize=(10, fig_h), facecolor='white')

    ax.barh(range(len(pair_names)), pair_vals)
    ax.set_yticks(range(len(pair_names)))
    ax.set_yticklabels(pair_names, fontsize=9)
    ax.set_xlabel("Energy distance")
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    ax.set_axisbelow(True)

    if len(pair_vals) > 0:
        offset = 0.01 * max(pair_vals) if max(pair_vals) > 0 else 0.01
        for i, v in enumerate(pair_vals):
            ax.text(v + offset, i, f"{v:.3f}", va='center', fontsize=8)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_png) or '.', exist_ok=True)
    plt.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("Saved ED ranking plot:", os.path.abspath(out_png))

def compute_pairwise_js_matrix(X_pdr, X_true, xnames, pmin, pmax,
                               n_grid=80, bw_2d=0.2):
    X_pdr = np.asarray(X_pdr, dtype=float)
    X_true = np.asarray(X_true, dtype=float)
    d = len(xnames)
    JS_mat = np.zeros((d, d), dtype=float)

    # JS is symmetric in its two distributions and the pair (i,j)/(j,i) describe
    # the same 2D marginal -> compute the lower triangle once and mirror it.
    for i in range(d):
        for j in range(i):
            Xp = X_pdr[:, [i, j]]
            Xt = X_true[:, [i, j]]

            x_all = np.concatenate([Xp[:, 0], Xt[:, 0]])
            y_all = np.concatenate([Xp[:, 1], Xt[:, 1]])

            x_grid = _grid_limits(x_all, pmin[i], pmax[i], n=n_grid)
            y_grid = _grid_limits(y_all, pmin[j], pmax[j], n=n_grid)

            _, _, P = kde2d(Xp[:, 0], Xp[:, 1], x_grid, y_grid, bw=bw_2d)
            _, _, Q = kde2d(Xt[:, 0], Xt[:, 1], x_grid, y_grid, bw=bw_2d)

            P_flat = P.ravel()
            Q_flat = Q.ravel()
            P_flat /= np.sum(P_flat)
            Q_flat /= np.sum(Q_flat)
            js = jensen_shannon_divergence(P_flat, Q_flat)
            JS_mat[i, j] = js
            JS_mat[j, i] = js

    return JS_mat


# ==============================================================
#   HEATMAP PLOTTING FOR ED / JS MATRICES
# ==============================================================
def plot_matrix_heatmap(M, labels, title, out_png, cmap='Blues', fmt="{:.2f}"):
    d = len(labels)
    fig, ax = plt.subplots(figsize=(1.2*d, 1.2*d), facecolor='white')
    vmin = np.nanmin(M)
    vmax = np.nanmax(M)
    im = ax.imshow(M, origin='lower', cmap=cmap, vmin=vmin, vmax=vmax)

    ax.set_xticks(range(d))
    ax.set_xticklabels(labels, rotation=90)
    ax.set_yticks(range(d))
    ax.set_yticklabels(labels)

    for i in range(d):
        for j in range(d):
            ax.text(j, i, fmt.format(M[i, j]),
                    ha='center', va='center', fontsize=6, color='black')

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.title(title)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_png) or '.', exist_ok=True)
    plt.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.close(fig)


# ==============================================================
#   TRUE POSTERIOR SAMPLING (FROM CLOUD MCMC CHAIN)
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
    M = pts.shape[0]
    ns = min(nsamples, M)
    rng = np.random.default_rng(SEED + 999)
    idx = rng.choice(M, size=ns, replace=False)
    return pts[idx, :]

def subsample_cloud(X, max_n=4000, seed=None):
    X = np.asarray(X, dtype=float)
    n = X.shape[0]
    if n <= max_n:
        return X
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=max_n, replace=False)
    return X[idx, :]


# ==============================================================
#   MAIN PDR MCMC + METRIC COMPUTATION
# ==============================================================
def run_case(USE_RADIATION: bool):
    set_global_seed(SEED)
    x_true = X_TRUE_11D.copy()
    PMask = np.ones_like(x_true)

    y_mask = YMASK_FIG9 if USE_RADIATION else YMASK_FIG7
    run_note = "_rad" if USE_RADIATION else "_norad"
    title_note = "(with radiation: OLR, OSR)" if USE_RADIATION else "(no radiation: PCP, LWP, IWP)"

    # truth for all 6 @120
    L1_six = run_cloud_11d_six(x_true[None, :])[0]

    # MCMC knobs from surrogate_config_11d.py
    nwalk = MCMC_NWALK
    nburn = MCMC_NBURN
    nmcmc = MCMC_NMCMC

    print(f"[INFO] MCMC settings from config: nwalk={nwalk}, nburn={nburn}, nmcmc={nmcmc}")
    x_sig = np.array([20.0, 0.05, 20.0, 0.05,
                      0.05, 0.05, 0.05,
                      0.05, 0.05,
                      1.e-4, 1.e-5])

    # init walkers inside bounds
    p0 = np.tile(x_true, nwalk).reshape((nwalk, 11)) + \
        np.random.normal(0.0, 1.0, (nwalk, 11)) * x_sig
    for j in range(11):
        lo, hi = PMIN_11[j], PMAX_11[j]
        bad = (p0[:, j] < lo) | (p0[:, j] > hi)
        while np.any(bad):
            p0[bad, j] = x_true[j] + np.random.normal(0.0, x_sig[j], bad.sum())
            bad = (p0[:, j] < lo) | (p0[:, j] > hi)

    # sanity (vectorized log_prob returns a length-nw list of (lp, blob))
    res0 = log_prob_pdr11d.log_prob_pdr11d(
        p0[:1], x_true, L1_six, Y_SIG_SIX, PMIN_11, PMAX_11, PMask, y_mask
    )
    lp0 = res0[0][0]
    if not np.isfinite(lp0):
        raise RuntimeError("Initial PDR log_prob is -inf; train PDR first.")

    dtype = [("Hx", float, (1 + 6,))]
    sampler = emcee.EnsembleSampler(
        nwalk, 11, log_prob_pdr11d.log_prob_pdr11d,
        blobs_dtype=dtype,
        vectorize=True,   # batched: predict() runs once per step, not per walker
        args=[x_true, L1_six, Y_SIG_SIX, PMIN_11, PMAX_11, PMask, y_mask, 1]
    )

    t0 = time.time()
    state = sampler.run_mcmc(p0, nburn, progress=True)
    sampler.reset()
    sampler.run_mcmc(state, nmcmc, progress=True)
    print(f"PDR MCMC {run_note} took {time.time() - t0:.1f}s")

    Xa = sampler.get_chain()
    log_pxy = sampler.get_log_prob()
    blobs = sampler.get_blobs()["Hx"]
    log_pyx = blobs[:, :, 0]
    HXa = blobs[:, :, 1:]

    os.makedirs('output', exist_ok=True)
    base = f'_EXP_3_pdr11d{run_note}.nc'
    ds = xr.Dataset(
        {
            "x_true": (["nx"], x_true),
            "PMask": (["nx"], PMask.astype(float)),
            "L1_six": (["ny"], L1_six),
            "L2_six": (["ny"], (Y_SIG_SIX ** 2)),  # sigma^2 saved
            "y_mask": (["ny"], y_mask.astype(float)),
            "Xa": (["nsamples", "nchains", "nx"], Xa),
            "HXa": (["nsamples", "nchains", "ny"], HXa),
            "p_yx": (["nsamples", "nchains"], log_pyx),
            "p_xy": (["nsamples", "nchains"], log_pxy),
        },
        coords={
            "nsamples": np.arange(Xa.shape[0]),
            "nchains": np.arange(Xa.shape[1]),
            "nx": np.arange(11),
            "ny": np.arange(6),
        },
        attrs={
            "xnames": ",".join(XNAMES),
            "ynames": "PCP@120,ACC@120,LWP@120,IWP@120,OLR@120,OSR@120",
            "Acceptance fraction": float(np.mean(sampler.acceptance_fraction)),
        },
    )
    out_nc = os.path.join('output', base)
    ds.to_netcdf(out_nc)
    print("Saved:", out_nc)

    # -----------------------------
    #   KDE FIGURES -> PDR_FIG_DIR (identical style to the DNN script)
    # -----------------------------
    Xa_plot = Xa.reshape(-1, Xa.shape[-1])   # full chain (ED/JS below use this)

    # Cap the sample count fed to the KDE figures (matches the Cloud pairgrid
    # script's MAX_SAMPLES). ED/JS pools further down still use the full Xa_plot.
    Xa_plot_kde = (subsample_cloud(Xa_plot, max_n=MAX_SAMPLES, seed=SUBSAMPLE_SEED)
                   if MAX_SAMPLES is not None else Xa_plot)
    print(f"[INFO] KDE figures use {Xa_plot_kde.shape[0]:,} of {Xa_plot.shape[0]:,} samples")

    plot_ag_bg_marginal(
        Xa_plot_kde, XNAMES, PMIN_11, PMAX_11, x_true,
        out_png=os.path.join(PDR_FIG_DIR, f"fig_agbg_marginal{base.replace('.nc', '.png')}"),
        title_note=f"(PDR 11D; {'OLR/OSR' if y_mask[-2:].sum() else 'PCP/LWP/IWP'})"
    )

    plot_pairwise_grid(
        Xa_plot_kde, XNAMES, PMIN_11, PMAX_11, x_true,
        out_png=os.path.join(PDR_FIG_DIR, f"fig_pairgrid{base.replace('.nc', '.png')}"),
        title=f"PDR 11D pairwise marginals {'(with rad)' if y_mask[-2:].sum() else '(no rad)'}"
    )

    # ---- DEEP-DIVE: (as, bs) and (ag, bg) close-up pairwise structure ----
    # Same setup, same KDE-capped samples (Xa_plot_kde), same plot_pairwise_grid
    # function -- just restricted to the snow (as, bs) and graupel (ag, bg)
    # fallspeed-diameter parameter groups so their internal correlation
    # structure (and any cross-coupling between the two groups) is easy to see
    # without the other 7 parameters diluting the panel.
    rad_tag = 'with rad' if y_mask[-2:].sum() else 'no rad'

    # (1) Combined 4x4 grid: as, bs, ag, bg together -> shows as-bs, ag-bg,
    #     AND the cross pairs (as-ag, as-bg, bs-ag, bs-bg) in one figure.
    plot_pairwise_grid(
        Xa_plot_kde, XNAMES, PMIN_11, PMAX_11, x_true,
        subset=['as', 'bs', 'ag', 'bg'],
        out_png=os.path.join(PDR_FIG_DIR, f"fig_pairgrid_snowgraupel4x4{base.replace('.nc','.png')}"),
        title=f"PDR 11D deep-dive: (as,bs) & (ag,bg) incl. cross pairs ({rad_tag})"
    )

    # (2) Standalone 2x2 grid: as, bs only (snow fallspeed-diameter pair)
    plot_pairwise_grid(
        Xa_plot_kde, XNAMES, PMIN_11, PMAX_11, x_true,
        subset=['as', 'bs'],
        out_png=os.path.join(PDR_FIG_DIR, f"fig_pairgrid_as_bs{base.replace('.nc','.png')}"),
        title=f"PDR 11D deep-dive: (as, bs) snow fallspeed-diameter ({rad_tag})"
    )

    # (3) Standalone 2x2 grid: ag, bg only (graupel fallspeed-diameter pair)
    plot_pairwise_grid(
        Xa_plot_kde, XNAMES, PMIN_11, PMAX_11, x_true,
        subset=['ag', 'bg'],
        out_png=os.path.join(PDR_FIG_DIR, f"fig_pairgrid_ag_bg{base.replace('.nc','.png')}"),
        title=f"PDR 11D deep-dive: (ag, bg) graupel fallspeed-diameter ({rad_tag})"
    )

    # ==========================================================
    #   JS + ENERGY DISTANCE vs CLOUD MCMC (TRUE)  -> mcmc__PDR/
    # ==========================================================
    true_chain_file = TRUE_CHAIN_FILE_RAD if USE_RADIATION else TRUE_CHAIN_FILE_NORAD
    if not os.path.exists(true_chain_file):
        print("[WARN] Cloud MCMC 'true' chain not found -> skipping ED/JS metrics:")
        print(f"       {true_chain_file}")
        return

    try:
        os.makedirs("mcmc__PDR", exist_ok=True)

        # ED validation -- deterministic fixed subsets
        X_pdr_pool = subsample_cloud(Xa_plot, max_n=10000, seed=SEED + 101)
        X_true_pool = sample_true_from_cloud_chain(true_chain_file, nsamples=10000)

        ED_MAX_N_SCALAR = 5000
        ED_MAX_N_PAIRWISE = 2000

        Xp = fixed_subsample(X_pdr_pool, max_n=ED_MAX_N_SCALAR, seed=SEED + 203)
        Xt = fixed_subsample(X_true_pool, max_n=ED_MAX_N_SCALAR, seed=SEED + 204)

        ED_self_pdr, _ = energy_distance(Xp, Xp, max_n=ED_MAX_N_SCALAR, seed=None)
        ED_self_true, _ = energy_distance(Xt, Xt, max_n=ED_MAX_N_SCALAR, seed=None)
        ED_pdr_true, _ = energy_distance(Xp, Xt, max_n=ED_MAX_N_SCALAR, seed=None)
        ED_true_pdr, _ = energy_distance(Xt, Xp, max_n=ED_MAX_N_SCALAR, seed=None)

        print("\n=== ENERGY DISTANCE VALIDATION TESTS (PDR vs Cloud MCMC) ===")
        print(f"Exact self-distance (PDR vs same PDR):       ED = {ED_self_pdr:.6e}")
        print(f"Exact self-distance (Cloud vs same Cloud):   ED = {ED_self_true:.6e}")
        print(f"ED(PDR, Cloud):                              ED = {ED_pdr_true:.6f}")
        print(f"ED(Cloud, PDR):                              ED = {ED_true_pdr:.6f}")
        print(f"Symmetry difference:                         {abs(ED_pdr_true - ED_true_pdr):.6e}")
        print("=== END VALIDATION TESTS ===\n")

        ED_mat = compute_pairwise_ed_matrix(
            X_pdr_pool, X_true_pool, XNAMES,
            max_n=ED_MAX_N_PAIRWISE, seed_base=SEED + 500
        )

        ed_heat_png = os.path.join("mcmc__PDR", f"fig_EDmatrix_pdr11d{run_note}.png")
        plot_pairwise_ed_heatmap(
            ED_mat, XNAMES, out_png=ed_heat_png,
            title=f"Energy distance (pairwise)\nPDR vs Cloud {run_note}", fmt="{:.2f}"
        )

        ed_rank_png = os.path.join("mcmc__PDR", f"fig_EDranking_pdr11d{run_note}.png")
        plot_pairwise_ed_ranking(
            ED_mat, XNAMES, out_png=ed_rank_png,
            title=f"Ranked pairwise energy distance\nPDR vs Cloud {run_note}", top_k=None
        )

        js_png = os.path.join("mcmc__PDR", f"fig_JSmatrix_pdr11d{run_note}.png")
        JS_mat = compute_pairwise_js_matrix(
            X_pdr_pool, X_true_pool, XNAMES, PMIN_11, PMAX_11,
            n_grid=80, bw_2d=0.2
        )
        plot_matrix_heatmap(
            JS_mat, XNAMES,
            title=f"JS divergence (pairwise) PDR vs Cloud {run_note}",
            out_png=js_png, fmt="{:.3f}"
        )

        summary_path = os.path.join('mcmc__PDR', f'summary_pdr11d{run_note}.txt')
        with open(summary_path, 'w') as f:
            f.write("=== PDR vs Cloud MCMC Posterior Metrics ===\n")
            f.write(f"Case: {'WITH radiation (OLR/OSR)' if USE_RADIATION else 'NO radiation (PCP/LWP/IWP)'}\n\n")
            f.write("Energy distance validation:\n")
            f.write(f"  Self ED(PDR):   {ED_self_pdr:.6f}\n")
            f.write(f"  Self ED(Cloud): {ED_self_true:.6f}\n")
            f.write(f"  ED(PDR,Cloud):  {ED_pdr_true:.6f}\n")
            f.write(f"  ED(Cloud,PDR):  {ED_true_pdr:.6f}\n")
            f.write(f"  |dED|:          {abs(ED_pdr_true - ED_true_pdr):.6e}\n\n")
            f.write("Pairwise ED matrix (rows/cols = XNAMES order):\n")
            np.savetxt(f, ED_mat, fmt="%.4f")

            f.write("\nRanked pairwise ED values (largest to smallest):\n")
            ranked_pairs = extract_pairwise_ranking(ED_mat, XNAMES)
            for pair_name, pair_val in ranked_pairs:
                f.write(f"{pair_name:15s}  {pair_val:.6f}\n")

            f.write("\nPairwise JS matrix:\n")
            np.savetxt(f, JS_mat, fmt="%.6f")
        print("Saved summary:", summary_path)

    except Exception as e:
        print("[WARN] Skipping JS/ED metrics (PDR vs Cloud) due to error:", repr(e))


def main():
    print("\n=== CASE A (Fig.7): NO RADIATION ===")
    run_case(False)

    print("\n=== CASE B (Fig.9): WITH RADIATION ===")
    run_case(True)


if __name__ == "__main__":
    main()
