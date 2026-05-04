# mcmc_utils.py
# Shared utilities for MCMC pipelines (DNN and PDR).
# Import from here in mcmc_with_dnn_updated.py and mcmc_with_pdr11d.py.

import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from scipy.spatial.distance import cdist

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
        def simps(y, x=None, axis=-1):
            return np.trapz(y, x=x, axis=axis)


# ==============================================================
#   SAVE / NETCDF SAFETY HELPERS
# ==============================================================

def _ensure_writable_dir(path: str):
    d = os.path.dirname(path) or "."
    os.makedirs(d, exist_ok=True)
    if not os.access(d, os.W_OK):
        raise PermissionError(f"Directory not writable: {d}")


def _safe_float(arr, dtype=np.float32):
    """Cast to dtype and replace ±Inf with ±1e30. NaN is preserved (valid in NetCDF4)."""
    a = np.asarray(arr)
    if a.dtype == object:
        a = np.array(a.tolist(), dtype=np.float64)
    a = a.astype(dtype, copy=False)
    a = np.where(np.isposinf(a),  np.array( 1e30, dtype=dtype), a)
    a = np.where(np.isneginf(a), np.array(-1e30, dtype=dtype), a)
    return a


def save_dataset_safely(ds: xr.Dataset, out_nc: str, debug_one_by_one: bool = False):
    _ensure_writable_dir(out_nc)

    if os.path.exists(out_nc):
        os.remove(out_nc)

    if debug_one_by_one:
        print("\n[DEBUG] Writing variables one-by-one to locate failure...")
        for v in list(ds.data_vars):
            try:
                tmp = xr.Dataset({v: ds[v]}, coords=ds.coords, attrs=ds.attrs)
                tmp_path = out_nc.replace(".nc", f".__test_{v}.nc")
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
                tmp.to_netcdf(tmp_path, engine="netcdf4", format="NETCDF4")
                print("  OK:", v, "dtype=", tmp[v].dtype, "shape=", tmp[v].shape)
            except Exception as e:
                print("  FAIL:", v, "dtype=", ds[v].dtype, "shape=", ds[v].shape)
                print("       error:", repr(e))
                raise
        print("[DEBUG] All per-variable writes succeeded.\n")

    encoding = {}
    for v in ds.data_vars:
        enc = {"zlib": True, "complevel": 4}
        if ds[v].ndim == 3:
            enc["chunksizes"] = (min(2000, ds[v].shape[0]), ds[v].shape[1], ds[v].shape[2])
        elif ds[v].ndim == 2:
            enc["chunksizes"] = (min(4000, ds[v].shape[0]), ds[v].shape[1])
        encoding[v] = enc

    try:
        ds.to_netcdf(out_nc, engine="netcdf4", format="NETCDF4", encoding=encoding)
        print("Saved:", out_nc)
        return
    except Exception as e1:
        print("[WARN] netcdf4 write failed:", repr(e1))

    try:
        ds.to_netcdf(out_nc, engine="h5netcdf")
        print("Saved with h5netcdf:", out_nc)
        return
    except Exception as e2:
        print("[ERROR] h5netcdf fallback also failed:", repr(e2))
        raise


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
    if not np.isfinite(a) or not np.isfinite(b) or a >= b:
        a, b = lo, hi
    return np.linspace(a, b, n)


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
#   ENERGY DISTANCE
#   Uses scipy.spatial.distance.cdist to avoid materializing the
#   full O(n*m*d) intermediate tensor — safe for max_n up to ~5000.
# ==============================================================

def energy_distance(X, Y, max_n=2000, seed=None):
    rng = np.random.default_rng(seed)
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    n, d = X.shape
    m, d2 = Y.shape
    assert d == d2

    if n > max_n:
        X = X[rng.choice(n, size=max_n, replace=False)]
        n = max_n
    if m > max_n:
        Y = Y[rng.choice(m, size=max_n, replace=False)]
        m = max_n

    dist_xy = cdist(X, Y)
    dist_xx = cdist(X, X)
    dist_yy = cdist(Y, Y)

    term_xy = (2.0 / (n * m)) * dist_xy.sum()
    term_xx = (1.0 / (n * n)) * dist_xx.sum()
    term_yy = (1.0 / (m * m)) * dist_yy.sum()

    D2 = max(term_xy - term_xx - term_yy, 0.0)
    return np.sqrt(D2), D2


def compute_pairwise_ed_matrix(X_a, X_b, xnames, max_n=2000):
    X_a = np.asarray(X_a, dtype=float)
    X_b = np.asarray(X_b, dtype=float)
    d = len(xnames)
    ED_mat = np.zeros((d, d), dtype=float)
    for i in range(d):
        for j in range(i + 1, d):
            val, _ = energy_distance(X_a[:, [i, j]], X_b[:, [i, j]], max_n=max_n)
            ED_mat[i, j] = val
            ED_mat[j, i] = val
    return ED_mat


def compute_pairwise_js_matrix(X_a, X_b, xnames, pmin, pmax, n_grid=80, bw_2d=0.2):
    X_a = np.asarray(X_a, dtype=float)
    X_b = np.asarray(X_b, dtype=float)
    d = len(xnames)
    JS_mat = np.zeros((d, d), dtype=float)

    for i in range(d):
        for j in range(i + 1, d):
            Xp = X_a[:, [i, j]]
            Xt = X_b[:, [i, j]]

            x_all = np.concatenate([Xp[:, 0], Xt[:, 0]])
            y_all = np.concatenate([Xp[:, 1], Xt[:, 1]])

            x_grid = _grid_limits(x_all, pmin[i], pmax[i], n=n_grid)
            y_grid = _grid_limits(y_all, pmin[j], pmax[j], n=n_grid)

            _, _, P = kde2d(Xp[:, 0], Xp[:, 1], x_grid, y_grid, bw=bw_2d)
            _, _, Q = kde2d(Xt[:, 0], Xt[:, 1], x_grid, y_grid, bw=bw_2d)

            P_flat = P.flatten()
            Q_flat = Q.flatten()
            P_flat /= np.sum(P_flat)
            Q_flat /= np.sum(Q_flat)
            val = jensen_shannon_divergence(P_flat, Q_flat)
            JS_mat[i, j] = val
            JS_mat[j, i] = val

    return JS_mat


# ==============================================================
#   HEATMAP PLOTTING
# ==============================================================

def plot_matrix_heatmap(M, labels, title, out_png, cmap='Blues', fmt="{:.2f}"):
    d = len(labels)
    fig, ax = plt.subplots(figsize=(1.2 * d, 1.2 * d), facecolor='white')
    im = ax.imshow(M, origin='lower', cmap=cmap,
                   vmin=np.nanmin(M), vmax=np.nanmax(M))

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
#   PLOTTING: ag-bg marginal + pairwise grid
# ==============================================================

def plot_ag_bg_marginal(Xa_plot, xnames, pmin, pmax, x_true, out_png, title_note=""):
    ag_i, bg_i = xnames.index('ag'), xnames.index('bg')
    ag, bg = Xa_plot[:, ag_i], Xa_plot[:, bg_i]
    ag_g = _grid_limits(ag, pmin[ag_i], pmax[ag_i])
    bg_g = _grid_limits(bg, pmin[bg_i], pmax[bg_i])
    Xg, Yg, Z = kde2d(ag, bg, ag_g, bg_g, bw=0.15)

    fig = plt.figure(figsize=(5.8, 5.2), facecolor='white')
    gs = fig.add_gridspec(2, 2, width_ratios=[4, 1.2],
                          height_ratios=[1.2, 4],
                          hspace=0.05, wspace=0.05)
    axx = fig.add_subplot(gs[0, 0])
    axy = fig.add_subplot(gs[1, 1])
    ax  = fig.add_subplot(gs[1, 0])

    ax.contourf(Xg, Yg, Z, levels=60, alpha=0.85, cmap='rainbow')
    ax.plot([x_true[ag_i]], [x_true[bg_i]], 'kx', ms=9, mew=2)
    ax.set_xlabel('ag')
    ax.set_ylabel('bg')
    ax.grid(alpha=0.25)

    axx.hist(ag, bins=60, density=True, histtype='step', lw=1.0)
    axy.hist(bg, bins=60, density=True, histtype='step', lw=1.0, orientation='horizontal')
    axx.tick_params(labelbottom=False)
    axy.tick_params(labelleft=False)

    plt.suptitle(f'ag-bg marginal posterior {title_note}', y=0.98, fontsize=12)
    os.makedirs(os.path.dirname(out_png) or '.', exist_ok=True)
    plt.savefig(out_png, dpi=130, bbox_inches='tight')
    plt.close(fig)


def plot_pairwise_grid(Xa_plot, xnames, pmin, pmax, x_true,
                       subset=None, out_png='plots/fig_pairgrid.png',
                       title="", bw_1d=0.15, bw_2d=0.15, n_grid=160,
                       figsize_per_cell=1.6, dpi=150):
    if subset is None:
        subset = xnames[:]
    idxs = [xnames.index(n) for n in subset]
    d = len(idxs)
    S = Xa_plot[:, idxs]

    g1 = [_grid_limits(S[:, j], pmin[idxs[j]], pmax[idxs[j]], n=n_grid) for j in range(d)]
    k1 = [kde1d(S[:, j], g1[j], bw=bw_1d) for j in range(d)]

    fig, axes = plt.subplots(d, d,
                             figsize=(figsize_per_cell * d, figsize_per_cell * d),
                             facecolor='white', squeeze=False)
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


# ==============================================================
#   DIAGNOSTIC HELPERS
# ==============================================================

def gelman_rubin_rhat(chains_3d: np.ndarray):
    nsteps, nwalkers, nparams = chains_3d.shape
    rhat = np.empty(nparams, dtype=float)
    for k in range(nparams):
        x = chains_3d[:, :, k]
        n = float(nsteps)
        chain_means = np.mean(x, axis=0)
        chain_vars  = np.var(x, axis=0, ddof=1)
        B = n * np.var(chain_means, ddof=1)
        W = np.mean(chain_vars)
        Var_hat = ((n - 1.0) / n) * W + (B / n)
        rhat[k] = np.sqrt(Var_hat / W)
    return rhat


def plot_traces(Xa, param_names, out_png, title=""):
    nsteps, nwalkers, nparams = Xa.shape
    fig, axes = plt.subplots(nparams, 1, figsize=(10, 2.0 * nparams), sharex=True)
    if nparams == 1:
        axes = [axes]
    for k in range(nparams):
        axes[k].plot(Xa[:, :, k], alpha=0.3)
        axes[k].set_ylabel(param_names[k])
        axes[k].grid(alpha=0.2)
    axes[-1].set_xlabel("MCMC step")
    if title:
        fig.suptitle(title, y=0.99, fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    os.makedirs(os.path.dirname(out_png) or '.', exist_ok=True)
    plt.savefig(out_png, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("Saved trace plots:", os.path.abspath(out_png))


# ==============================================================
#   TIMING HELPERS
# ==============================================================

def plot_mcmc_timing(timing_dict: dict, out_png: str):
    label       = timing_dict['run_note'].replace('_', '').upper()
    total_s     = timing_dict['total_time_s']
    total_h     = total_s / 3600.0
    per_step_ms = timing_dict['per_step_time_ms']
    nwalk       = timing_dict['nwalk']
    nmcmc       = timing_dict['nmcmc']
    surrogate   = timing_dict.get('surrogate', 'Surrogate')

    fig, axes = plt.subplots(1, 2, figsize=(10, 5), facecolor='white')

    ax = axes[0]
    bar = ax.bar(['Total\n400K steps'], [total_s], color='steelblue',
                 width=0.4, edgecolor='black', linewidth=0.8)
    ax.bar_label(bar, labels=[f"{total_s:.1f} s\n({total_h:.2f} h)"],
                 padding=6, fontsize=11, fontweight='bold')
    ax.set_ylabel("Wall-clock time (seconds)", fontsize=12)
    ax.set_title(f"Total MCMC time\n({nmcmc:,} steps, {nwalk} walkers, {label})", fontsize=12)
    ax.set_ylim(0, total_s * 1.25)
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)

    ax2 = axes[1]
    bar2 = ax2.bar(['Per MCMC\nstep'], [per_step_ms], color='darkorange',
                   width=0.4, edgecolor='black', linewidth=0.8)
    ax2.bar_label(bar2, labels=[f"{per_step_ms:.3f} ms"],
                  padding=6, fontsize=11, fontweight='bold')
    ax2.set_ylabel("Wall-clock time (milliseconds)", fontsize=12)
    ax2.set_title(f"Per-step MCMC time\n(1 step = 1 ensemble proposal, {label})", fontsize=12)
    ax2.set_ylim(0, per_step_ms * 1.25)
    ax2.yaxis.grid(True, alpha=0.3)
    ax2.set_axisbelow(True)

    fig.suptitle(f"{surrogate} Surrogate MCMC — Computational Timing",
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_png) or '.', exist_ok=True)
    plt.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved timing plot: {os.path.abspath(out_png)}")


def save_timing_txt(timing_dict: dict, out_txt: str):
    surrogate = timing_dict.get('surrogate', 'Surrogate')
    os.makedirs(os.path.dirname(out_txt) or '.', exist_ok=True)
    with open(out_txt, 'w') as f:
        f.write(f"=== {surrogate} MCMC Timing Summary ===\n")
        f.write(f"Case                  : {timing_dict['run_note']}\n")
        f.write(f"Walkers               : {timing_dict['nwalk']}\n")
        f.write(f"Production steps      : {timing_dict['nmcmc']:,}\n")
        f.write(f"Total production time : {timing_dict['total_time_s']:.2f} s  "
                f"({timing_dict['total_time_s'] / 3600:.4f} h)\n")
        f.write(f"Per-step time         : {timing_dict['per_step_time_ms']:.4f} ms\n")
    print(f"Saved timing summary : {os.path.abspath(out_txt)}")


# ==============================================================
#   TRUE POSTERIOR SAMPLING
# ==============================================================

def sample_true_from_cloud_chain(nc_path, nsamples=5000, seed=None):
    from surrogate_config_11d import SEED as _SEED
    if not os.path.exists(nc_path):
        raise FileNotFoundError(f"Cloud MCMC file not found: {nc_path}")
    with xr.open_dataset(nc_path) as ds:
        if "Xa" not in ds:
            raise RuntimeError("Cloud MCMC file missing 'Xa' variable")
        Xa = ds["Xa"].values
    if Xa.ndim != 3:
        raise RuntimeError(f"Expected Xa with 3 dims, got shape {Xa.shape}")
    nsamp, nch, npar = Xa.shape
    pts = Xa.reshape(-1, npar)
    ns  = min(nsamples, pts.shape[0])
    rng = np.random.default_rng(seed if seed is not None else _SEED + 999)
    return pts[rng.choice(pts.shape[0], size=ns, replace=False), :]


def subsample_cloud(X, max_n=5000, seed=None):
    X = np.asarray(X, dtype=float)
    n = X.shape[0]
    if n <= max_n:
        return X
    rng = np.random.default_rng(seed)
    return X[rng.choice(n, size=max_n, replace=False), :]
