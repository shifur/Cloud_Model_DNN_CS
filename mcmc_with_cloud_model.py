
# mcmc_with_cloud_updated_close_look.py
#
# Loads the ALREADY-COMPLETED Cloud CRM1D MCMC chains (the .nc files produced by
# run_mcmc_crm1d_full11.py) and generates the (as,bs)/(ag,bg) deep-dive pairwise
# figures directly from the saved "Xa" chain -- no CRM1D model run, no MCMC
# resampling, no log_prob_crm1d import needed.
#
# By default this uses the exact TRUE_CHAIN_FILE_NORAD / TRUE_CHAIN_FILE_RAD
# paths already hardcoded in mcmc_with_dnn_updated.py, so it can simply be run
# with no arguments:
#
#   python mcmc_with_cloud_updated_close_look.py
#
# To point at different chain files instead:
#
#   python mcmc_with_cloud_updated_close_look.py \
#       --norad /path/to/mcmc_crm1d__EXP_3_full11_ag400.00000_bg0.40000_norad.nc \
#       --rad   /path/to/mcmc_crm1d__EXP_3_full11_ag400.00000_bg0.40000_rad.nc \
#       --outdir ./CRM_deepdive_figures

import os
import argparse
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# ------------------------------------------------------------------
# Robust simps wrapper (SciPy old/new compatible) -- identical to the
# other scripts so KDE normalization matches exactly.
# ------------------------------------------------------------------
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


# ------------------------------------------------------------------
# Exact paths to the Cloud MCMC "true" posterior chains (11D full CRM),
# taken directly from mcmc_with_dnn_updated.py (TRUE_CHAIN_FILE_NORAD /
# TRUE_CHAIN_FILE_RAD). Used as the default --norad / --rad values below
# so the script can be run with no arguments.
# ------------------------------------------------------------------
DEFAULT_TRUE_CHAIN_FILE_NORAD = "/home/ss24ce/last_time-2/FINAL_CRM11D_20260216_140339/output/mcmc_crm1d__EXP_3_full11_ag400.00000_bg0.40000_norad.nc"
DEFAULT_TRUE_CHAIN_FILE_RAD   = "/home/ss24ce/last_time-2/FINAL_CRM11D_20260222_044235/output/mcmc_crm1d__EXP_3_full11_ag400.00000_bg0.40000_rad.nc"

# ------------------------------------------------------------------
# Bounds and parameter ordering used when these chains were generated
# (must match run_mcmc_crm1d_full11.py exactly, since the .nc files
# don't store pmin/pmax themselves -- only x_true/P1/P2/PMask).
# ------------------------------------------------------------------
XNAMES = ['as', 'bs', 'ag', 'bg', 'N0r', 'N0s', 'N0g', 'rhos', 'rhog', 'qc0', 'qi0']
PMIN_MCMC = np.array([0.0, 0.0, 20.0, 0.01666667, 0.05, 0.05, 0.05,
                      0.01666667, 0.01666667, 1.e-4, 2.e-6])
PMAX_MCMC = np.array([1000.0, 1.0, 1180.0, 0.98333333, 5.0, 2.5, 2.5,
                      0.98333333, 0.98333333, 2.e-3, 1.e-3])


# ==============================================================
#   KDE HELPERS (identical body to run_mcmc_crm1d_full11.py)
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
#   LOAD CHAIN FROM EXISTING .nc FILE
# ==============================================================
def load_chain(nc_path):
    if not os.path.exists(nc_path):
        raise FileNotFoundError(f"Chain file not found: {nc_path}")
    ds = xr.open_dataset(nc_path)

    if "Xa" not in ds:
        raise RuntimeError(f"'{nc_path}' has no 'Xa' variable -- not a valid chain file")
    Xa = ds["Xa"].values                      # (nsamples, nchains, nxp)
    x_true = ds["x_true"].values if "x_true" in ds else None

    # Prefer xnames stored in the file's attrs if present (keeps this script
    # correct even if a future run changes parameter order), else fall back
    # to the XNAMES module-level default above.
    if "xnames" in ds.attrs:
        xnames = ds.attrs["xnames"].split(",")
    else:
        xnames = XNAMES

    Xa_plot = Xa.reshape(-1, Xa.shape[-1])
    ds.close()
    return Xa_plot, x_true, xnames


def make_deepdive_plots(nc_path, run_note, outdir):
    print(f"\n[INFO] Loading existing chain ({run_note}): {nc_path}")
    Xa_plot, x_true, xnames = load_chain(nc_path)
    print(f"[INFO] Loaded {Xa_plot.shape[0]:,} samples x {Xa_plot.shape[1]} params "
          f"(xnames={xnames})")

    rad_tag = 'with rad' if run_note == 'rad' else 'no rad'
    base = f"_EXP_3_full11_{run_note}"

    # (1) Combined 4x4 grid: as, bs, ag, bg together -> as-bs, ag-bg,
    #     AND the four cross pairs (as-ag, as-bg, bs-ag, bs-bg)
    plot_pairwise_grid(
        Xa_plot, xnames, PMIN_MCMC, PMAX_MCMC, x_true,
        subset=['as', 'bs', 'ag', 'bg'],
        out_png=os.path.join(outdir, f"fig_pairgrid_snowgraupel4x4{base}.png"),
        title=f"CRM 11D deep-dive: (as,bs) & (ag,bg) incl. cross pairs ({rad_tag})"
    )

    # (2) Standalone 2x2: as, bs only (snow fallspeed-diameter pair)
    plot_pairwise_grid(
        Xa_plot, xnames, PMIN_MCMC, PMAX_MCMC, x_true,
        subset=['as', 'bs'],
        out_png=os.path.join(outdir, f"fig_pairgrid_as_bs{base}.png"),
        title=f"CRM 11D deep-dive: (as, bs) snow fallspeed-diameter ({rad_tag})"
    )

    # (3) Standalone 2x2: ag, bg only (graupel fallspeed-diameter pair)
    plot_pairwise_grid(
        Xa_plot, xnames, PMIN_MCMC, PMAX_MCMC, x_true,
        subset=['ag', 'bg'],
        out_png=os.path.join(outdir, f"fig_pairgrid_ag_bg{base}.png"),
        title=f"CRM 11D deep-dive: (ag, bg) graupel fallspeed-diameter ({rad_tag})"
    )


def main():
    ap = argparse.ArgumentParser(
        description="Generate (as,bs)/(ag,bg) deep-dive plots from existing Cloud MCMC .nc chains"
    )
    ap.add_argument("--norad", type=str, default=DEFAULT_TRUE_CHAIN_FILE_NORAD,
                    help="Path to the NO-RADIATION chain .nc file "
                         "(defaults to the exact TRUE_CHAIN_FILE_NORAD path "
                         "used in mcmc_with_dnn_updated.py)")
    ap.add_argument("--rad", type=str, default=DEFAULT_TRUE_CHAIN_FILE_RAD,
                    help="Path to the WITH-RADIATION chain .nc file "
                         "(defaults to the exact TRUE_CHAIN_FILE_RAD path "
                         "used in mcmc_with_dnn_updated.py)")
    ap.add_argument("--outdir", type=str, default="CRM_deepdive_figures",
                    help="Output directory for the new figures")
    args = ap.parse_args()

    if args.norad is None and args.rad is None:
        ap.error("Provide at least one of --norad or --rad (a path to an existing .nc chain).")

    os.makedirs(args.outdir, exist_ok=True)
    print(f"[INFO] Saving deep-dive figures under: {os.path.abspath(args.outdir)}")

    if args.norad is not None:
        make_deepdive_plots(args.norad, "norad", args.outdir)

    if args.rad is not None:
        make_deepdive_plots(args.rad, "rad", args.outdir)


if __name__ == "__main__":
    main()
