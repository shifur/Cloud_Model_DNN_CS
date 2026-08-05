"""
appendix_a_figures.py
======================
Builds the THREE consolidated Appendix A convergence figures HCH requested,
each with two rows (top = MCMC-DNN, bottom = MCMC-CS/PDR):

  Figure A : walker acceptance-fraction histograms, one column per
             representative chain length.
  Figure B : running-mean stabilization for (ag, bg, as, bs), one column
             per parameter.
  Figure C : joint (ag, bg) 2D KDE stabilization, one column per
             representative chain length.

This reads the RAW chain NetCDFs already saved by the two convergence
scripts (mcmc_convergence_study_dnn.py / mcmc_convergence_study_pdr.py):

    DNN : runs/run_<RUN_ID>/output/_EXP_3_dnn11d<run_note>.nc   (Xa, p_xy)
    PDR : output/_EXP_3_pdr11d<run_note>.nc                     (Xa, p_xy)

It does NOT assume both chains were run to the same length, or that the
PDR/CS chain has data at every one of the 5 standard checkpoints
(20k/50k/80k/100k/400k). Each surrogate's own available checkpoints are
discovered independently from however many production steps that
surrogate's saved Xa actually contains; only the checkpoints that exist
for a given surrogate are plotted in its row. If the two surrogates don't
share the same checkpoint set, each row is labeled with its own actual
values rather than silently reusing the other surrogate's schedule.

Run this AFTER both mcmc_convergence_study_dnn.py and
mcmc_convergence_study_pdr.py have completed and written their output NetCDFs.
"""

import os
import glob
import numpy as np
import xarray as xr
import emcee
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# =============================================================================
# CONFIGURATION -- edit these three things for your run
# =============================================================================

# DNN output folder is timestamped per run (runs/run_<RUN_ID>/output/...).
# RUN_NOTE must be set explicitly -- "_rad" and "_norad" are DIFFERENT
# likelihood configurations (6-output vs 4-output) with genuinely different
# (ag,bg) posteriors, and a wildcard glob + "pick the most recently modified
# file" can silently select the wrong one if both have been run. This
# happened once already: "Figure 7" in the manuscript is the NO-RADIATION
# case (mcmc_with_dnn_updated.py labels it "CASE A (Fig.7): NO RADIATION"),
# but the glob previously picked up a "_rad" run instead because it was
# more recently modified.
RUN_NOTE = "_rad"   # set to "_rad" or "_norad" to match the figure you're rebuilding

DNN_NC_GLOB = f"runs/run_*/output/_EXP_3_dnn11d{RUN_NOTE}.nc"
PDR_NC_GLOB = f"output/_EXP_3_pdr11d{RUN_NOTE}.nc"

# --- SCCM (MCMC-true) reference chain -----------------------------------------
# Exact paths from mcmc_with_cloud_updated_close_look.py
# (DEFAULT_TRUE_CHAIN_FILE_NORAD / _RAD).  The SCCM posterior already exists;
# no MCMC is re-run.  Selected by RUN_NOTE so the reference always matches the
# surrogate rows' likelihood configuration.  This is the completed 400k chain,
# so it supports the full (ag,bg) checkpoint schedule below.
SCCM_NC_RAD   = "/home/ss24ce/last_time-2/FINAL_CRM11D_20260222_044235/output/mcmc_crm1d__EXP_3_full11_ag400.00000_bg0.40000_rad.nc"
SCCM_NC_NORAD = "/home/ss24ce/last_time-2/FINAL_CRM11D_20260216_140339/output/mcmc_crm1d__EXP_3_full11_ag400.00000_bg0.40000_norad.nc"
SCCM_NC = SCCM_NC_RAD if RUN_NOTE == "_rad" else SCCM_NC_NORAD

# Standard checkpoint schedule -- each row only uses whichever of these are
# <= that chain's actual number of saved production steps.
CHECKPOINT_SCHEDULE = [20000, 50000, 80000, 100000, 400000]

# Per-pair checkpoint overrides for the joint-distribution figures. (ag,bg)
# uses the requested schedule; (as,bs) keeps the standard one.
JOINT_PDF_CHECKPOINTS = {
    ("ag", "bg"): [80000, 100000, 200000, 350000, 400000],
    ("as", "bs"): [20000, 50000, 80000, 100000, 400000],
}

# The joint-distribution figures (Plots 3 & 4) are their own related pair --
# posterior geometry -- kept separate from the scalar convergence diagnostics
# (Plot 1 running mean, Plot 2 R-hat).
JOINT_PDF_PAIRS = [("as", "bs"), ("ag", "bg")]

RUNNING_MEAN_PARAMS = ("ag", "bg", "as", "bs")
RUNNING_MEAN_WINDOW = 200

OUT_DIR = "convergence_study_appendix_a"
os.makedirs(OUT_DIR, exist_ok=True)

XNAMES = ['as', 'bs', 'ag', 'bg', 'N0r', 'N0s', 'N0g', 'rhos', 'rhog', 'qc0', 'qi0']
PMIN_11 = np.array([50, 0.10, 50, 0.10, 0.05, 0.05, 0.05, 0.05, 0.05, 1e-4, 2e-6])
PMAX_11 = np.array([1000, 1.00, 1200, 0.90, 5.00, 2.50, 2.50, 1.00, 1.00, 2e-3, 1e-3])

# SCCM chain bounds (from mcmc_with_cloud_updated_close_look.py: PMIN_MCMC /
# PMAX_MCMC).  These differ from the surrogate box -- ag starts at 20 not 50,
# bg spans [0.0167, 0.983] -- so the reference row's KDE panels use their own
# limits and are not clipped to the surrogate box.
PMIN_SCCM = np.array([0.0, 0.0, 20.0, 0.01666667, 0.05, 0.05, 0.05,
                      0.01666667, 0.01666667, 1.e-4, 2.e-6])
PMAX_SCCM = np.array([1000.0, 1.0, 1180.0, 0.98333333, 5.0, 2.5, 2.5,
                      0.98333333, 0.98333333, 2.e-3, 1.e-3])

# Row display order (SCCM first -> top row), with per-row bounds and colours.
ROW_ORDER  = ["MCMC-true", "MCMC-DNN", "MCMC-CS"]
ROW_BOUNDS = {"MCMC-true": (PMIN_SCCM, PMAX_SCCM),
              "MCMC-DNN":  (PMIN_11, PMAX_11),
              "MCMC-CS":   (PMIN_11, PMAX_11)}
ROW_COLOR  = {"MCMC-true": "tab:green",
              "MCMC-DNN":  "tab:blue",
              "MCMC-CS":   "tab:orange"}

# Display label used for row y-labels and legends. The SCCM reference is
# always spelled out in full so no figure shows a bare "MCMC-true".
DISPLAY_LABEL = {"MCMC-true": "MCMC-true (SCCM reference)",
                 "MCMC-DNN":  "MCMC-DNN",
                 "MCMC-CS":   "MCMC-CS"}


# =============================================================================
# Loading helpers
# =============================================================================
def _latest(glob_pattern):
    matches = sorted(glob.glob(glob_pattern), key=os.path.getmtime)
    if not matches:
        raise FileNotFoundError(f"No file matched: {glob_pattern}")
    return matches[-1]


def load_chain(nc_path):
    """
    Load Xa in raw (nsteps, nwalkers, nparams) shape and x_true.

    Used for ALL rows including SCCM.  Unlike the flattening loader in
    mcmc_with_cloud_updated_close_look.py, this keeps the step and walker axes,
    which the running-mean, R-hat, and per-checkpoint KDE diagnostics require.
    """
    ds = xr.open_dataset(nc_path)
    Xa = ds["Xa"].values          # (nsteps, nwalkers, nparams)
    x_true = ds["x_true"].values if "x_true" in ds else None
    ds.close()
    if Xa.ndim != 3:
        raise RuntimeError(
            f"Expected 3D Xa (nsteps, nwalkers, nparams) from {nc_path}, got "
            f"shape {Xa.shape}. Do not pass a pre-flattened chain here.")
    return Xa, x_true


def available_checkpoints(nsteps, schedule=CHECKPOINT_SCHEDULE):
    """
    Only the checkpoints in `schedule` that this chain can actually support,
    plus the chain's own final length if it doesn't already match one of the
    schedule values. No checkpoint is assumed to exist just because it
    exists for the OTHER surrogate.
    """
    cps = np.asarray(schedule, dtype=int)
    cps = cps[(cps >= 2) & (cps <= nsteps)]
    if cps.size == 0 or cps[-1] != nsteps:
        cps = np.r_[cps, nsteps]
    return np.unique(cps).astype(int)


def acceptance_per_checkpoint(Xa, cp):
    """
    Per-walker acceptance fraction within the first `cp` production steps,
    derived directly from the chain (a step counts as accepted if the
    walker's position changed). This is necessary because emcee only
    exposes a single whole-chain acceptance_fraction, not a per-checkpoint
    one, so histograms "at several representative chain lengths" have to be
    reconstructed from the raw positions rather than read off a stored
    scalar.
    """
    sub = Xa[:cp]
    if sub.shape[0] < 2:
        return np.full(Xa.shape[1], np.nan)
    moved = np.any(np.diff(sub, axis=0) != 0.0, axis=-1)   # (cp-1, nwalkers)
    return moved.mean(axis=0)


def kde2d(x, y, xgrid, ygrid, bw=0.15):
    xy = np.vstack([x, y])
    try:
        kde = gaussian_kde(xy, bw_method=bw)
    except Exception:
        kde = gaussian_kde(xy)
    Xg, Yg = np.meshgrid(xgrid, ygrid)
    Z = kde(np.vstack([Xg.ravel(), Yg.ravel()])).reshape(Xg.shape)
    return Xg, Yg, Z


def grid_limits(samples, lo, hi, qlo=0.5, qhi=99.5, n=120):
    a = np.percentile(samples, qlo)
    b = np.percentile(samples, qhi)
    a = max(a, lo); b = min(b, hi)
    if not np.isfinite(a) or not np.isfinite(b) or a >= b:
        a, b = lo, hi
    return np.linspace(a, b, n)


def split_rhat_per_param(Xa_sub):
    """
    Standard split-R-hat: each walker's chain is split into a first and
    second half, and each half is treated as an independent chain (so a
    run with `nwalkers` walkers yields 2*nwalkers chains for the
    Gelman-Rubin calculation). Matches the "split-Rhat (each walker split
    early/late)" definition used in the original convergence scripts.
    """
    nsteps, nwalkers, nparams = Xa_sub.shape
    half = nsteps // 2
    rhat = np.full(nparams, np.nan)
    if half < 2:
        return rhat
    seg1 = Xa_sub[:half]
    seg2 = Xa_sub[half:2 * half]
    for k in range(nparams):
        chains = np.concatenate([seg1[:, :, k].T, seg2[:, :, k].T], axis=0)  # (2*nwalkers, half)
        n = chains.shape[1]
        chain_means = chains.mean(axis=1)
        chain_vars = chains.var(axis=1, ddof=1)
        W = chain_vars.mean()
        B = n * chain_means.var(ddof=1)
        var_hat = ((n - 1) / n) * W + B / n
        rhat[k] = np.sqrt(var_hat / W) if W > 0 else np.nan
    return rhat


def autocorr_and_ess_per_param(Xa_sub, nwalkers):
    """
    Integrated autocorrelation time and effective sample size per parameter,
    via emcee's own autocorr estimator (same tool used by the original
    convergence scripts' sampler). ESS = n_prod * n_walkers / (2*tau).
    """
    nsteps, _, nparams = Xa_sub.shape
    taus = np.full(nparams, np.nan)
    ess = np.full(nparams, np.nan)
    for k in range(nparams):
        chain_2d = Xa_sub[:, :, k]   # (nsteps, nwalkers)
        try:
            tau = emcee.autocorr.integrated_time(chain_2d, tol=0)
            taus[k] = float(np.mean(tau))
        except Exception:
            taus[k] = np.nan
        if np.isfinite(taus[k]) and taus[k] > 0:
            ess[k] = nsteps * nwalkers / (2 * taus[k])
    return taus, ess


# =============================================================================
# Figure A: acceptance-fraction histograms, 2 rows x N columns
# =============================================================================
def figure_A_acceptance(chains, out_png):
    """
    chains: dict like {"MCMC-DNN": (Xa, x_true), "MCMC-CS": (Xa, x_true)}
    Each row uses that surrogate's OWN available checkpoints -- if the two
    differ, columns are not forced to align 1:1; each row is labeled with
    its own actual chain-length values.
    """
    rows = list(chains.keys())
    per_row_cps = {}
    for name, (Xa, _) in chains.items():
        nsteps = Xa.shape[0]
        per_row_cps[name] = available_checkpoints(nsteps)

    ncols = max(len(v) for v in per_row_cps.values())
    fig, axes = plt.subplots(len(rows), ncols, figsize=(3.2 * ncols, 3.0 * len(rows)),
                             squeeze=False)

    for ri, name in enumerate(rows):
        Xa, _ = chains[name]
        cps = per_row_cps[name]
        for ci in range(ncols):
            ax = axes[ri, ci]
            if ci >= len(cps):
                ax.axis("off")
                continue
            cp = cps[ci]
            acc = acceptance_per_checkpoint(Xa, cp)
            ax.hist(acc, bins=20, density=True, color="tab:blue" if "DNN" in name else "tab:orange")
            ax.set_title(f"{cp:,} steps", fontsize=10)
            ax.grid(alpha=0.25)
            if ci == 0:
                ax.set_ylabel(f"{DISPLAY_LABEL.get(name, name)}\nDensity", fontsize=10)
            if ri == len(rows) - 1:
                ax.set_xlabel("Acceptance fraction")

    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", os.path.abspath(out_png))
    for name, cps in per_row_cps.items():
        print(f"  {name} checkpoints used: {list(cps)}")


# =============================================================================
# Figure B: running means, 2 rows x len(RUNNING_MEAN_PARAMS) columns
# =============================================================================
def figure_B_running_means(chains, out_png, params=RUNNING_MEAN_PARAMS,
                            window=RUNNING_MEAN_WINDOW):
    rows = list(chains.keys())
    use = [p for p in params if p in XNAMES]

    fig, axes = plt.subplots(len(rows), len(use), figsize=(3.6 * len(use), 3.0 * len(rows)),
                             sharex=False, squeeze=False)

    for ri, name in enumerate(rows):
        Xa, x_true = chains[name]
        nsteps = Xa.shape[0]
        for ci, pname in enumerate(use):
            k = XNAMES.index(pname)
            series = Xa[:, :, k].mean(axis=1)   # walker-averaged per step
            w = min(window, max(1, len(series)))
            kernel = np.ones(w) / w
            run = np.convolve(series, kernel, mode="valid")
            steps = np.arange(w - 1, len(series))

            ax = axes[ri, ci]
            ax.plot(np.arange(len(series)), series, color="0.75", lw=0.5)
            ax.plot(steps, run, color="tab:blue" if "DNN" in name else "tab:orange", lw=1.4)
            if x_true is not None:
                ax.axhline(x_true[k], ls="--", lw=1.0, color="k", alpha=0.7)
            ax.grid(alpha=0.25)
            if ci == 0:
                ax.set_ylabel(DISPLAY_LABEL.get(name, name), fontsize=10)
            if ri == 0:
                ax.set_title(pname, fontsize=11)
            if ri == len(rows) - 1:
                ax.set_xlabel("Production MCMC step")

    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", os.path.abspath(out_png))


# =============================================================================
# Figure C: joint (ag, bg) KDE stabilization, 2 rows x N columns
# =============================================================================
def figure_C_joint_pdf(chains, out_png, param_x="ag", param_y="bg",
                       bw=0.15, n_grid=160):
    """
    Joint posterior KDE for (param_x, param_y) as a function of chain length.
    Rows: MCMC-true (top), MCMC-DNN, MCMC-CS.  Uses the per-pair checkpoint
    schedule and each row's own parameter bounds (SCCM differs from surrogate).
    """
    x_i, y_i = XNAMES.index(param_x), XNAMES.index(param_y)
    rows = [r for r in ROW_ORDER if r in chains]
    schedule = JOINT_PDF_CHECKPOINTS.get((param_x, param_y), CHECKPOINT_SCHEDULE)
    per_row_cps = {name: available_checkpoints(chains[name][0].shape[0], schedule)
                   for name in rows}
    ncols = max(len(v) for v in per_row_cps.values())

    fig, axes = plt.subplots(len(rows), ncols, figsize=(3.4 * ncols, 3.2 * len(rows)),
                             squeeze=False)

    for ri, name in enumerate(rows):
        Xa, x_true = chains[name]
        pmin, pmax = ROW_BOUNDS[name]        # SCCM uses its own bounds
        cps = per_row_cps[name]
        x_full = Xa[:, :, x_i].ravel()
        y_full = Xa[:, :, y_i].ravel()
        x_grid = grid_limits(x_full, pmin[x_i], pmax[x_i], n=n_grid)
        y_grid = grid_limits(y_full, pmin[y_i], pmax[y_i], n=n_grid)

        for ci in range(ncols):
            ax = axes[ri, ci]
            if ci >= len(cps):
                ax.axis("off")
                continue
            cp = cps[ci]
            sub = Xa[:cp]
            xv = sub[:, :, x_i].ravel()
            yv = sub[:, :, y_i].ravel()
            Xg, Yg, Z = kde2d(xv, yv, x_grid, y_grid, bw=bw)
            ax.contourf(Xg, Yg, Z, levels=50, cmap="rainbow", alpha=0.9)
            ax.contour(Xg, Yg, Z, levels=10, linewidths=1.0, colors="crimson")
            if x_true is not None:
                ax.plot([x_true[x_i]], [x_true[y_i]], "kx", ms=7, mew=1.6)
            ax.set_title(f"{cp:,} steps", fontsize=10)
            if ci == 0:
                ax.set_ylabel(f"{DISPLAY_LABEL.get(name, name)}\n{param_y}", fontsize=10)
            if ri == len(rows) - 1:
                ax.set_xlabel(param_x)

    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", os.path.abspath(out_png))
    for name, cps in per_row_cps.items():
        print(f"  {name} checkpoints used: {list(cps)}")


# =============================================================================
# Figure B: running means, 2 rows x len(RUNNING_MEAN_PARAMS) columns
# =============================================================================
def figure_B_running_means(chains, out_png, params=RUNNING_MEAN_PARAMS,
                            window=RUNNING_MEAN_WINDOW):
    rows = list(chains.keys())
    use = [p for p in params if p in XNAMES]

    fig, axes = plt.subplots(len(rows), len(use), figsize=(3.6 * len(use), 3.0 * len(rows)),
                             sharex=False, squeeze=False)

    for ri, name in enumerate(rows):
        Xa, x_true = chains[name]
        nsteps = Xa.shape[0]
        for ci, pname in enumerate(use):
            k = XNAMES.index(pname)
            series = Xa[:, :, k].mean(axis=1)   # walker-averaged per step
            w = min(window, max(1, len(series)))
            kernel = np.ones(w) / w
            run = np.convolve(series, kernel, mode="valid")
            steps = np.arange(w - 1, len(series))

            ax = axes[ri, ci]
            ax.plot(np.arange(len(series)), series, color="0.75", lw=0.5)
            ax.plot(steps, run, color="tab:blue" if "DNN" in name else "tab:orange", lw=1.4)
            if x_true is not None:
                ax.axhline(x_true[k], ls="--", lw=1.0, color="k", alpha=0.7)
            ax.grid(alpha=0.25)
            if ci == 0:
                ax.set_ylabel(DISPLAY_LABEL.get(name, name), fontsize=10)
            if ri == 0:
                ax.set_title(pname, fontsize=11)
            if ri == len(rows) - 1:
                ax.set_xlabel("Production MCMC step")

    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", os.path.abspath(out_png))


# =============================================================================
# Figure C: joint (ag, bg) KDE stabilization, 2 rows x N columns
# =============================================================================
# =============================================================================
# Figure D: all-11-parameter R-hat / ESS small-multiples (addresses HCH's
# "convergence needs to be validated over all 11 parameters" concern with an
# actual visual, not just numbers buried in a CSV)
# =============================================================================
def figure_D_all_params_diagnostics(chains, out_png_rhat, out_png_ess):
    per_row_cps = {name: available_checkpoints(Xa.shape[0]) for name, (Xa, _) in chains.items()}

    # Compute split-Rhat and ESS at every available checkpoint, per parameter,
    # for each surrogate independently (no cross-assumption of checkpoints).
    rhat_data = {}   # name -> (checkpoints, [nparams] array per checkpoint)
    ess_data = {}
    for name, (Xa, _) in chains.items():
        nwalkers = Xa.shape[1]
        cps = per_row_cps[name]
        rhat_arr = np.full((len(cps), len(XNAMES)), np.nan)
        ess_arr = np.full((len(cps), len(XNAMES)), np.nan)
        for i, cp in enumerate(cps):
            sub = Xa[:cp]
            rhat_arr[i, :] = split_rhat_per_param(sub)
            _, ess_arr[i, :] = autocorr_and_ess_per_param(sub, nwalkers)
        rhat_data[name] = (cps, rhat_arr)
        ess_data[name] = (cps, ess_arr)

    def _small_multiples(data_dict, ylabel, out_png, hline=None):
        nparams = len(XNAMES)
        ncols = 4
        nrows = int(np.ceil(nparams / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(3.4 * ncols, 2.6 * nrows),
                                 squeeze=False)
        axes_flat = axes.ravel()
        for k, pname in enumerate(XNAMES):
            ax = axes_flat[k]
            for name, (cps, arr) in data_dict.items():
                ax.plot(cps, arr[:, k], marker="o", ms=4, lw=1.3,
                        color=ROW_COLOR.get(name, None), label=DISPLAY_LABEL.get(name, name))
            if hline is not None:
                for hv, hs in hline:
                    ax.axhline(hv, ls=hs, lw=0.8, color="k", alpha=0.6)
            ax.set_title(pname, fontsize=10)
            ax.grid(alpha=0.25)
            if k == 0:
                ax.legend(fontsize=7)
        for k in range(nparams, len(axes_flat)):
            axes_flat[k].set_visible(False)
        fig.supylabel(ylabel, fontsize=11)
        fig.supxlabel("Production MCMC steps used", fontsize=11)
        fig.tight_layout()
        fig.savefig(out_png, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print("Saved:", os.path.abspath(out_png))

    _small_multiples(rhat_data, "split $\\hat{R}$", out_png_rhat,
                      hline=[(1.01, "--"), (1.05, ":")])
    _small_multiples(ess_data, "ESS", out_png_ess)


# =============================================================================
# Figure E: trace-plot version of Figure B (same 2 rows x 4 columns layout),
# for direct side-by-side comparison against the running-mean figure so the
# redundant one can be dropped, per HCH's comment
# =============================================================================
def figure_E_traces(chains, out_png, params=RUNNING_MEAN_PARAMS, max_walkers_shown=12):
    rows = list(chains.keys())
    use = [p for p in params if p in XNAMES]

    fig, axes = plt.subplots(len(rows), len(use), figsize=(3.6 * len(use), 3.0 * len(rows)),
                             sharex=False, squeeze=False)

    for ri, name in enumerate(rows):
        Xa, x_true = chains[name]
        nwalkers = Xa.shape[1]
        show = min(max_walkers_shown, nwalkers)
        for ci, pname in enumerate(use):
            k = XNAMES.index(pname)
            ax = axes[ri, ci]
            for w in range(show):
                ax.plot(Xa[:, w, k], lw=0.4, alpha=0.5,
                        color="tab:blue" if "DNN" in name else "tab:orange")
            if x_true is not None:
                ax.axhline(x_true[k], ls="--", lw=1.0, color="k", alpha=0.8)
            ax.grid(alpha=0.2)
            if ci == 0:
                ax.set_ylabel(DISPLAY_LABEL.get(name, name), fontsize=10)
            if ri == 0:
                ax.set_title(pname, fontsize=11)
            if ri == len(rows) - 1:
                ax.set_xlabel("Production MCMC step")

    fig.suptitle(f"Raw walker traces (first {max_walkers_shown} walkers shown)", fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", os.path.abspath(out_png))
    print("  Compare against figB_running_means.png to decide which one to keep.")


# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":
    dnn_nc = _latest(DNN_NC_GLOB)
    pdr_nc = _latest(PDR_NC_GLOB)
    print(f"Requested configuration: RUN_NOTE = {RUN_NOTE!r}")
    print("SCCM chain file:", SCCM_NC)
    print("DNN  chain file:", dnn_nc)
    print("PDR  chain file:", pdr_nc)

    # Fail loudly rather than silently plotting the wrong likelihood
    # configuration if the resolved files don't match the requested RUN_NOTE.
    assert RUN_NOTE in dnn_nc, f"DNN file does not match RUN_NOTE={RUN_NOTE!r}: {dnn_nc}"
    assert RUN_NOTE in pdr_nc, f"PDR file does not match RUN_NOTE={RUN_NOTE!r}: {pdr_nc}"
    if not os.path.exists(SCCM_NC):
        raise FileNotFoundError(f"SCCM reference chain not found: {SCCM_NC}")

    Xa_sccm, xtrue_sccm = load_chain(SCCM_NC)
    Xa_dnn, xtrue_dnn = load_chain(dnn_nc)
    Xa_pdr, xtrue_pdr = load_chain(pdr_nc)
    print(f"SCCM chain length: {Xa_sccm.shape[0]:,} steps, {Xa_sccm.shape[1]} walkers")
    print(f"DNN  chain length: {Xa_dnn.shape[0]:,} steps, {Xa_dnn.shape[1]} walkers")
    print(f"PDR  chain length: {Xa_pdr.shape[0]:,} steps, {Xa_pdr.shape[1]} walkers")

    # SCCM first -> top row of every figure.
    chains = {
        "MCMC-true": (Xa_sccm, xtrue_sccm),
        "MCMC-DNN":  (Xa_dnn, xtrue_dnn),
        "MCMC-CS":   (Xa_pdr, xtrue_pdr),
    }

    # ---- The MCMC convergence appendix: exactly the plots on the whiteboard.
    # Plot 1: running mean  (SCCM / DNN / CS rows)
    figure_B_running_means(chains, os.path.join(OUT_DIR, "plot1_running_mean.png"))

    # Plot 2: R-hat  (SCCM / DNN / CS rows).  ESS is written alongside as
    # supplementary; drop it from the manuscript if only R-hat is wanted.
    figure_D_all_params_diagnostics(
        chains,
        out_png_rhat=os.path.join(OUT_DIR, "plot2_rhat_all_params.png"),
        out_png_ess=os.path.join(OUT_DIR, "supp_ess_all_params.png"),
    )

    # Plots 3 & 4: joint posterior distributions -- (as,bs) and (ag,bg).
    # Kept as their own separate figures (posterior geometry), distinct from
    # the scalar diagnostics above.
    for px, py in JOINT_PDF_PAIRS:
        figure_C_joint_pdf(
            chains, os.path.join(OUT_DIR, f"plot_joint_{px}_{py}.png"),
            param_x=px, param_y=py,
        )

    print(f"\nDone. Convergence-appendix figures are in: {os.path.abspath(OUT_DIR)}")
    print("  Plot 1  -> plot1_running_mean.png")
    print("  Plot 2  -> plot2_rhat_all_params.png")
    print("  Plot 3  -> plot_joint_as_bs.png")
    print("  Plot 4  -> plot_joint_ag_bg.png")
