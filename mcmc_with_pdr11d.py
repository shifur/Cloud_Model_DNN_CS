# mcmc_with_pdr11d.py
import os
import time
import numpy as np
import xarray as xr
import emcee
from multiprocessing import get_context

from surrogate_config_11d import (
    set_global_seed, SEED,
    XNAMES, X_TRUE_11D, PMIN_11, PMAX_11,
    Y_SIG_SIX, YMASK_FIG7, YMASK_FIG9,
    MCMC_NWALK, MCMC_NBURN, MCMC_NMCMC,
    TRUE_CHAIN_FILE_NORAD, TRUE_CHAIN_FILE_RAD,
)
from crm_eval_11d_six import run_cloud_11d_six
import log_prob_pdr11d
from mcmc_utils import (
    _safe_float, save_dataset_safely,
    energy_distance, compute_pairwise_ed_matrix, compute_pairwise_js_matrix,
    plot_matrix_heatmap, plot_ag_bg_marginal, plot_pairwise_grid,
    gelman_rubin_rhat, plot_traces,
    plot_mcmc_timing, save_timing_txt,
    sample_true_from_cloud_chain, subsample_cloud,
)


def run_case(USE_RADIATION: bool,
             final_output_dir: str, final_plots_dir: str, final_metrics_dir: str):
    set_global_seed(SEED)
    rng = np.random.default_rng(SEED)

    x_true = X_TRUE_11D.copy()
    PMask  = np.ones_like(x_true)

    y_mask   = YMASK_FIG9 if USE_RADIATION else YMASK_FIG7
    run_note = "_rad" if USE_RADIATION else "_norad"

    L1_six = run_cloud_11d_six(x_true[None, :])[0]

    nwalk = MCMC_NWALK
    nburn = MCMC_NBURN
    nmcmc = MCMC_NMCMC

    x_sig = np.array([20.0, 0.05, 20.0, 0.05,
                      0.05, 0.05, 0.05, 0.05, 0.05,
                      1.e-4, 1.e-5])

    p0 = np.tile(x_true, nwalk).reshape((nwalk, 11)) + \
         rng.standard_normal((nwalk, 11)) * x_sig
    for j in range(11):
        lo, hi = PMIN_11[j], PMAX_11[j]
        bad = (p0[:, j] < lo) | (p0[:, j] > hi)
        while np.any(bad):
            p0[bad, j] = x_true[j] + rng.standard_normal(bad.sum()) * x_sig[j]
            bad = (p0[:, j] < lo) | (p0[:, j] > hi)

    lp0, _ = log_prob_pdr11d.log_prob_pdr11d(
        p0[0], x_true, L1_six, Y_SIG_SIX, PMIN_11, PMAX_11, PMask, y_mask
    )
    if not np.isfinite(lp0):
        raise RuntimeError("Initial PDR log_prob is -inf; train PDR first.")

    nprocs = int(os.environ.get("MCMC_NPROCS", "8"))
    print(f"[INFO] emcee parallel pool: nprocs={nprocs}")

    dtype = [("Hx", float, (1 + 6,))]
    ctx   = get_context("spawn")
    t0    = time.time()

    with ctx.Pool(processes=nprocs) as pool:
        sampler = emcee.EnsembleSampler(
            nwalk, 11, log_prob_pdr11d.log_prob_pdr11d,
            blobs_dtype=dtype,
            args=[x_true, L1_six, Y_SIG_SIX, PMIN_11, PMAX_11, PMask, y_mask, 1],
            pool=pool
        )
        state = sampler.run_mcmc(p0, nburn, progress=True)
        sampler.reset()

        t_prod_start = time.time()
        sampler.run_mcmc(state, nmcmc, progress=True)
        t_prod_end = time.time()

    total_prod_time_s = t_prod_end - t_prod_start
    per_step_time_ms  = (total_prod_time_s / nmcmc) * 1e3

    print(f"PDR MCMC {run_note} took {time.time() - t0:.1f}s")
    print(f"\n=== MCMC TIMING ({run_note}) ===")
    print(f"  Production steps      : {nmcmc:,}")
    print(f"  Total production time : {total_prod_time_s:.2f} s  "
          f"({total_prod_time_s / 3600:.4f} h)")
    print(f"  Per-step time         : {per_step_time_ms:.4f} ms")
    print("================================\n")

    timing_dict = {
        'surrogate'        : 'PDR',
        'run_note'         : run_note,
        'nwalk'            : nwalk,
        'nmcmc'            : nmcmc,
        'total_time_s'     : total_prod_time_s,
        'per_step_time_ms' : per_step_time_ms,
    }
    plot_mcmc_timing(timing_dict,
                     os.path.join(final_metrics_dir, f"fig_timing_pdr11d{run_note}.png"))
    save_timing_txt(timing_dict,
                    os.path.join(final_metrics_dir, f"timing_pdr11d{run_note}.txt"))

    Xa      = sampler.get_chain()
    log_pxy = sampler.get_log_prob()
    blobs   = sampler.get_blobs()
    try:
        blobs = blobs["Hx"]
    except Exception:
        pass
    log_pyx = blobs[:, :, 0]
    HXa     = blobs[:, :, 1:]

    acc = float(np.mean(sampler.acceptance_fraction))
    print(f"Mean acceptance (PDR{run_note}): {acc:.3f}")

    try:
        tau_arr = sampler.get_autocorr_time(quiet=True)
        print("Autocorr times (PDR):", tau_arr)
    except Exception as e:
        print("Autocorr time failed (PDR):", repr(e))
        tau_arr = None

    try:
        rhat = gelman_rubin_rhat(Xa)
        print("Gelman-Rubin R-hat (PDR):", rhat)
    except Exception as e:
        print("R-hat failed (PDR):", repr(e))
        rhat = None

    plot_traces(Xa, XNAMES,
                os.path.join(final_plots_dir, f"trace_pdr11d{run_note}.png"),
                title="Trace plots (PDR 11D, all walkers)")

    Xa_s      = _safe_float(Xa,      dtype=np.float32)
    HXa_s     = _safe_float(HXa,     dtype=np.float32)
    log_pxy_s = _safe_float(log_pxy, dtype=np.float32)
    log_pyx_s = _safe_float(log_pyx, dtype=np.float32)

    base   = f'_EXP_3_pdr11d{run_note}.nc'
    out_nc = os.path.join(final_output_dir, base)

    ds = xr.Dataset(
        {"x_true": (["nx"],                        _safe_float(x_true,              dtype=np.float32)),
         "PMask":  (["nx"],                        _safe_float(PMask.astype(float),  dtype=np.float32)),
         "L1_six": (["ny"],                        _safe_float(L1_six,              dtype=np.float32)),
         "L2_six": (["ny"],                        _safe_float(Y_SIG_SIX ** 2,      dtype=np.float32)),
         "y_mask": (["ny"],                        _safe_float(y_mask.astype(float), dtype=np.float32)),
         "Xa":     (["nsamples", "nchains", "nx"], Xa_s),
         "HXa":    (["nsamples", "nchains", "ny"], HXa_s),
         "p_yx":   (["nsamples", "nchains"],       log_pyx_s),
         "p_xy":   (["nsamples", "nchains"],       log_pxy_s)},
        coords={"nsamples": np.arange(Xa_s.shape[0]),
                "nchains":  np.arange(Xa_s.shape[1]),
                "nx":       np.arange(11),
                "ny":       np.arange(6)},
        attrs={"xnames":                 ",".join(XNAMES),
               "ynames":                 "PCP@120,ACC@120,LWP@120,IWP@120,OLR@120,OSR@120",
               "Acceptance fraction":    float(acc),
               "MCMC_total_prod_time_s": float(total_prod_time_s),
               "MCMC_per_step_time_ms":  float(per_step_time_ms),
               "MCMC_nmcmc_steps":       int(nmcmc)}
    )
    if tau_arr is not None:
        ds["tau"] = (["nx"], _safe_float(tau_arr, dtype=np.float32))
        ds.tau.attrs["long_name"] = "Autocorrelation time per parameter (PDR 11D)"
    if rhat is not None:
        ds["rhat"] = (["nx"], _safe_float(rhat, dtype=np.float32))
        ds.rhat.attrs["long_name"] = "Gelman-Rubin R-hat per parameter (PDR 11D)"

    save_dataset_safely(ds, out_nc)

    Xa_plot = Xa_s.reshape(-1, Xa_s.shape[-1])
    plot_ag_bg_marginal(
        Xa_plot, XNAMES, PMIN_11, PMAX_11, x_true,
        out_png=os.path.join(final_plots_dir, f'fig_agbg_marginal{base.replace(".nc", ".png")}'),
        title_note=f"(PDR 11D; {'OLR/OSR' if y_mask[-2:].sum() else 'PCP/LWP/IWP'})"
    )
    plot_pairwise_grid(
        Xa_plot, XNAMES, PMIN_11, PMAX_11, x_true,
        out_png=os.path.join(final_plots_dir, f'fig_pairgrid{base.replace(".nc", ".png")}'),
        title=f"PDR 11D pairwise marginals {'(with rad)' if y_mask[-2:].sum() else '(no rad)'}"
    )

    try:
        X_pdr_sub  = subsample_cloud(Xa_plot, max_n=4000, seed=SEED + 101)
        X_true_sub = sample_true_from_cloud_chain(
            TRUE_CHAIN_FILE_RAD if USE_RADIATION else TRUE_CHAIN_FILE_NORAD,
            nsamples=4000, seed=SEED + 999)

        ED_self_pdr, _  = energy_distance(X_pdr_sub,  X_pdr_sub,  max_n=1500, seed=SEED + 201)
        ED_self_true, _ = energy_distance(X_true_sub, X_true_sub, max_n=1500, seed=SEED + 202)
        ED_pdr_true, _  = energy_distance(X_pdr_sub,  X_true_sub, max_n=2000, seed=SEED + 203)
        ED_true_pdr, _  = energy_distance(X_true_sub, X_pdr_sub,  max_n=2000, seed=SEED + 204)

        print("\n=== ENERGY DISTANCE VALIDATION (PDR vs Cloud MCMC) ===")
        print(f"Self ED(PDR):    {ED_self_pdr:.6f}")
        print(f"Self ED(Cloud):  {ED_self_true:.6f}")
        print(f"ED(PDR -> Cloud) = {ED_pdr_true:.6f}")
        print(f"ED(Cloud -> PDR) = {ED_true_pdr:.6f}")
        print(f"Difference:      {abs(ED_pdr_true - ED_true_pdr):.6e}")
        print("=== END ===\n")

        ED_mat = compute_pairwise_ed_matrix(X_pdr_sub, X_true_sub, XNAMES, max_n=2000)
        plot_matrix_heatmap(ED_mat, XNAMES,
            title=f"Energy distance (pairwise) PDR vs Cloud {run_note}",
            out_png=os.path.join(final_metrics_dir, f"fig_EDmatrix_pdr11d{run_note}.png"),
            fmt="{:.2f}")

        JS_mat = compute_pairwise_js_matrix(
            X_pdr_sub, X_true_sub, XNAMES, PMIN_11, PMAX_11, n_grid=80, bw_2d=0.2)
        plot_matrix_heatmap(JS_mat, XNAMES,
            title=f"JS divergence (pairwise) PDR vs Cloud {run_note}",
            out_png=os.path.join(final_metrics_dir, f"fig_JSmatrix_pdr11d{run_note}.png"),
            fmt="{:.3f}")

        summary_path = os.path.join(final_metrics_dir, f'summary_pdr11d{run_note}.txt')
        with open(summary_path, 'w') as f:
            f.write("=== PDR vs Cloud MCMC Posterior Metrics ===\n")
            f.write(f"Case: {'WITH radiation (OLR/OSR)' if USE_RADIATION else 'NO radiation (PCP/LWP/IWP)'}\n\n")
            f.write(f"  Self ED(PDR):   {ED_self_pdr:.6f}\n")
            f.write(f"  Self ED(Cloud): {ED_self_true:.6f}\n")
            f.write(f"  ED(PDR,Cloud):  {ED_pdr_true:.6f}\n")
            f.write(f"  ED(Cloud,PDR):  {ED_true_pdr:.6f}\n")
            f.write(f"  |DeltaED|:      {abs(ED_pdr_true - ED_true_pdr):.6e}\n\n")
            f.write("Pairwise ED matrix:\n")
            np.savetxt(f, ED_mat, fmt="%.4f")
            f.write("\nPairwise JS matrix:\n")
            np.savetxt(f, JS_mat, fmt="%.6f")
        print("Saved summary:", summary_path)

    except Exception as e:
        print("[WARN] Skipping ED/JS metrics:", repr(e))


def main():
    run_tag           = time.strftime("%Y%m%d_%H%M%S")
    final_dir         = os.path.abspath(f"FINAL_PDR11D_{run_tag}")
    final_output_dir  = os.path.join(final_dir, "output")
    final_plots_dir   = os.path.join(final_dir, "plots")
    final_metrics_dir = os.path.join(final_dir, "mcmc__PDR")

    for _d in [final_dir, final_output_dir, final_plots_dir, final_metrics_dir]:
        os.makedirs(_d, exist_ok=True)
    print(f"[INFO] Final outputs will be saved under: {final_dir}")

    print("\n=== CASE A (Fig.7): NO RADIATION ===")
    run_case(False, final_output_dir, final_plots_dir, final_metrics_dir)
    print("\n=== CASE B (Fig.9): WITH RADIATION ===")
    run_case(True,  final_output_dir, final_plots_dir, final_metrics_dir)


if __name__ == "__main__":
    main()
