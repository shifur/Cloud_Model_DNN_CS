# mcmc_with_dnn_updated.py
import os
import time
import numpy as np
import xarray as xr
import emcee

from surrogate_config_11d import (
    set_global_seed, SEED,
    XNAMES, X_TRUE_11D, PMIN_11, PMAX_11,
    Y_SIG_SIX, YMASK_FIG7, YMASK_FIG9,
    MCMC_NWALK, MCMC_NBURN, MCMC_NMCMC,
    TRUE_CHAIN_FILE_NORAD, TRUE_CHAIN_FILE_RAD,
)
from crm_eval_11d_six import run_cloud_11d_six
import log_prob_dnn11d
from mcmc_utils import (
    _safe_float, save_dataset_safely,
    energy_distance, compute_pairwise_ed_matrix, compute_pairwise_js_matrix,
    plot_matrix_heatmap, plot_ag_bg_marginal, plot_pairwise_grid,
    gelman_rubin_rhat, plot_traces,
    plot_mcmc_timing, save_timing_txt,
    sample_true_from_cloud_chain, subsample_cloud,
)


def run_case(USE_RADIATION: bool, dir_output: str, dir_plots: str, dir_mcmc: str):
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
                      0.05, 0.05, 0.05,
                      0.05, 0.05,
                      1.e-4, 1.e-5])

    p0 = np.tile(x_true, nwalk).reshape((nwalk, 11)) + \
         rng.standard_normal((nwalk, 11)) * x_sig
    for j in range(11):
        lo, hi = PMIN_11[j], PMAX_11[j]
        bad = (p0[:, j] < lo) | (p0[:, j] > hi)
        while np.any(bad):
            p0[bad, j] = x_true[j] + rng.standard_normal(bad.sum()) * x_sig[j]
            bad = (p0[:, j] < lo) | (p0[:, j] > hi)

    lp0, _ = log_prob_dnn11d.log_prob_dnn11d(
        p0[0], x_true, L1_six, Y_SIG_SIX,
        PMIN_11, PMAX_11, PMask, y_mask, 1
    )
    if not np.isfinite(lp0):
        raise RuntimeError("Initial DNN log_prob is -inf; train DNN first.")

    dtype = [("Hx", float, (1 + 6,))]
    sampler = emcee.EnsembleSampler(
        nwalk, 11, log_prob_dnn11d.log_prob_dnn11d,
        blobs_dtype=dtype,
        args=[x_true, L1_six, Y_SIG_SIX,
              PMIN_11, PMAX_11, PMask, y_mask, 1]
    )

    t0 = time.time()
    state = sampler.run_mcmc(p0, nburn, progress=True)
    sampler.reset()

    t_prod_start = time.time()
    sampler.run_mcmc(state, nmcmc, progress=True)
    t_prod_end = time.time()

    total_prod_time_s = t_prod_end - t_prod_start
    per_step_time_ms  = (total_prod_time_s / nmcmc) * 1e3

    print(f"DNN MCMC {run_note} took {time.time() - t0:.1f}s")
    print(f"\n=== MCMC TIMING ({run_note}) ===")
    print(f"  Production steps      : {nmcmc:,}")
    print(f"  Total production time : {total_prod_time_s:.2f} s  "
          f"({total_prod_time_s / 3600:.4f} h)")
    print(f"  Per-step time         : {per_step_time_ms:.4f} ms")
    print("================================\n")

    timing_dict = {
        'surrogate'        : 'DNN',
        'run_note'         : run_note,
        'nwalk'            : nwalk,
        'nmcmc'            : nmcmc,
        'total_time_s'     : total_prod_time_s,
        'per_step_time_ms' : per_step_time_ms,
    }
    plot_mcmc_timing(timing_dict, os.path.join(dir_mcmc, f"fig_timing_dnn11d{run_note}.png"))
    save_timing_txt(timing_dict,  os.path.join(dir_mcmc, f"timing_dnn11d{run_note}.txt"))

    Xa      = sampler.get_chain()
    log_pxy = sampler.get_log_prob()

    b = sampler.get_blobs()
    try:
        Hx_struct = b["Hx"]
    except Exception:
        Hx_struct = b

    Hx_struct = np.asarray(Hx_struct)
    if Hx_struct.dtype == object:
        Hx_struct = np.array(Hx_struct.tolist(), dtype=np.float64)

    log_pyx = np.asarray(Hx_struct[:, :, 0])
    HXa     = np.asarray(Hx_struct[:, :, 1:])

    acc = float(np.mean(sampler.acceptance_fraction))
    print(f"Mean acceptance (DNN{run_note}): {acc:.3f}")

    try:
        tau_arr = sampler.get_autocorr_time(quiet=True)
        print("Autocorr times (per parameter, DNN):", tau_arr)
    except Exception as e:
        print("Autocorr time computation failed (DNN):", repr(e))
        tau_arr = None

    try:
        rhat = gelman_rubin_rhat(Xa)
        print("Gelman-Rubin R-hat (per parameter, DNN):", rhat)
    except Exception as e:
        print("R-hat computation failed (DNN):", repr(e))
        rhat = None

    plot_traces(Xa, XNAMES,
                os.path.join(dir_plots, f"trace_dnn11d{run_note}.png"),
                title="Trace plots (DNN 11D, all walkers)")

    Xa_s      = _safe_float(Xa,      dtype=np.float32)
    HXa_s     = _safe_float(HXa,     dtype=np.float32)
    log_pxy_s = _safe_float(log_pxy, dtype=np.float32)
    log_pyx_s = _safe_float(log_pyx, dtype=np.float32)

    x_true_s = _safe_float(x_true,              dtype=np.float32)
    PMask_s  = _safe_float(PMask.astype(float),  dtype=np.float32)
    L1_six_s = _safe_float(L1_six,              dtype=np.float32)
    L2_six_s = _safe_float(Y_SIG_SIX ** 2,      dtype=np.float32)
    y_mask_s = _safe_float(y_mask.astype(float), dtype=np.float32)

    base   = f"_EXP_3_dnn11d{run_note}.nc"
    out_nc = os.path.join(dir_output, base)

    ds = xr.Dataset(
        {"x_true": (["nx"],                        x_true_s),
         "PMask":  (["nx"],                        PMask_s),
         "L1_six": (["ny"],                        L1_six_s),
         "L2_six": (["ny"],                        L2_six_s),
         "y_mask": (["ny"],                        y_mask_s),
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
        ds.tau.attrs["long_name"] = "Autocorrelation time per parameter (DNN 11D)"
    if rhat is not None:
        ds["rhat"] = (["nx"], _safe_float(rhat, dtype=np.float32))
        ds.rhat.attrs["long_name"] = "Gelman-Rubin R-hat per parameter (DNN 11D)"

    save_dataset_safely(ds, out_nc)

    Xa_plot = Xa_s.reshape(-1, Xa_s.shape[-1])

    plot_ag_bg_marginal(
        Xa_plot, XNAMES, PMIN_11, PMAX_11, x_true,
        out_png=os.path.join(dir_plots, f"fig_agbg_marginal{base.replace('.nc', '.png')}"),
        title_note=f"(DNN 11D; {'OLR/OSR' if y_mask[-2:].sum() else 'PCP/LWP/IWP'})"
    )
    plot_pairwise_grid(
        Xa_plot, XNAMES, PMIN_11, PMAX_11, x_true,
        out_png=os.path.join(dir_plots, f"fig_pairgrid{base.replace('.nc', '.png')}"),
        title=f"DNN 11D pairwise marginals {'(with rad)' if y_mask[-2:].sum() else '(no rad)'}"
    )

    try:
        X_dnn_sub  = subsample_cloud(Xa_plot, max_n=5000, seed=SEED + 301)
        X_true_sub = sample_true_from_cloud_chain(
            TRUE_CHAIN_FILE_RAD if USE_RADIATION else TRUE_CHAIN_FILE_NORAD,
            nsamples=5000, seed=SEED + 999)

        ED_self_dnn, _  = energy_distance(X_dnn_sub,  X_dnn_sub,  max_n=2000, seed=SEED + 302)
        ED_self_true, _ = energy_distance(X_true_sub, X_true_sub, max_n=2000, seed=SEED + 303)
        ED_dnn_true, _  = energy_distance(X_dnn_sub,  X_true_sub, max_n=2000, seed=SEED + 304)
        ED_true_dnn, _  = energy_distance(X_true_sub, X_dnn_sub,  max_n=2000, seed=SEED + 305)

        print("\n=== ENERGY DISTANCE VALIDATION TESTS (DNN vs Cloud MCMC) ===")
        print(f"Self-distance (DNN vs DNN):     ED = {ED_self_dnn:.6f}")
        print(f"Self-distance (Cloud vs Cloud): ED = {ED_self_true:.6f}")
        print(f"ED(DNN -> Cloud) = {ED_dnn_true:.6f}")
        print(f"ED(Cloud -> DNN) = {ED_true_dnn:.6f}")
        print(f"Difference       = {abs(ED_dnn_true - ED_true_dnn):.6e}")
        print("=== END VALIDATION TESTS ===\n")

        ED_mat = compute_pairwise_ed_matrix(X_dnn_sub, X_true_sub, XNAMES, max_n=2000)
        plot_matrix_heatmap(
            ED_mat, XNAMES,
            title=f"Energy distance (pairwise) DNN vs Cloud {run_note}",
            out_png=os.path.join(dir_mcmc, f"fig_EDmatrix_dnn11d{run_note}.png"),
            fmt="{:.2f}"
        )

        JS_mat = compute_pairwise_js_matrix(
            X_dnn_sub, X_true_sub, XNAMES, PMIN_11, PMAX_11, n_grid=80, bw_2d=0.2
        )
        plot_matrix_heatmap(
            JS_mat, XNAMES,
            title=f"JS divergence (pairwise) DNN vs Cloud {run_note}",
            out_png=os.path.join(dir_mcmc, f"fig_JSmatrix_dnn11d{run_note}.png"),
            fmt="{:.3f}"
        )

        summary_path = os.path.join(dir_mcmc, f"summary_dnn11d{run_note}.txt")
        with open(summary_path, 'w') as f:
            f.write("=== DNN vs Cloud MCMC Posterior Metrics ===\n")
            f.write(f"Case: {'WITH radiation (OLR/OSR)' if USE_RADIATION else 'NO radiation (PCP/LWP/IWP)'}\n\n")
            f.write("Energy distance validation:\n")
            f.write(f"  Self ED(DNN):   {ED_self_dnn:.6f}\n")
            f.write(f"  Self ED(Cloud): {ED_self_true:.6f}\n")
            f.write(f"  ED(DNN,Cloud):  {ED_dnn_true:.6f}\n")
            f.write(f"  ED(Cloud,DNN):  {ED_true_dnn:.6f}\n")
            f.write(f"  |DeltaED|:      {abs(ED_dnn_true - ED_true_dnn):.6e}\n\n")
            f.write("Pairwise ED matrix (rows/cols = XNAMES order):\n")
            np.savetxt(f, ED_mat, fmt="%.4f")
            f.write("\nPairwise JS matrix:\n")
            np.savetxt(f, JS_mat, fmt="%.6f")
        print("Saved summary:", summary_path)

    except Exception as e:
        print("[WARN] Skipping JS/ED metrics (DNN vs Cloud) due to error:", repr(e))


def main():
    run_id   = time.strftime("%Y%m%d_%H%M%S")
    out_root = os.path.join("runs", f"run_{run_id}")

    dir_output = os.path.join(out_root, "output")
    dir_plots  = os.path.join(out_root, "plots")
    dir_mcmc   = os.path.join(out_root, "mcmc__DNN")

    os.makedirs(dir_output, exist_ok=True)
    os.makedirs(dir_plots,  exist_ok=True)
    os.makedirs(dir_mcmc,   exist_ok=True)

    print(f"[INFO] Saving ALL outputs under: {os.path.abspath(out_root)}")

    print("\n=== CASE A (Fig.7): NO RADIATION ===")
    run_case(False, dir_output, dir_plots, dir_mcmc)
    print("\n=== CASE B (Fig.9): WITH RADIATION ===")
    run_case(True,  dir_output, dir_plots, dir_mcmc)


if __name__ == "__main__":
    main()
