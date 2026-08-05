

import numpy as np
from pdr_emulator_11d_gpu import PDR11D   # renamed module (was pdr_emulator_11d)

_emul = None

# Caches to avoid recomputing every call.
# Safe under multiprocessing.spawn (each worker has its own memory copy).
# Not thread-safe — do not use with threading concurrency.
_cache = {
    "y_mask_key": None,
    "sel_idx":    None,
    "log_norm":   None,
    "inv_var":    None,
}


def _fail_blob():
    return np.array([-np.inf] + [np.nan] * 6, dtype=float)


def _prep_mask_cache(y_mask, y_sig_six):
    y_mask_key = tuple(np.asarray(y_mask, dtype=int).tolist())
    if _cache["y_mask_key"] == y_mask_key:
        return

    sel     = (np.asarray(y_mask, dtype=int) == 1)
    sel_idx = np.where(sel)[0]
    sig     = np.asarray(y_sig_six, dtype=float)[sel_idx]
    var     = sig * sig
    inv_var = 1.0 / var
    d       = sel_idx.size
    log_norm = -0.5 * (d * np.log(2.0 * np.pi) + np.sum(np.log(var)))

    _cache["y_mask_key"] = y_mask_key
    _cache["sel_idx"]    = sel_idx
    _cache["log_norm"]   = float(log_norm)
    _cache["inv_var"]    = inv_var


def log_prob_pdr11d(P, xtrue, L1_six, y_sig_six,
                    pmin_full, pmax_full, PMask, y_mask, LType=1):
    """
    P : (nwalkers, 11) ensemble  (or a single (11,) vector).
    Returns a length-nwalkers list of (log_prob_i, blob_i), where
    blob_i = [log_like_i, y_pred_i(6)]  (shape (7,)).
    """
    global _emul

    if _emul is None:
        _emul = PDR11D('models_11d/pdr11d_model.npz')

    if LType == 1:
        _prep_mask_cache(y_mask, y_sig_six)

    P = np.atleast_2d(np.asarray(P, dtype=float))   # (nw, n_free)
    nw = P.shape[0]

    # Build full 11-D parameter vectors for every walker.
    pidx = np.nonzero(PMask)[0]
    X = np.tile(np.asarray(xtrue, dtype=float), (nw, 1))   # (nw, 11)
    X[:, pidx] = P

    # Per-walker bounds check (out-of-bounds -> -inf, NaN blob).
    in_b = np.all((X >= pmin_full) & (X <= pmax_full), axis=1)   # (nw,)

    log_like   = np.full(nw, -np.inf, dtype=float)
    y_pred_all = np.full((nw, 6), np.nan, dtype=float)

    if np.any(in_b):
        Xin = X[in_b]
        Yin = _emul.predict(Xin)              # (n_in, 6)  -- SINGLE batched call
        y_pred_all[in_b] = Yin

        if LType == 1:
            sel  = _cache["sel_idx"]
            mu   = np.asarray(L1_six, dtype=float)[sel]                 # (k,)
            diff = Yin[:, sel] - mu[None, :]                            # (n_in, k)
            ll   = _cache["log_norm"] - 0.5 * np.sum(
                       (diff * diff) * _cache["inv_var"][None, :], axis=1)
            log_like[in_b] = ll
        else:
            log_like[in_b] = 0.0

    # Per-walker (log_prob, blob) records for emcee's vectorize path.
    blobs = np.concatenate([log_like[:, None], y_pred_all], axis=1)    # (nw, 7)
    return list(zip(log_like.tolist(), blobs))

# log_prob_pdr11d.py
# ============================================================================
# VECTORISED over walkers for emcee vectorize=True.
#
# WHAT CHANGED vs the per-walker version:
#   - The first argument is now the FULL ensemble P of shape (nwalkers, 11)
#     (a single (11,) vector still works -> treated as one walker).
#   - PDR11D.predict() is called ONCE on the whole in-bounds batch instead of
#     once per walker, so the 11-dim Legendre loop + array allocations happen
#     a single time per MCMC step rather than nwalkers (=32) times.
#   - The Gaussian likelihood is evaluated vectorised over walkers.
#   - Returns a length-nwalkers list of (log_prob_i, blob_i) tuples, which is
#     exactly what emcee's vectorize path iterates over. blob_i is the same
#     (7,) = [log_like, y_pred(6)] record as before, so blobs_dtype and all
#     downstream get_blobs()["Hx"] slicing are UNCHANGED.
#
# The math per walker is identical to the old code (any difference is float
# round-off from batched vs per-row summation in predict()).
# ============================================================================


