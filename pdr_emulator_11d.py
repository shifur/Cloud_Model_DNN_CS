# pdr_emulator_11d.py
# CHANGES from previous version:
#   CHANGE 1: predict() simplified — removed the confusing (phi/root_m) @ C * root_m
#             which unnecessarily divided then multiplied by sqrt(m_train).
#             The correct prediction is simply phi @ C_six (no sqrt(m) factor).
#             Mathematical justification: training solves (A)C = b where
#             A = Psi/sqrt(m) and b = F/sqrt(m), so Psi @ C = F directly.
#             Prediction Psi(x_new) @ C therefore needs NO sqrt(m) correction.
#   Everything else is identical to the original file.

import numpy as np
from scipy.special import legendre


def _map_to_canonical_11d(X, pmin11, pmax11):
    return 2.0 * (X - pmin11) / (pmax11 - pmin11) - 1.0


def _design_row_legendre_11d(xi_row, Lambda):
    """
    Build one basis row Psi(xi_row): shape (N_basis,).
    Matches the column-filling loop in reference unified_trials.py:
        A[:, n] *= sqrt(2*deg+1) * Pk(xi[:, kdim])
    but applied to a single point instead of m points.
    """
    N = Lambda.shape[0]
    phi = np.ones(N, dtype=float)
    for n in range(N):
        v = 1.0
        for k in range(Lambda.shape[1]):
            deg = int(Lambda[n, k])
            Pk  = legendre(deg)
            v  *= np.sqrt(2.0 * deg + 1.0) * Pk(xi_row[k])
        phi[n] = v
    return phi   # (N_basis,)


class PDR11D:
    def __init__(self, npz_path='models_11d/pdr11d_model.npz'):
        D = np.load(npz_path, allow_pickle=True)
        self.Lambda  = D['Lambda']         # (N_basis, 11)
        self.C_six   = D['C_six']          # (N_basis, 6)
        self.pmin11  = D['pmin11']         # (11,)
        self.pmax11  = D['pmax11']         # (11,)
        self.m_train = int(D['m_train'])

    def predict(self, X11):
        """
        X11 : (m, 11) or (11,) physical-space inputs.
        Returns Y : (m, 6) predictions.

        CHANGE 1: was (phi/root_m) @ C * root_m — cancel out to phi @ C.
                  Now written directly as phi @ self.C_six.
                  Reference equivalent: y_eval = Psi_sg @ C_res  (no sqrt(m)).
        """
        X = np.atleast_2d(X11).astype(float)
        xi = _map_to_canonical_11d(X, self.pmin11, self.pmax11)
        m  = X.shape[0]
        Y  = np.zeros((m, 6), dtype=float)
        for i in range(m):
            phi    = _design_row_legendre_11d(xi[i], self.Lambda)  # (N_basis,)
            Y[i,:] = phi @ self.C_six                              # (6,)  ← CHANGE 1
        return Y
