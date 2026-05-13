# pdr_emulator_11d.py
# CHANGES from previous version:
#   CHANGE 1: predict() simplified — phi @ C_six directly (no sqrt(m) factor).
#             Mathematical justification: training solves (A)C = b where
#             A = Psi/sqrt(m) and b = F/sqrt(m), so Psi @ C = F directly.
#
#   CHANGE 2: _design_row_legendre_11d() replaced by _design_row_vectorised().
#             Root cause of slowness:
#               Old: for n in range(4291) → for k in range(11) → legendre(deg)
#                    = 47,201 scipy object creations per predict() call → 6172 ms
#               Fix: legval(xk, leg_mat[k].T) evaluates all 4291 polynomials
#                    at once — only 11 Python iterations remain → 2.5 ms
#                    = ~2400x speedup. Same math, different code path.
#             Precomputed in __init__ (once at load):
#               self.norms   — sqrt(2*Lambda+1), shape (N_basis, 11)
#               self.leg_mat — unit-vector Legendre matrices, one per dimension

import numpy as np
from numpy.polynomial.legendre import legval


def _map_to_canonical_11d(X, pmin11, pmax11):
    return 2.0 * (X - pmin11) / (pmax11 - pmin11) - 1.0


class PDR11D:
    def __init__(self, npz_path='models_11d/pdr11d_model.npz'):
        D = np.load(npz_path, allow_pickle=True)
        self.Lambda  = D['Lambda']         # (N_basis, 11)
        self.C_six   = D['C_six']          # (N_basis, 6)
        self.pmin11  = D['pmin11']         # (11,)
        self.pmax11  = D['pmax11']         # (11,)
        self.m_train = int(D['m_train'])
        self.N_basis = self.Lambda.shape[0]
        self.d       = self.Lambda.shape[1]

        # ── Precompute norms (once at load, never repeated during MCMC) ──
        # norms[n,k] = sqrt(2*Lambda[n,k]+1)
        self.norms = np.sqrt(2.0 * self.Lambda + 1.0).astype(float)

        # ── Precompute Legendre matrices (once at load) ───────────────────
        # leg_mat[k] shape: (N_basis, max_deg_k+2)
        # leg_mat[k][n, deg] = 1.0  → legval(x, row) = P_deg(x)
        # legval(xk, leg_mat[k].T) evaluates all N_basis polynomials at xk
        # simultaneously — replaces the inner Python loop over N_basis.
        print(f"    [PDR11D] Precomputing Legendre matrices (d={self.d})...",
              end="", flush=True)
        self.leg_mat = []
        for k in range(self.d):
            degs_k  = self.Lambda[:, k].astype(int)
            max_deg = int(degs_k.max())
            mat = np.zeros((self.N_basis, max_deg + 2), dtype=float)
            for n in range(self.N_basis):
                mat[n, degs_k[n]] = 1.0
            self.leg_mat.append(mat)
        print("done")

    def _design_row_vectorised(self, xi_row):
        """
        Build basis row phi(xi_row): shape (N_basis,).

        phi[n] = prod_{k=1}^{11}  sqrt(2*Lambda[n,k]+1) * P_{Lambda[n,k]}(xi_k)

        Only 11 Python iterations (over d).
        Zero iterations over N_basis — numpy does all 4291 at once via legval.
        Mathematically identical to the old _design_row_legendre_11d().
        """
        phi = np.ones(self.N_basis, dtype=float)
        for k in range(self.d):
            xk   = float(xi_row[k])
            vals = legval(xk, self.leg_mat[k].T)   # (N_basis,) — all at once
            phi *= self.norms[:, k] * vals
        return phi

    def predict(self, X11):
        """
        X11 : (m, 11) or (11,) physical-space inputs.
        Returns Y : (m, 6) predictions.

        CHANGE 1: phi @ self.C_six  (no sqrt(m) factor needed).
        CHANGE 2: uses _design_row_vectorised (2.5 ms) not legendre loop (6172 ms).
        """
        X  = np.atleast_2d(X11).astype(float)
        xi = _map_to_canonical_11d(X, self.pmin11, self.pmax11)
        m  = X.shape[0]
        Y  = np.zeros((m, 6), dtype=float)
        for i in range(m):
            phi    = self._design_row_vectorised(xi[i])   # (N_basis,)
            Y[i,:] = phi @ self.C_six                     # (6,)
        return Y
