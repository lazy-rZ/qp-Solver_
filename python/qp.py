import numpy as np
from scipy.linalg import cho_factor, cho_solve

class QuadraticProgram:
    """
    ## Convex quadratic program:

        minimize   0.5 x^T Q x + c^T x
        subject to A x = b
                   G x <= h

    ## Inequalities are handled with slacks s >= 0:
        G x + s = h
    """

    def __init__(self, Q, c, A=None, b=None, G=None, h=None):
        Q = np.atleast_2d(np.array(Q, dtype=float))
        c = np.atleast_1d(np.array(c, dtype=float))
        n = c.size
        if Q.shape != (n, n):
            raise ValueError("Q must be (n, n) with n = len(c)")
        self.Q, self.c, self.n = Q, c, n

        # Equalities
        if A is not None:
            A = np.atleast_2d(np.array(A, dtype=float))
            b = np.atleast_1d(np.array(b, dtype=float))
            if A.shape[0] != b.size or A.shape[1] != n:
                raise ValueError("A must be (m_e, n) and b length m_e")
        self.A = A
        self.b = b if A is not None else None

        # Inequalities
        if G is not None:
            G = np.atleast_2d(np.array(G, dtype=float))
            h = np.atleast_1d(np.array(h, dtype=float))
            if G.shape[0] != h.size or G.shape[1] != n:
                raise ValueError("G must be (m_i, n) and h length m_i")
        self.G = G
        self.h = h if G is not None else None

        self.m_e = 0 if self.A is None else self.A.shape[0]
        self.m_i = 0 if self.G is None else self.G.shape[0]


class MehrotraIPMSolver:
    """
    ## Primal dual interior-point method with Mehrotra predictor corrector.
    """

    def __init__(
        self,
        max_iter=50,
        tol=1e-8,
        mu_tol=None,
        verbose=False,
        eta=0.99,
        regularization=1e-9,
    ):
        """
        Parameters
        ----------
        max_iter : int
            Maximum IPM iterations.
        tol : float
            Tolerance on KKT residuals (∞-norm).
        mu_tol : float or None
            Tolerance on complementarity; if None, uses tol.
        verbose : bool
            Print iteration log if True.
        eta : float
            Fraction-to-boundary parameter in (0, 1), e.g. 0.9–0.995.
        regularization : float
            Small diagonal regularization on KKT blocks.
        """
        self.max_iter = max_iter
        self.tol = float(tol)
        self.mu_tol = float(mu_tol) if mu_tol is not None else float(tol)
        self.verbose = verbose
        self.eta = float(eta)
        self.regularization = float(regularization)

    @staticmethod
    def _norm_inf(v):
        v = np.linalg.norm(np.asarray(v).ravel(), np.inf) if np.size(v) else 0.0
        return float(v)

    def _initial_point(self, qp: QuadraticProgram):
        """Infeasible-start, but interior w.r.t. inequalities."""
        n, m_e, m_i = qp.n, qp.m_e, qp.m_i
        Q, c, A, b, G, h = qp.Q, qp.c, qp.A, qp.b, qp.G, qp.h

        # x: LS solution of A x = b or zero
        if m_e > 0:
            x, *_ = np.linalg.lstsq(A, b, rcond=None)
        else:
            x = np.zeros(n)
        x = x.reshape(-1)

        # s, z for inequalities
        if m_i > 0:
            s = h - G @ x
            # Shift to make slacks strictly positive
            min_s = float(s.min())
            if min_s <= 0:
                s += (1.0 - min_s)
            z = np.ones_like(s)
        else:
            s = np.zeros(0)
            z = np.zeros(0)

        lam = np.zeros(m_e)
        return x, s, lam, z

    def _residuals(self, qp: QuadraticProgram, x, s, lam, z):
        """KKT residuals except complementarity."""
        Q, c, A, b, G, h = qp.Q, qp.c, qp.A, qp.b, qp.G, qp.h

        r_dual = Q @ x + c
        if A is not None:
            r_dual += A.T @ lam
        if G is not None and z.size:
            r_dual += G.T @ z

        r_pe = A @ x - b if A is not None else np.zeros(0)
        r_pi = G @ x + s - h if G is not None else np.zeros(0)

        return r_dual, r_pe, r_pi


    def _kkt_solve(
        self, qp: QuadraticProgram,
        x, s, lam, z,
        r_dual, r_pe, r_pi, r_cent
    ):
        """
        Solve reduced KKT system using Schur complement to eliminate dz and ds.
        
        System solved:
        [ Q + G.T (S^-1 Z) G    A.T ] [ dx   ]   [ rhs_x ]
        [ A                      0  ] [ dlam ] = [ rhs_e ]
        """
        Q, A, G = qp.Q, qp.A, qp.G
        n, m_e, m_i = qp.n, qp.m_e, qp.m_i

        # 1. Form the "Augmented Q" matrix (Schur complement of constraints)
        # Q_aug = Q + reg*I + G.T @ (S^-1 Z) @ G
        # This matrix handles the curvature (Q) and the inequality barriers.
        
        if m_i > 0:
            inv_s = 1.0 / (s + self.regularization)
            phi = z * inv_s  # Diagonal elements of Z * S^-1
            Q_aug = Q + self.regularization * np.eye(n) + G.T @ (phi[:, None] * G)
            
            # Modify RHS for the reduced system
            rhs_z_term = -r_cent + z * r_pi
            rhs_x = -r_dual - G.T @ (inv_s * rhs_z_term)
        else:
            # Fallback for pure equality case (no G)
            Q_aug = Q + self.regularization * np.eye(n)
            rhs_x = -r_dual
            inv_s = np.zeros(0)

        # 2. Solve for dx and dlam
        if m_e == 0:
            # Matrix is SPD, use Cholesky
            try:
                c, lower = cho_factor(Q_aug)
                dx = cho_solve((c, lower), rhs_x)
            except Exception:
                # Fallback if regularization wasn't enough (rare)
                dx = np.linalg.solve(Q_aug, rhs_x)
            dlam = np.zeros(0)
            
        else:
            # Matrix is Indefinite (Saddle Point), use LU (standard solve)
            # K_aug = [ Q_aug  A^T ]
            #         [ A      0   ]
            K_aug = np.block([
                [Q_aug, A.T],
                [A, np.zeros((m_e, m_e))]
            ])
            rhs_e = -r_pe
            rhs_aug = np.concatenate([rhs_x, rhs_e])
            
            sol = np.linalg.solve(K_aug, rhs_aug)
            dx = sol[:n]
            dlam = sol[n:]

        # 3. Recover dz and ds via back-substitution
        if m_i > 0:
            ds = -r_pi - G @ dx
            dz = inv_s * (-r_cent - z * ds)
        else:
            ds = np.zeros(0)
            dz = np.zeros(0)

        return dx, ds, dlam, dz

    def _fraction_to_boundary(self, s, ds, z, dz):
        """Largest step preserving positivity of s,z times eta."""
        alpha_pri = 1.0
        alpha_dual = 1.0

        if s.size:
            idx = ds < 0
            if np.any(idx):
                alpha_pri = min(alpha_pri,
                                self.eta * np.min(-s[idx] / ds[idx]))

        if z.size:
            idx = dz < 0
            if np.any(idx):
                alpha_dual = min(alpha_dual,
                                 self.eta * np.min(-z[idx] / dz[idx]))

        return float(alpha_pri), float(alpha_dual)

    def solve(self, qp: QuadraticProgram):
        """
        Solve the QP and return (x, s, lambda, z, info).

        info is a dict with iteration history + termination status.
        """
        x, s, lam, z = self._initial_point(qp)
        n, m_e, m_i = qp.n, qp.m_e, qp.m_i

        history = {
            "res_dual": [],
            "res_pri_eq": [],
            "res_pri_in": [],
            "mu": [],
        }

        for k in range(self.max_iter):
            r_dual, r_pe, r_pi = self._residuals(qp, x, s, lam, z)

            # Complementarity bits
            if m_i > 0:
                mu = float(s @ z / m_i)
                r_cent = s * z
            else:
                mu = 0.0
                r_cent = np.zeros(0)

            nd = self._norm_inf(r_dual)
            npe = self._norm_inf(r_pe)
            npi = self._norm_inf(r_pi)
            ncent = self._norm_inf(r_cent)

            history["res_dual"].append(nd)
            history["res_pri_eq"].append(npe)
            history["res_pri_in"].append(npi)
            history["mu"].append(mu)

            if self.verbose:
                print(
                    f"iter {k:2d}: "
                    f"||r_dual||={nd:.2e}, "
                    f"||r_pe||={npe:.2e}, "
                    f"||r_pi||={npi:.2e}, "
                    f"mu={mu:.2e}"
                )

            # Termination condition
            if max(nd, npe, npi) < self.tol and mu < self.mu_tol:
                status = "optimal"
                break

            if m_i == 0:
                # Equality-only QP: single Newton step
                dx, ds, dlam, dz = self._kkt_solve(
                    qp, x, s, lam, z, r_dual, r_pe, r_pi, r_cent
                )
                alpha_pri = alpha_dual = 1.0
            else:
                # Predictor (affine-scaling) step 
                r_cent_aff = r_cent # sigma = 0
                dx_aff, ds_aff, dlam_aff, dz_aff = self._kkt_solve(
                    qp, x, s, lam, z, r_dual, r_pe, r_pi, r_cent_aff
                )

                # Fraction-to-boundary for affine step
                alpha_aff_pri, alpha_aff_dual = self._fraction_to_boundary(
                    s, ds_aff, z, dz_aff
                )
                if alpha_aff_pri < 1e-10 or alpha_aff_dual < 1e-10:
                    status = "numerical_issue"
                    break

                # Affine complementarity
                mu_aff = ((s + alpha_aff_pri * ds_aff)
                          @ (z + alpha_aff_dual * dz_aff)) / max(1, m_i)

                # Mehrotra sigma
                sigma = (mu_aff / mu) ** 3 if mu > 0 else 0.0
                sigma = float(np.clip(sigma, 0.0, 1.0))

                # Corrector (with sigma and second order) 
                r_cent_corr = s * z + ds_aff * dz_aff - sigma * mu * np.ones(m_i)

                dx, ds, dlam, dz = self._kkt_solve(
                    qp, x, s, lam, z, r_dual, r_pe, r_pi, r_cent_corr
                )

                alpha_pri, alpha_dual = self._fraction_to_boundary(
                    s, ds, z, dz
                )

            alpha = min(alpha_pri, alpha_dual)
            if alpha <= 0:
                status = "numerical_issue"
                break

            # Update
            x = x + alpha * dx
            lam = lam + alpha * dlam
            if m_i > 0:
                s = s + alpha * ds
                z = z + alpha * dz

        else:
            status = "max_iter_exceeded"

        info = {
            "status": status,
            "iterations": len(history["mu"]),
            "history": history,
        }
        return x, s, lam, z, info
