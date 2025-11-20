import numpy as np


class QuadraticProgram:
    """
    Convex quadratic program in the form:

        minimize   0.5 * x^T Q x + c^T x
        subject to A x = b
                   G x <= h

    Internally, we handle inequalities with slacks s >= 0:
        G x + s = h
    """

    def __init__(self, Q, c, A=None, b=None, G=None, h=None):
        Q = np.atleast_2d(np.array(Q, dtype=float))
        c = np.atleast_1d(np.array(c, dtype=float))

        n = c.size
        if Q.shape != (n, n):
            raise ValueError("Q must have shape (n, n) with n = len(c)")

        self.Q = Q
        self.c = c
        self.n = n

        # Equalities: A x = b
        if A is not None:
            A = np.atleast_2d(np.array(A, dtype=float))
            b = np.atleast_1d(np.array(b, dtype=float))
            if A.shape[0] != b.size or A.shape[1] != n:
                raise ValueError("A must be (m_e, n) and b must have length m_e")
            self.A = A
            self.b = b
        else:
            self.A = None
            self.b = None

        # Inequalities: G x <= h
        if G is not None:
            G = np.atleast_2d(np.array(G, dtype=float))
            h = np.atleast_1d(np.array(h, dtype=float))
            if G.shape[0] != h.size or G.shape[1] != n:
                raise ValueError("G must be (m_i, n) and h must have length m_i")
            self.G = G
            self.h = h
        else:
            self.G = None
            self.h = None

        self.m_e = 0 if self.A is None else self.A.shape[0]
        self.m_i = 0 if self.G is None else self.G.shape[0]


class PrimalDualInteriorPointSolver:
    """
    Primal-dual interior-point method for convex QPs.

    Solves:

        minimize   0.5 * x^T Q x + c^T x
        subject to A x = b
                   G x <= h

    using a (short-step style) primal-dual interior-point method with
    barrier parameter mu and centering parameter sigma.
    """

    def __init__(self, max_iter=50, tol=1e-8, sigma=0.5, verbose=True):
        """
        Parameters
        ----------
        max_iter : int
            Maximum number of Newton (IPM) iterations.
        tol : float
            Tolerance on residuals and complementarity.
        sigma : float in (0, 1)
            Centering parameter for the perturbed complementarity:
                S z ≈ sigma * mu * 1
        verbose : bool
            If True, print iteration diagnostics.
        """
        self.max_iter = max_iter
        self.tol = tol
        self.sigma = sigma
        self.verbose = verbose

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _norm_inf(v):
        v = np.asarray(v)
        return 0.0 if v.size == 0 else np.linalg.norm(v, np.inf)

    def _initial_point(self, qp: QuadraticProgram):
        """
        Heuristic strictly-feasible-ish starting point (x, s, lambda, z).

        - x: least-squares solution for A x = b (if equalities)
        - s: h - G x shifted to be strictly positive
        - z: positive duals for inequalities
        - lambda: zeros for equalities
        """
        n, m_e, m_i = qp.n, qp.m_e, qp.m_i

        # Start from x = 0 or least-squares equality solution
        if m_e > 0:
            # Minimize ||A x - b|| in least squares sense
            x, *_ = np.linalg.lstsq(qp.A, qp.b, rcond=None)
        else:
            x = np.zeros(n)

        # Inequality slacks and duals
        if m_i > 0:
            G, h = qp.G, qp.h
            s = h - G @ x
            # Make sure s > 0
            min_s = np.min(s)
            if min_s <= 0:
                s += (1.0 - min_s)
            z = np.ones_like(s)
        else:
            s = np.zeros(0)
            z = np.zeros(0)

        # Equality duals
        lam = np.zeros(m_e)

        return x, s, lam, z

    def _residuals_primal_dual(self, qp: QuadraticProgram, x, s, lam, z):
        """
        Unperturbed KKT residuals:

            r_dual = Q x + c + A^T lambda + G^T z
            r_pe   = A x - b
            r_pi   = G x + s - h
        """
        Q, c, A, b, G, h = qp.Q, qp.c, qp.A, qp.b, qp.G, qp.h

        # Dual residual
        r_dual = Q @ x + c
        if A is not None:
            r_dual += A.T @ lam
        if G is not None:
            r_dual += G.T @ z

        # Primal equality residual
        if A is not None:
            r_pe = A @ x - b
        else:
            r_pe = np.zeros(0)

        # Primal inequality residual
        if G is not None:
            r_pi = G @ x + s - h
        else:
            r_pi = np.zeros(0)

        return r_dual, r_pe, r_pi

    def _newton_step(self, qp: QuadraticProgram, x, s, lam, z, sigma, mu):
        """
        Compute the Newton step (dx, ds, dlam, dz) for the perturbed KKT system:

            r_dual(x,lambda,z) = 0
            r_pe(x)            = 0
            r_pi(x,s)          = 0
            S z - sigma * mu * 1 = 0

        using the standard primal–dual interior-point formulation.

        We eliminate ds using r_pi, then solve a reduced KKT system
        for (dx, dlam, dz), and finally recover ds.
        """
        Q, c, A, b, G, h = qp.Q, qp.c, qp.A, qp.b, qp.G, qp.h
        n, m_e, m_i = qp.n, qp.m_e, qp.m_i

        r_dual, r_pe, r_pi = self._residuals_primal_dual(qp, x, s, lam, z)

        if m_i > 0:
            e = np.ones(m_i)
            # Perturbed complementarity residual: S z - sigma * mu * 1
            r_cent = s * z - sigma * mu * e
            S = np.diag(s)
            Z = np.diag(z)
        else:
            r_cent = np.zeros(0)
            S = np.zeros((0, 0))
            Z = np.zeros((0, 0))

        # We solve for (dx, dlam, dz) in the reduced KKT system:
        #
        #   Q dx + A^T dlam + G^T dz = -r_dual
        #   A dx                    = -r_pe
        #  -Z G dx           + S dz = -r_cent + Z r_pi
        #
        # then recover ds from:
        #
        #   G dx + ds = -r_pi  => ds = -r_pi - G dx

        dim = n + m_e + m_i
        K = np.zeros((dim, dim))
        rhs = np.zeros(dim)

        # Row/col layout: [x (n), lambda (m_e), z (m_i)]

        # Q block
        K[0:n, 0:n] = Q

        # A blocks
        if m_e > 0:
            K[0:n, n:n + m_e] = A.T
            K[n:n + m_e, 0:n] = A

        # G & (S,Z) blocks (inequalities)
        if m_i > 0:
            # Top-right block: G^T for z
            K[0:n, n + m_e:n + m_e + m_i] = G.T
            # Bottom-left block: -Z G
            K[n + m_e:n + m_e + m_i, 0:n] = -Z @ G
            # Bottom-right block: S
            K[n + m_e:n + m_e + m_i, n + m_e:n + m_e + m_i] = S

        # RHS
        rhs[0:n] = -r_dual
        if m_e > 0:
            rhs[n:n + m_e] = -r_pe
        if m_i > 0:
            rhs[n + m_e:n + m_e + m_i] = -r_cent + Z @ r_pi

        # Solve KKT system
        try:
            sol = np.linalg.solve(K, rhs)
        except np.linalg.LinAlgError as exc:
            raise RuntimeError("KKT system is singular or ill-conditioned.") from exc

        dx = sol[0:n]
        dlam = sol[n:n + m_e]
        dz = sol[n + m_e:n + m_e + m_i]

        # Recover ds from G dx + ds = -r_pi
        if m_i > 0:
            ds = -r_pi - qp.G @ dx
        else:
            ds = np.zeros(0)

        return dx, ds, dlam, dz, r_dual, r_pe, r_pi, r_cent

    # ------------------------------------------------------------------
    # Main solve
    # ------------------------------------------------------------------
    def solve(self, qp: QuadraticProgram):
        """
        Solve the given QP with a primal–dual interior-point method.

        Returns
        -------
        x, s, lambda, z : np.ndarray
            Primal variables x, inequality slacks s, duals for equalities lambda,
            duals for inequalities z.
        """
        x, s, lam, z = self._initial_point(qp)
        n, m_e, m_i = qp.n, qp.m_e, qp.m_i

        for k in range(self.max_iter):
            # Compute residuals and barrier parameter mu
            r_dual, r_pe, r_pi = self._residuals_primal_dual(qp, x, s, lam, z)

            if m_i > 0:
                mu = (s @ z) / m_i
                comp = self._norm_inf(s * z)
            else:
                mu = 0.0
                comp = 0.0

            # Diagnostics
            nd = self._norm_inf(r_dual)
            npe = self._norm_inf(r_pe)
            npi = self._norm_inf(r_pi)

            if self.verbose:
                print(
                    f"iter {k:02d} | "
                    f"r_dual = {nd:.2e}, "
                    f"r_pe = {npe:.2e}, "
                    f"r_pi = {npi:.2e}, "
                    f"comp = {comp:.2e}, "
                    f"mu = {mu:.2e}"
                )

            # Stopping criterion (basic but reasonable):
            if max(nd, npe, npi, comp, mu) < self.tol:
                break

            # Compute Newton step for current sigma, mu
            dx, ds, dlam, dz, _, _, _, _ = self._newton_step(
                qp, x, s, lam, z, self.sigma, mu
            )

            # Step length to keep s > 0, z > 0
            alpha = 1.0
            if m_i > 0:
                idx = ds < 0
                if np.any(idx):
                    alpha = min(alpha, 0.99 * np.min(-s[idx] / ds[idx]))

                idx = dz < 0
                if np.any(idx):
                    alpha = min(alpha, 0.99 * np.min(-z[idx] / dz[idx]))

            # Update variables
            x = x + alpha * dx
            s = s + alpha * ds
            lam = lam + alpha * dlam
            z = z + alpha * dz

        return x, s, lam, z
