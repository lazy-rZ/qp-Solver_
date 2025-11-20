import numpy as np
from qp import QuadraticProgram, PrimalDualInteriorPointSolver


def main():
    # Problem:
    #   minimize 0.5 * x^2
    #   subject to x >= 1
    #
    # Write as G x <= h:
    #   -x <= -1

    Q = np.array([[1.0]])
    c = np.array([0.0])

    A = None
    b = None

    G = np.array([[-1.0]])
    h = np.array([-1.0])

    qp = QuadraticProgram(Q, c, A, b, G, h)
    solver = PrimalDualInteriorPointSolver(
        max_iter=50,
        tol=1e-8,
        sigma=0.5,
        verbose=True,
    )

    x, s, lam, z = solver.solve(qp)

    print("\nSolution:")
    print("x       =", x)
    print("slacks  =", s)
    print("lambda  =", lam)
    print("z (ineq duals) =", z)
    print("Objective value 0.5 x^2 =", 0.5 * x[0] ** 2)


if __name__ == "__main__":
    main()
