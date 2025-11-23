import numpy as np
from qp import QuadraticProgram, MehrotraIPMSolver


def main():

    # Quadratic term + Linear term
    Q = np.array([
        [4.0, 0.0],
        [0.0, 2.0],])
    c = np.array([-3.0, 4.0])        

    # equality constraints
    A = np.array([[1.0, 1.0]])
    b = np.array([10.0])

    # Inequality
    G = np.array([
        [-1.0,  0.0],  
        [ 0.0,  1.0],   
        [-1.0, -2.0],])

    h = np.array([
        -3.0,
        5.0,
        -4.0])

    qp = QuadraticProgram(Q, c, A, b, G, h)

    solver = MehrotraIPMSolver(
        max_iter=50,
        tol=1e-8,
        verbose=True
    )

    x, s, lam, z, info = solver.solve(qp)

    print("x =", x)
    print("slacks =", s)
    print("lambda (eq duals) =", lam)
    print("z (ineq duals) =", z)
    print("Objective value =", 0.5 * x[0] ** 2)
    print("\nStatus:", info["status"], "Iterations:", info["iterations"])


if __name__ == "__main__":
    main()
