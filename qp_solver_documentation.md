# Quadratic Programming Solver — Documentation

This module implements a convex quadratic programming (QP) solver using a **primal–dual interior-point method** with **Mehrotra’s predictor–corrector algorithm**.  
It includes:

- `QuadraticProgram` – a container for QP data  
- `MehrotraIPMSolver` – the optimization solver  

---

# 1. Problem Formulation

The solver handles problems of the form:

```
minimize    0.5 * x^T Q x + c^T x
subject to  A x = b
            G x <= h
```

Inequality constraints are rewritten with slack variables `s >= 0`:

```
G x + s = h
```

Dual variables:

- `lambda` for equalities  
- `z >= 0` for inequalities  

---

# 2. QuadraticProgram Class

### Constructor

```python
QuadraticProgram(Q, c, A=None, b=None, G=None, h=None)
```

This class stores and validates all QP data.

### Stored Attributes

| Attribute | Description |
|----------|-------------|
| `Q` | Hessian matrix (n × n) |
| `c` | Linear term (n) |
| `A` | Equality constraint matrix (m_e × n) |
| `b` | Equality constraint RHS (m_e) |
| `G` | Inequality constraint matrix (m_i × n) |
| `h` | Inequality constraint RHS (m_i) |
| `m_e` | Number of equality constraints |
| `m_i` | Number of inequality constraints |

### Validation Performed

- `Q` must be square and match `len(c)`
- If provided, `A` shape must match `b`
- If provided, `G` shape must match `h`

---

# 3. MehrotraIPMSolver Class

A primal–dual interior-point solver implementing Mehrotra’s predictor–corrector algorithm.

### Initialization

```python
MehrotraIPMSolver(
    max_iter=50,
    tol=1e-8,
    mu_tol=None,
    verbose=False,
    eta=0.99,
    regularization=1e-9
)
```

### Parameters

- **max_iter**: Maximum iterations  
- **tol**: Tolerance for KKT residuals  
- **mu_tol**: Complementarity tolerance (defaults to `tol`)  
- **verbose**: Print iteration logs  
- **eta**: Fraction-to-boundary parameter (0 < eta < 1)  
- **regularization**: Small diagonal stabilization term  

---

# 4. Algorithm Overview

The solver uses standard primal–dual interior-point steps:

### Residuals

- **Dual residual**: `Qx + c + A^T lam + G^T z`
- **Primal equality residual**: `A x - b`
- **Primal inequality residual**: `G x + s - h`
- **Complementarity residual**: `s * z`

Complementarity measure:

```
mu = sum(s * z) / m_i
```

---

# 5. Predictor Step

Computes a direction assuming no centering correction.  
Determines how far the step can go before violating positivity of `s` and `z`.

```
alpha_aff_pri  = max step until s stays >= 0
alpha_aff_dual = max step until z stays >= 0
```

Predicts new complementarity:

```
mu_aff = (s + alpha_aff_pri * ds_aff) dot (z + alpha_aff_dual * dz_aff) / m_i
```

---

# 6. Corrector Step

Computes centering parameter:

```
sigma = (mu_aff / mu)^3
```

Creates corrected complementarity residual:

```
r_cent_corrected = s*z + ds_aff * dz_aff - sigma * mu * 1
```

Solves KKT system again using corrected residual.

---

# 7. KKT System Solve

The solver forms a reduced KKT system:

```
[ Q + G^T (S^-1 Z) G    A^T ] [ dx   ] = [ rhs_x ]
[ A                       0 ] [ dlam ]   [ rhs_e ]
```

Depending on constraints:

- With no equalities → Cholesky solve  
- Otherwise → dense linear solve

Slack and dual steps (`ds`, `dz`) are recovered afterward.

---

# 8. Step Size Selection

To maintain positivity:

```
alpha_pri  = eta * min_i( -s_i / ds_i  ) for ds_i < 0
alpha_dual = eta * min_i( -z_i / dz_i ) for dz_i < 0
```

Final step:

```
alpha = min(alpha_pri, alpha_dual)
```

---

# 9. Update

```
x   = x + alpha * dx
lam = lam + alpha * dlam
s   = s + alpha * ds
z   = z + alpha * dz
```

---

# 10. Termination Criteria

The solver stops when:

- All residuals are below tolerance `tol`
- Complementarity `mu` is below `mu_tol`

Or stops with:

- `"numerical_issue"`
- `"max_iter_exceeded"`

---

# 11. Solver Interface

### Solve a QP

```python
x, s, lam, z, info = solver.solve(qp)
```

### Returns

| Output | Description |
|--------|-------------|
| `x` | Primal solution |
| `s` | Slack variables |
| `lam` | Equality dual variables |
| `z` | Inequality dual variables |
| `info` | Dictionary with status, residual history, iterations |

---

# 12. Example

```python
import numpy as np
from qp import QuadraticProgram, MehrotraIPMSolver

Q = np.eye(2)
c = np.array([-1.0, -1.0])
G = np.array([[1, 2],
              [-1, 0]])
h = np.array([1, 0])

qp = QuadraticProgram(Q, c, G=G, h=h)
solver = MehrotraIPMSolver(verbose=True)

x, s, lam, z, info = solver.solve(qp)

print("Solution:", x)
print("Status:", info["status"])
```

---

# 13. Numerical Stability Features

- Diagonal regularization of `Q`  
- Regularization of slack inverses  
- Fallback from Cholesky to general solve  
- Fraction-to-boundary safeguard  

Ensures robustness for poorly conditioned problems.
