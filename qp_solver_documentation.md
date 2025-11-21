
# Quadratic Program (QP) Interior-Point Solver  
### *Documentation of Implementation*

---

# 1. Problem Definition

We want to solve a **convex quadratic program**:

$$
\begin{aligned}
\min_x \quad & \frac12 x^T Q x + c^T x \\
\text{s.t.} \quad & A x = b \\
                  & G x \le h
\end{aligned}
$$

Where:

- $ Q \in \mathbb{R}^{n \times n} $ is **positive semidefinite**
- $ c \in \mathbb{R}^n $
- $ A \in \mathbb{R}^{m_e \times n} $
- $ G \in \mathbb{R}^{m_i \times n} $

---

# 2. Slack Variables

Convert inequalities:

$$
Gx \le h
$$

into equalities using slacks:

$$
Gx + s = h, \quad s \ge 0
$$

Dual variables:

- $\lambda$: equalities
- $z$: inequalities

---

# 3. KKT Optimality Conditions

Optimal solution $(x^*, s^*, \lambda^*, z^*)$:

### 1. Stationarity
$$
Qx + c + A^T\lambda + G^T z = 0
$$

### 2. Feasibility
$$
A x = b, \quad Gx + s = h
$$

### 3. Complementarity
$$
s_i z_i = 0
$$

Interior-point methods relax this:

$$
s_i z_i = \mu
$$

where:

$$
\mu = \frac{s^T z}{m_i}
$$

---

# 4. Newton System

Linearize KKT conditions and solve:

$$
(dx, ds, d\lambda, dz)
$$

Reduced Newton system:

$$
\begin{bmatrix}
Q & A^T & G^T \\
A & 0 & 0 \\
-ZG & 0 & S
\end{bmatrix}
\begin{bmatrix}
dx \\ d\lambda \\ dz
\end{bmatrix}
=
\begin{bmatrix}
-r_{dual} \\ -r_{pe} \\ -r_{cent} + Z r_{pi}
\end{bmatrix}
$$

This is the system solved in the implementation.

---

# 5. Mehrotra Predictor–Corrector Algorithm

IPM workflow:

---

## Step 1 — Compute residuals

- Dual:  
  `r_dual = Qx + c + Aᵀλ + Gᵀz`
- Equality:  
  `r_pe = Ax - b`
- Inequality:  
  `r_pi = Gx + s - h`
- Complementarity:  
  `r_cent = s * z`

---

## Step 2 — Predictor Step (σ = 0)

Solve Newton system → affine-scaling step:

$$
(dx_{aff}, ds_{aff}, d\lambda_{aff}, dz_{aff})
$$

Compute full step length keeping $s > 0, z > 0$:

$$
\alpha_{aff}
$$

Estimate new complementarity:

$$
\mu_{aff} = \frac{(s+\alpha ds)^T(z+\alpha dz)}{m_i}
$$

---

## Step 3 — Compute σ

$$
\sigma = (\mu_{aff}/\mu)^3
$$

Key Mehrotra heuristic.

---

## Step 4 — Corrector Step

Correct complementarity for second-order effects:

$$
r_{cent\_corr} = s z + ds_{aff} dz_{aff} - \sigma \mu
$$

Solve Newton system again.

---

## Step 5 — Fraction-to-boundary step size

$$
\alpha = \eta \cdot \min\{ \alpha_{pri}, \alpha_{dual} \}
$$

with $ \eta = 0.99 $.

---

## Step 6 — Update

$$
x \leftarrow x + \alpha dx
$$
$$
s \leftarrow s + \alpha ds
$$
$$
\lambda \leftarrow \lambda + \alpha d\lambda
$$
$$
z \leftarrow z + \alpha dz
$$

---

# 6. Pseudocode

```
function solve_qp(Q, c, A, b, G, h):

    initialize x, s > 0, λ, z > 0

    for k in 1..max_iter:

        compute residuals
        compute µ = (sᵀ z)/m_i

        if residuals < tol and µ < tol:
            return solution

        # Predictor step (σ = 0)
        solve Newton system → (dx_aff, ds_aff, dλ_aff, dz_aff)
        compute α_aff
        compute μ_aff
        σ = (μ_aff / μ)^3

        # Corrector step
        build corrected complementarity residual
        solve Newton system → (dx, ds, dλ, dz)

        compute α_pri, α_dual
        α = η * min(α_pri, α_dual)

        update x, s, λ, z

    return failure
```

---

# 7. KKT Matrix Structure

$$
K =
\begin{bmatrix}
Q + δI & A^T & G^T \\
A & 0 & 0 \\
-ZG & 0 & S + δI
\end{bmatrix}
$$

Where:

- S = diag(s)
- Z = diag(z)
- δ = small regularization

---

# 8. Convergence Criteria

Solver stops when:

$$
\|r_{dual}\| < tol
$$
$$
\|r_{pe}\| < tol
$$
$$
\|r_{pi}\| < tol
$$
$$
\mu < tol
$$

---

# 9. Verification

To verify correctness:

### Check KKT residuals  
### Test known solutions  
### Check feasibility Gx≤h, Ax=b, s≥0, z≥0  

---

