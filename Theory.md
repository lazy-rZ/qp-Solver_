# Introduction to Quadratic Programming using the Interior Point Method

## Prerequisite

To succesfully understand QP and IPM it is important to know the basis for
 1. Vectors and matrices.
 2. Matrix transpose and inverse of a matrix.
 3. Solving linear systems $Ax=b$.
 4. Vector norms such as L2-norm $||x||$.
 5. Positive Semi-Definite Matrices, $x^TAx≥0$ for any vector $x$.
 6. Derivatives, gradients $∇f$ and Hessians $∇²f$.
 7. Legrange Multipliers: If you want to minimize $f(x)$ subject to $g(x) = 0$, you form the Lagrangian: $L(x, λ) = f(x) + λg(x)$. The optimum is found where $∇L(x, λ) = 0$.
 8. Optimization basics
    1. Convexity: A convex function has a "bowl" shape. A convex set has no dents. The key property: Any local minimum is also a global minimum. This is why we require matrix A to be PSD.
    2. Unconstrained Optimization: The first-order optimality condition for an unconstrained problem is $∇f(x) = 0$

## Now to begin understanding QP

We define QP in its standard form as: 

```
Minimize:   f(x) = (1/2)xᵀPx + qᵀx
Subject to: Ax = b    (Equality constraints)
            Gx ≤ h    (Inequality constraints)
            x ≥ 0     (Non-negativity constraints, sometimes included in Gx ≤ h)
```

Where, $x$ is the vector of variables we want to find. $P$ is a symmetric, Positive Semi-Definite matrix. $q$ is a vector. $A$ and $G$ are matrices defining the constraints. $b$ and $h$ are vectors.

The intuition is that the objective $f(x)$ is a "bowl" (if P is PSD). It's a generalization of a parabola to multiple dimensions. Then the constraints $Ax=b$ define a hyperplane (a flat surface) and the constraints $Gx ≤ h$ define a "feasible region" (a convex polyhedron). We are looking for the point inside the feasible region that is at the very bottom of the bowl.

## Before moving on to IPM

It is important to understand what KKT conditions are. In short we can say that they are generalization of Lagrange multipliers to handle inequalities. For the QP problem above, the KKT conditions are necessary and sufficient for optimality (because of convexity). They state that for a solution $x*$ to be optimal, there must exist vectors of Lagrange multipliers $λ*$ (for $Ax=b$) and s* (for $Gx ≤ h$) such that:

```
    Stationarity: Px* + q + Aᵀλ* + Gᵀs* = 0

    Primal Feasibility: Ax* = b, Gx* ≤ h, x* ≥ 0

    Dual Feasibility: s* ≥ 0

    Complementary Slackness: s*ᵀ(Gx* - h) = 0 and (x*)ᵀs* = 0 (This is the key one for inequalities!)
```
Just remeber that for an inequality constraint, either:
* The constraint is "tight" ($G_i x* = h_i$), meaning we are exactly on the boundary, and its multiplier can be positive ($s_i > 0$), OR

* The constraint is "loose" ($G_i x* < h_i$), meaning it's not actively limiting us, and its multiplier must be zero ($s_i = 0$).

The Goal of a QP Solver: Find the triple ($x, λ, s$) that satisfies the KKT conditions.

## The Crux of IPM

The core idea is to handle the inequality constraints ($Gx ≤ h, x ≥ 0$) by converting them into a "barrier" that gets added to the objective function. We use a logarithmic barrier function, because $-log(z)$ approaches infinity as $z$ approaches $0$ from the positive side. This creates an infinitely tall wall that prevents the solution from ever leaving the feasible region.

For the constraint $x ≥ 0$, the barrier is $-μ∑log(x_i)$. The parameter $μ > 0$ is called the barrier parameter.

Our new problem becomes:

```
Minimize:   (1/2)xᵀPx + qᵀx - μ ∑ log(x_i)   (for simplicity, assuming only x≥0 constraints)
Subject to: Ax = b
```

We solve the barrier problem for a sequence of $μ$ values: $μ_0 > μ_1 > μ_2 > ... > 0$. The solutions to these problems form a path called the central path. The IPM follows this path to the optimal solution (which is at $μ=0$).

## Newton's Method

The barrier problem is an equality constrained problem. We can write its KKT conditions (using Lagrange multiplier $λ$ for $Ax=b$).

$$
L_μ(x, λ) = (1/2)xᵀPx + qᵀx - μ ∑ log(x_i) + λᵀ(Ax - b)
$$

The KKT conditions for this problem are:

 1. Stationarity: 

$$
∇_x L_μ = Px + q - μX⁻¹1 + Aᵀλ = 0 
$$

(where X is a diagonal matrix with x on its diagonal, and 1 is a vector of ones).

 2. Primal Feasibility: $Ax - b = 0$

This is a system of nonlinear equations:

```
F(x, λ) = [ Px + q + Aᵀλ - μX⁻¹1 ] = 0
          [       Ax - b         ] 

```

Newton's Method is the premier algorithm for solving systems of nonlinear equations. It works by iteratively linearizing the system. The update step ($Δx, Δλ$) is found by solving the Newton System:

```
J(x, λ) * [Δx] = -F(x, λ)
          [Δλ]
```

Where $J$ is the Jacobian of $F$. For our $F$, the Newton system looks like this:

```
[ P + μX⁻²   Aᵀ ] [Δx] = - [ Px + q + Aᵀλ - μX⁻¹1 ]
[    A       0  ] [Δλ]     [         Ax - b        ]
```
