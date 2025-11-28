# Introduction Quadratic Problem

In this document you will find all relevant theory needed to understand quadratic programming (QP), together with a focused tutorial on the implementation in this project.

## Convex Optimization Basics

An optimization problems looks like:

$$
min_x f(x), x \in C
$$

where $f(x)$ is our objective function and $C$ is a set of constraints usually descibred by equations and inequalities.

But solving But solving general optimization is extremely hard unless they are convex. This is becasue in convex problems:
1. Any local minimum is a global minimum.
2. Problems are numerically stable.
3. Duality theory can be applied and it is powerful.
4. We can use efficient algorithms like interior-point methods.

Let start off with some definiations:

A set is said to be convex if a line segment between any two points is inside the set. Examples are half-spaces, hyperplanes and even instersection of convex sets. Think of a convex set as a sphere in 3D, picking any two points inside this set, we can draw a line that connect them and the line itself will be inside the set. Now if that sphere had a whole, then we would not have a convex set.

A function is said to be convex if the region above the graph is convex, or just imagine the graph as bowl shaped. All linear functions are convex and quadratics with postive semidefinite Q are also convex.

Now we can define our convex optimization problem where our objective function is convex and our equality constraint is linear and inequality us convex.

## Quadratic Programs

A QP is convex optimization problem:

$$
min_x \frac{1}{2} x^TQx + c^Tx
$$

subject to : $Ax=b$ and $Gx\le h$.

Here Q is positive positive semidefinite meaning it $x^TQx\ge 0$ for all vectors $x$. In other words the objective function has a bowl shape and never curves downward. 

## The Duality Principle

For our context duality is the idea that every optimization problem (the primal) has an associated dual problem whose solutions give information about the primal. For instance for the convex set we have a primal defintion above, now the dual perspective of convex sets are a hyperplane such that the entire set is on the positive side. Now if we repeat this for all possible hyperplanes, their intersection of all positive regions will give us the set it self. This can be thought of as wrapping the sphere in many hyperplanes, and the intersection of all hyperplanes will just be the sphere itself. Similarly, the dual definition of convex function is the tangent draw at some point of the graph will always be under the entire graph, or rephrased as the graph will alwasy be above it's tangets.

This idea will be usefull when we starting implementing the QP solver.

## Lagrangians and KKT Conditions

The Lagrangian is a single function that combines the objective function we want to minimize with all the constraints and their dual variables. 

For the constrained convex problem, we form the Langrangian:

$$
L(x,λ,z) = \frac{1}{2} ​x^TQx+c^Tx+λ^T(Ax−b)+z^T(Gx−h)
$$

where $λ$ is the dual variables for the equality constraint and $z\ge 0$: is dual variables for inequality constraints. 

I good intuitive explanation is that solving a optimization with no constraints is easy, so we remove the constraints by incorporationg them into the objective function as pentaly. If the solution is satisfied by the constraint then the penalty is zero, otherwise it is very large. 

The dual problem is a second optimization problem derived from the Lagrangian. We ask, what is the tightest lower bound on the optimal value of the primal problem? Therefore we start by first minimizng the Lagrangian with respect to x, this gives the dual function:

$$
g(λ,z) = x_{inf​} \space L(x,λ,z) 
$$

which tell us how small the objective can get for the given dual variables. Then we want to maximize the dual function subject to $z\ge 0$. Therefore we have a min max problem.

### Short Recap:

Assume we have a general optimization problem:

```
    minimize{x} f(x)
    subject to:
        h_i(x) = 0 for i = 1,...
        g_j(x) <= 0 for j = 1,... 
```

rewrite as:

```
    minimize{x} f(x) + maximize{λ_i, z_j >= 0} Sum(λ_i * h_i(x)) + Sum(z_j g_j(x))
```

To get the dual problem we pull out maximize operator switch the order:

```
    maximize{λ_i, z_j >= 0} minimize{x} f(x) + Sum(λ_i * h_i(x)) + Sum(z_j g_j(x))
```

The function inside is the Lagrangian we defined for QP before.

Now we did all this because: for convex problems gives us a lower bound on the optimal value of the primal. However under mild conditions, primal optimum = dual optimum which is important because it means solving the dual tell you the primal answer. 

### So goin back to the Lagrangian

To find where the Lagrangian is minimized over x, we take the gradient of the Lagrangian: $∇_x ​L(x,λ,z) = 0$. This is our stationarity condition defined as $∇_x ​L = Qx+c+A^T λ + G^T z=0$.

Now we combine everything which gives us the KKT Conditions. To handle the inequalities more cleanly, we introduce slack variables: $s = h − Gx$ where $s\ge 0$.

The KKT conditions are the first order optimality conditions for constrained convex problems where we have:

1. Primal Feasibility

* Equality constraints: $Ax−b=0$
* Slack definition: $Gx+s−h=0$
* Slack positivity: $s\ge 0$

These are constraints on the primal variables.

2. Dual Feasibility

* These ensure z is a valid dual variable for inequalities: $z\ge 0$

3. Stationarity of the Lagrangian: $∇_x ​L(x,λ,z) = 0$

4. Complementarity Conditions: $s_i * z_i​ = 0$

* Meaning if slack $s_i > 0$ (constraint not tight) then $z_i = 0$ or if constraint is tight $s_i = 0$ then $z_i > 0$.
* We will later relax this to $s_i z_i​ = μ$ and let $μ$ go to zero.

## Write the KKT Conditions as a System of Equations

We need to modify the last equation because otherwise the system is non-smooth, nondifferentiable, and impossible for Newton’s method to solve directly. 

Interior-point methods replaces the Complementarity Conditions with $s_i z_i​ = μ$ where if $μ$ is large then point stays away from the boundary otherwise if $μ$ is small then point approaches the true KKT solution. So As $μ$ goes to 0, we converge to the exact KKT conditions.

Now we have a system of nonlinear equations: $F(x,s,λ,z,μ)=0$ so we apply Newton’s method to compute search directions: $(dx,ds,dλ,dz)$. Consider the argument as vector $w$, then to solve $F(w)=0$, Newton’s method computes a search direction $J_f(w)dw=-f(x)$ where $J_f$ is the jacobian of all partial derivatives, our update is thus $w_{new}​=w+α*dw$, where $0<α\le 1$ is the step size.

This is a huge linear system so we can use Schur complement, a technique for solving block-structured linear systems.

### Schur Complement

Assume we have a system:

$$
\begin{bmatrix} A & B^T \\\ B & 0 \end{bmatrix} \begin{bmatrix} x \\\ y \end{bmatrix} = \begin{bmatrix} r1 \\\ r2 \end{bmatrix}
$$

In our case this is the structure of the KKT matrix. The trick is to solve for x using a reduced system (the Schur complement) then recover y afterward. 

When you linearize the KKT system, the Newton system becomes:

$$
\begin{bmatrix} (Q+G^TS^{-1}ZG) & A^T \\\ A & 0 \end{bmatrix} \begin{bmatrix} dx \\\ dλ \end{bmatrix} = \begin{bmatrix} rhs_x \\\ rhs_λ \end{bmatrix}
$$

Where S=diag(s), Z=diag(z), $S^{−1}Z$ comes from differentiating $sz=μ$ and rhs_x and rhs_λ come from residuals. This reduced matrix is called the augmented Hessian or the Schur complement of (S,Z). This is just a optimization trick because, the full system size $(n+m_e+m_i)^2$ where using Schur complement, we solve instead a system of size $(n + m_e)$.

### Compute Step Sizes (fraction-to-boundary rule)

After solving for the search directions, we must ensure:

* $s+αds>0$
* $z+αdz>0$

because s and z must remain strictly positive inside interior-point methods. So we compute:
```
α_{pri} ​= max{α : s + αds >0}
α_{dual} ​= max{α : z + αdz >0}
```

Then scale both by a safety factor $η$ (e.g., 0.99).

## Updating variables and Mehrotra’s predictor-corrector step

Now we can simply just update each variables as following:
* x ← x + αdx
* s ← s + αds
* λ ← λ + αdλ
* z ← z + αdz

We also need to Decrease μ, which is where Mehrotra’s improvement comes in. This gives much faster convergence. It computes:

* an affine-scaling step (predictor)
* an estimate of μ_aff
* a centering correction step
* an adaptive new μ based on curvature

Remember that we in the standard primal-dual interior-point method solved the perturbed KKT system: $s_i ​z_i​=μ$. Where:

* μ > 0 is the barrier parameter
* μ → 0 as we converge

The basic algorithm does 1 Newton step per iteration, decreasing μ each time. But the affine Newton step (with μ = 0) often pushes the iterates too close to the boundary or in the wrong direction. If we then shrink μ, the iterates become poorly centered (far from the “central path”).

### Predictor Step (Affine Scaling Direction)
We solve the KKT Newton system with: $sz=0$, this gives the affine scaling direction. This direction shows us where would we go if we fully removed the barrier and tried to reach the KKT solution directly. But this step tends to drive: some s → 0 and or some z → 0 Too fast.

We compute how far we can go before hitting boundaries : $α_{aff}^{pri}​$ and $α_{aff}^{dual}$ and estimate the affine complementarity:

$$
μ_{aff}​ = \frac{(s + α_{aff}​ ds_{aff}​)^T (z + α_{aff} ​dz_{aff}​)}{m}​
$$

This value is usually much smaller than μ. It tells us if we followed this direction all the way, how close would we get to the actual KKT complementarity. This produces information about curvature and scaling.

### Compute Adaptive Centering Parameter σ
Classical methods choose σ $\in$ [0.1, 0.5, 0.9] manually.

But we can compute σ automatically using $μ_{aff}$: $σ=(\frac{μ_{aff}}{μ}​​)3$.

This formula has magical properties:

* If predictor moves us far along the path → σ small → smaller centering step
* If predictor messes up centrality → $μ_{aff}$ ≫ μ → σ large → stronger correction

### Corrector Step
Now we solve another Newton system, but with a right-hand side modified to include second-order correction terms: $s*z + ds_{aff} * dz_{aff} −σμI​$. 

This corrector step:
* improves the curvature approximation
* restores centrality
* “undoes” the predictor’s tendency to move us toward the boundary
* allows much larger steps
* accelerates convergence (often from 50 iterations to fewer than 10)

### Combined Step = Predictor + Corrector

The final search direction is: $dx=dx_{corr}$, $ds=ds_{corr}$, $z=dz_{corr}$, $dλ=dλ_{corr}$. This direction is: aggressive (thanks to predictor) and safe and centered (thanks to corrector). Then we compute: $α=min(α_{pri}​,α_{dual}​)$ and take the step.

### Recap
1. Taking an aggressive “predictor” step with μ = 0 (affine scaling).
2. Computing how close this predictor step gets to satisfying true complementarity $(μ_{aff})$.
3. Choosing an adaptive centering parameter σ from $μ_{aff}$.
4. Taking a second “corrector” step that restores centrality and curvature.
5. Using this combined direction to take large, safe steps.
