# demo.py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from qp import QuadraticProgram, MehrotraIPMSolver
from orca import Agent, get_orca_constraint

# Simulation time step
DT = 0.05

# ORCA time horizon
TAU = 5.0

# Global agent defaults
DEFAULT_RADIUS = 0.5 
DEFAULT_SENSING_RANGE = 5.0  

# QP solver parameters
QP_TOL = 1e-4
QP_MAX_ITER = 15

# Visualization & world settings
WORLD_SIZE = 20
MAX_FRAMES = 300
FOCUS_AGENT = 0

def make_agents():
    """
    Add agents with start and goal position
    """
    return [
        Agent([-10.5, -5.0],  [5.0, 2.5],
              radius=DEFAULT_RADIUS, sensing_range=DEFAULT_SENSING_RANGE),

        Agent([-10.0, 15.0],   [4.0, -4.0],
              radius=DEFAULT_RADIUS, sensing_range=DEFAULT_SENSING_RANGE),

        Agent([3.0, 2.5],    [-12.0, 5.1],
              radius=DEFAULT_RADIUS, sensing_range=DEFAULT_SENSING_RANGE),

        Agent([14.0, -5.0],  [2.0, 2.0],
              radius=DEFAULT_RADIUS, sensing_range=DEFAULT_SENSING_RANGE),
        
        Agent([12.0, 2.0],  [0.0, 0.0],
              radius=DEFAULT_RADIUS, sensing_range=DEFAULT_SENSING_RANGE),
        
        Agent([1.0, -10.0],  [4.0, -2.0],
              radius=DEFAULT_RADIUS, sensing_range=DEFAULT_SENSING_RANGE),
    ]


def run_simulation():
    agents = make_agents()
    solver = MehrotraIPMSolver(tol=QP_TOL, max_iter=QP_MAX_ITER, verbose=False)

    fig, ax = plt.subplots(figsize=(12, 12))
    ax.set_xlim(-WORLD_SIZE, WORLD_SIZE)
    ax.set_ylim(-WORLD_SIZE, WORLD_SIZE)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.set_title("ORCA QP Solver Demo")

    # Draw agent circles
    circles = [
        plt.Circle(a.pos, a.radius,
                   color="royalblue" if i == FOCUS_AGENT else "gray",
                   alpha=0.9 if i == FOCUS_AGENT else 0.5)
        for i, a in enumerate(agents)
    ]
    for c in circles:
        ax.add_patch(c)

    # Focus agent visuals
    focus = agents[FOCUS_AGENT]
    sensing_circle = plt.Circle(focus.pos, focus.sensing_range,
                                fill=False, linestyle="--", alpha=0.2)
    ax.add_patch(sensing_circle)

    vel_arrow = ax.arrow(0, 0, 0, 0, width=0.05, color="red")
    pref_arrow = ax.arrow(0, 0, 0, 0, width=0.03, color="green")

    constraint_lines = [
        ax.plot([], [], "k-", lw=1.5)[0] for _ in range(len(agents) - 1)
    ]


    def update(_frame):
        new_vels = []
        focus_constraints = []

        for i, a in enumerate(agents):
            a.update_pref_vel()

            # QP objective: 0.5*‖v − pref_vel‖^2 =  minimize deviation from preferred
            Q = np.eye(2)
            c = -a.pref_vel

            # Box constraints
            G = [[1, 0], [0, 1], [-1, 0], [0, -1]]
            h = [a.v_max, a.v_max, a.v_max, a.v_max]

            # ORCA half-plane constraints
            local_constraints = []
            for j, b in enumerate(agents):
                if j == i:
                    continue
                if np.linalg.norm(a.pos - b.pos) <= a.sensing_range:
                    g_row, h_val, n_viz, p_viz = get_orca_constraint(a, b, tau=TAU)
                    G.append(g_row)
                    h.append(h_val)
                    if i == FOCUS_AGENT:
                        local_constraints.append((n_viz, p_viz))

            if i == FOCUS_AGENT:
                focus_constraints = local_constraints

            qp = QuadraticProgram(Q, c, G=G, h=h)
            try:
                v_opt, _, _, _, info = solver.solve(qp)
                if info["status"] not in ["optimal", "max_iter_exceeded"]:
                    v_opt = np.zeros(2)
            except:
                v_opt = np.zeros(2)

            new_vels.append(v_opt)

        # Apply velocities
        for i, a in enumerate(agents):
            a.vel = new_vels[i]
            a.pos += a.vel * DT
            circles[i].center = a.pos

        # Update focus visuals
        f = agents[FOCUS_AGENT]
        sensing_circle.center = f.pos
        vel_arrow.set_data(x=f.pos[0], y=f.pos[1], dx=f.vel[0], dy=f.vel[1])
        pref_arrow.set_data(x=f.pos[0], y=f.pos[1], dx=f.pref_vel[0], dy=f.pref_vel[1])

        # Draw ORCA constraint lines
        for line in constraint_lines:
            line.set_data([], [])

        for k, (n_viz, p_viz) in enumerate(focus_constraints):
            if k >= len(constraint_lines):
                break
            center = f.pos + p_viz
            tangent = np.array([-n_viz[1], n_viz[0]])
            p1, p2 = center - 3 * tangent, center + 3 * tangent
            constraint_lines[k].set_data([p1[0], p2[0]], [p1[1], p2[1]])

        return circles + [sensing_circle, vel_arrow, pref_arrow] + constraint_lines

    anim = animation.FuncAnimation(fig, update,
                                   frames=MAX_FRAMES, interval=30, blit=False)

    plt.show()
    return anim


if __name__ == "__main__":
    run_simulation()
