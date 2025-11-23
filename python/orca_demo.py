# demo.py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from qp import QuadraticProgram, MehrotraIPMSolver
from orca import Agent, get_orca_constraint, get_speed_constraints

# Simulation time step
DT = 0.05

# ORCA time horizon
TAU = 2.0

# Global agent defaults
DEFAULT_RADIUS = 1 
DEFAULT_SENSING_RANGE = 5.0  

# QP solver parameters
QP_TOL = 1e-4
QP_MAX_ITER = 15

# Visualization & world settings
WORLD_SIZE = 15
MAX_FRAMES = 300
N_AGENTS = 10
FOCUS_AGENT = 0

def make_agents():
    """
    Generates N_AGENTS with random start and goal positions.
    Ensures no two start positions overlap and no two goal positions overlap.
    """
    agents = []
    
    MIN_DIST = DEFAULT_RADIUS * 2.5 
    MARGIN = WORLD_SIZE - 2.0

    def get_valid_positions(count, existing_positions=None):
        """
        Generates 'count' random 2D positions that are at least MIN_DIST 
        apart from each other and any optional existing_positions.
        """
        valid_positions = []
        if existing_positions is not None:
            pass

        max_attempts = 10000
        attempts = 0

        while len(valid_positions) < count:
            if attempts > max_attempts:
                raise RuntimeError("World is too small to fit all agents with current radius!")
            
            candidate = np.random.uniform(-MARGIN, MARGIN, 2)
            
            collision = False
            for p in valid_positions:
                if np.linalg.norm(candidate - p) < MIN_DIST:
                    collision = True
                    break
            
            if not collision:
                valid_positions.append(candidate)
            
            attempts += 1
            
        return valid_positions

    # Generate N unique start positions
    start_positions = get_valid_positions(N_AGENTS)

    # Generate N unique goal positions 
    goal_positions = get_valid_positions(N_AGENTS)

    # Create Agent objects
    for i in range(N_AGENTS):
        # specific check: ensure start is not too close to its OWN goal 
        # (otherwise agent stops immediately)
        while np.linalg.norm(start_positions[i] - goal_positions[i]) < 2.0:
             goal_positions[i] = np.random.uniform(-MARGIN, MARGIN, 2)

        agents.append(
            Agent(
                position=start_positions[i],
                goal=goal_positions[i],
                radius=DEFAULT_RADIUS,
                sensing_range=DEFAULT_SENSING_RANGE
            )
        )

    return agents


def run_simulation():
    agents = make_agents()
    solver = MehrotraIPMSolver(tol=QP_TOL, max_iter=QP_MAX_ITER, verbose=False)

    fig, ax = plt.subplots(figsize=(10, 10))
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
        focus_constraints = [] # Reset visualization data for this frame

        for i, a in enumerate(agents):
            a.update_pref_vel()

            # QP Objective: Minimize deviation from preferred velocity
            Q = np.eye(2)
            c = -a.pref_vel

            # Speed Limit Constraints (Octagon Approximation)
            # This generates 8 lines around the velocity circle
            G_circle, h_circle = get_speed_constraints(a.v_max, num_sides=8)
            G = G_circle.tolist()
            h = h_circle.tolist()

            # ORCA Constraints
            local_constraints = [] # Store viz data for THIS agent

            for j, b in enumerate(agents):
                if i == j: continue
                
                # Check distance
                if np.linalg.norm(a.pos - b.pos) <= a.sensing_range:
                    # CALCULATE ORCA
                    result = get_orca_constraint(a, b, tau=TAU)
                    
                    if result is not None:
                        g_row, h_val, n_viz, p_viz = result
                        
                        # Add to solver matrices
                        G.append(g_row)
                        h.append(h_val)
                        
                        # Store for visualization if this is the Focus Agent
                        if i == FOCUS_AGENT:
                            local_constraints.append((n_viz, p_viz))

            # Save focus constraints for drawing phase later
            if i == FOCUS_AGENT:
                focus_constraints = local_constraints

            qp = QuadraticProgram(Q, c, G=G, h=h)
            try:
                v_opt, _, _, _, info = solver.solve(qp)
                
                # STOP ON FAIL LOGIC
                if info["status"] not in ["optimal", "max_iter_exceeded"]:
                    v_opt = np.zeros(2)
                # Double check for NaNs
                elif np.isnan(v_opt).any():
                    v_opt = np.zeros(2)
                    
            except Exception:
                # If solver crashes (singular matrix), stop the agent
                v_opt = np.zeros(2)

            new_vels.append(v_opt)
        
        # Move Agents
        for i, a in enumerate(agents):
            a.vel = new_vels[i]
            a.pos += a.vel * DT
            circles[i].center = a.pos

        # Update Focus Agent Arrows
        f = agents[FOCUS_AGENT]
        sensing_circle.center = f.pos
        # Red arrow = Actual Velocity
        vel_arrow.set_data(x=f.pos[0], y=f.pos[1], dx=f.vel[0], dy=f.vel[1])
        # Green arrow = Preferred Velocity
        pref_arrow.set_data(x=f.pos[0], y=f.pos[1], dx=f.pref_vel[0], dy=f.pref_vel[1])

        # Reset all lines to invisible first
        for line in constraint_lines:
            line.set_data([], [])

        # Draw new lines
        for k, (n_viz, p_viz) in enumerate(focus_constraints):
            if k >= len(constraint_lines):
                break
            
            # GEOMETRY EXPLANATION:
            # p_viz is a point in Velocity Space (relative to agent).
            # We want to draw this in World Space, so we add f.pos.
            origin_in_world = f.pos + p_viz
            
            # The line is perpendicular to the normal n_viz.
            # Tangent vector = (-ny, nx)
            tangent = np.array([-n_viz[1], n_viz[0]])
            
            # Make the line long enough to look infinite on screen
            p1 = origin_in_world - 5.0 * tangent
            p2 = origin_in_world + 5.0 * tangent
            
            constraint_lines[k].set_data([p1[0], p2[0]], [p1[1], p2[1]])

        return circles + [sensing_circle, vel_arrow, pref_arrow] + constraint_lines

    anim = animation.FuncAnimation(fig, update,
                                   frames=MAX_FRAMES, interval=30, blit=False)

    plt.show()
    return anim


if __name__ == "__main__":
    run_simulation()
