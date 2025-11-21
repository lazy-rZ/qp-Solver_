# orca.py
import numpy as np

class Agent:
    def __init__(
        self,
        position,
        goal,
        radius=0.5,
        v_max=2.0,
        sensing_range=5.0
    ):
        self.pos = np.array(position, dtype=float)
        self.goal = np.array(goal, dtype=float)

        self.radius = radius
        self.v_max = v_max
        self.sensing_range = sensing_range

        self.vel = np.zeros(2)
        self.pref_vel = np.zeros(2)

    def update_pref_vel(self):
        direction = self.goal - self.pos
        dist = np.linalg.norm(direction)
        if dist > 0.01:
            speed = self.v_max if dist > 1.0 else self.v_max * dist
            self.pref_vel = direction / dist * speed
        else:
            self.pref_vel = np.zeros(2)


def get_orca_constraint(a, b, tau=2.0):
    """
    Return ORCA constraint: G_row, h_val, normal_for_visualization, P_visualization
    Constraint is G_row ⋅ v <= h_val
    """
    x_rel = b.pos - a.pos
    v_rel = a.vel - b.vel

    dist_sq = np.dot(x_rel, x_rel)
    R = a.radius + b.radius
    R2 = R * R

    u = np.zeros(2)
    n = np.zeros(2)

    if dist_sq > R2:
        # No collision
        w = v_rel - x_rel / tau
        w_sq = np.dot(w, w)
        dot_w_x = np.dot(w, x_rel)

        if dot_w_x < 0 and (dot_w_x**2) > R2 * w_sq:
            # Cutoff circle
            w_len = np.sqrt(w_sq)
            unit_w = w / w_len
            u = (R / tau - w_len) * unit_w
            n = unit_w
        else:
            # Legs
            leg_len = np.sqrt(dist_sq - R2)
            det = x_rel[0] * w[1] - x_rel[1] * w[0]

            if det > 0:
                leg = np.array([
                    x_rel[0] * leg_len - x_rel[1] * R,
                    x_rel[0] * R + x_rel[1] * leg_len
                ]) / dist_sq
            else:
                leg = np.array([
                    x_rel[0] * leg_len + x_rel[1] * R,
                    -x_rel[0] * R + x_rel[1] * leg_len
                ]) / dist_sq

            u = np.dot(v_rel, leg) * leg - v_rel
            n = u / np.linalg.norm(u)
    else:
        # Collision
        dt = 0.1
        w = v_rel - x_rel / dt
        w_len = np.linalg.norm(w)
        unit_w = w / w_len
        u = (R / dt - w_len) * unit_w
        n = unit_w

    P = a.vel + 0.5 * u
    G_row = -n
    h_val = -np.dot(n, P)
    return G_row, h_val, n, P
