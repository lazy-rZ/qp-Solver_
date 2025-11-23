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
    R = a.radius + b.radius + 0.5 # Allow for space between the agents
    R_sq = R * R

    if dist_sq < R_sq:
        # Must separate immediately. 
        # This is always a constraint, regardless of velocity.
        dt = 0.1 
        w = v_rel - x_rel / dt
        w_len = np.linalg.norm(w)
        unit_w = w / w_len
        
        u = (R / dt - w_len) * unit_w
        n = unit_w
        
        P = a.vel + 0.5 * u
        return -n, -np.dot(n, P), n, P

    else:
        # The "Cutoff Center" is x_rel / tau
        center = x_rel / tau
        # Vector from Cutoff Center to Relative Velocity
        w = v_rel - center
        w_sq = np.dot(w, w)
        
        # Project w onto the relative position line
        dot_w_x = np.dot(w, x_rel)
        
        # (Collision within time tau, but mainly due to converging speed)
        if dot_w_x < 0 and (dot_w_x**2) > R_sq * w_sq:
            w_len = np.sqrt(w_sq)
            
            # Radius of the VO cutoff circle is R / tau
            if w_len < R / tau:
                # INSIDE Cutoff Circle -> CONSTRAIN
                unit_w = w / w_len
                u = (R / tau - w_len) * unit_w
                n = unit_w
                
                P = a.vel + 0.5 * u
                return -n, -np.dot(n, P), n, P
            else:
                # OUTSIDE Cutoff Circle -> SAFE
                return None

        else:
            leg_len = np.sqrt(dist_sq - R_sq)
            
            # Determinant to see which side of the center line we are on
            det = x_rel[0] * w[1] - x_rel[1] * w[0]
            
            # Calculate the direction of the relevant leg
            if det > 0:
                # Left leg
                leg_dir = np.array([
                    x_rel[0] * leg_len - x_rel[1] * R,
                    x_rel[0] * R + x_rel[1] * leg_len
                ]) / dist_sq
            else:
                # Right leg
                leg_dir = np.array([
                    x_rel[0] * leg_len + x_rel[1] * R,
                    -x_rel[0] * R + x_rel[1] * leg_len
                ]) / dist_sq

            # Check if v_rel is "behind" the leg (closer to the centerline than the leg)
            # If we project v_rel onto the normal of the leg, does it point "in"?
            
            # Simpler check used in RVO2:
            # Since we already failed the Cutoff Circle check, if the projection of v_rel onto the leg line is positive, we are in the cone.
            
            # We compare v_rel against the leg direction. 
            # u = (projection of v_rel onto leg) - v_rel
            dot_v_leg = np.dot(v_rel, leg_dir)
            u = dot_v_leg * leg_dir - v_rel
            
            '''
            If the vector 'u' points AWAY from the VO interior, we are actually safe.
            But geometrically, if we reached this block, and dist_sq > R_sq, and we aren't in the cutoff circle...
            
            We need to verify we are actually heading toward the object.
            If dot_v_leg > 0, we are moving roughly along the leg.
            But we only collide if we are to the "inside" of the leg.
            
            Strict check: Is the determinant of (leg_dir, v_rel) having the correct sign?
            A robust way is to check the length of the correction vector u.
            
            If we are outside, the closest point on the VO is the vertex (cutoff), but we handled that in 2A. Or we are diverging.
            
            To keep it simple for this DEMO, we only assume if we are in this geometric region relative to the lines, we calculate the constraint.
            '''
            n = u / np.linalg.norm(u)
            P = a.vel + 0.5 * u
            
            # If the required velocity change 'u' is tiny, we are practically safe.
            if np.linalg.norm(u) < 1e-6:
                return None

            # Also explicit check if v_rel is actually inside the leg half-plane.
            # If constraint normal points towards v_rel, it means v_rel is already satisfying the constraint (outside).
            if np.dot(n, a.vel - P) >= 0:
                 return None

            return -n, -np.dot(n, P), n, P


def get_speed_constraints(v_max, num_sides=8):
    """
    Generates G and h for a regular polygon approximation of a circle.
    Results in: G_row @ v <= v_max
    """
    angles = np.linspace(0, 2*np.pi, num_sides, endpoint=False)
    
    # The normal vectors (G rows) are just (cos(theta), sin(theta))
    G = np.column_stack((np.cos(angles), np.sin(angles)))
    h = np.full(num_sides, v_max)
    
    return G, h
