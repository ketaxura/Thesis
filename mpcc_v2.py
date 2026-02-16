"""
Model Predictive Contouring Control (MPCC) for Unicycle Robot
Tracks a straight line path using CasADi and IPOPT
"""

import numpy as np
import casadi as ca
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation



class MPCCUnicycle:
    def __init__(self):
        # Prediction horizon
        self.N = 20  # Number of prediction steps
        self.dt = 0.1  # Time step (s)
        
        # Unicycle parameters
        self.v_max = 2.0  # Max linear velocity (m/s)
        self.omega_max = 2.0  # Max angular velocity (rad/s)
        
        # State bounds
        self.x_min, self.x_max = -10, 10
        self.y_min, self.y_max = -10, 10
        
        # Path parameters (straight line: y = mx + c)
        self.path_slope = 0.5  # Slope of the line
        self.path_intercept = 2.0  # y-intercept
        
        # MPCC weights
        self.Q_contour = 100.0  # Contouring error weight
        self.Q_lag = 1.0  # Lag error weight
        self.R_v = 0.1  # Linear velocity weight
        self.R_omega = 0.1  # Angular velocity weight
        self.Q_progress = 50.0  # Progress weight (encourage moving along path)
        
        # Setup optimization problem
        self.setup_optimization()
        
    def setup_optimization(self):
        """Setup the CasADi optimization problem"""
        
        # State variables: [x, y, theta, s]
        # s is the path parameter (position along the line)
        self.n_states = 4
        self.n_controls = 2
        
        # Create optimization variables
        self.opti = ca.Opti()
        
        # Decision variables
        self.X = self.opti.variable(self.n_states, self.N + 1)  # States
        self.U = self.opti.variable(self.n_controls, self.N)  # Controls
        
        # Parameters (initial state)
        self.X0 = self.opti.parameter(self.n_states, 1)
        
        # Extract state variables
        x = self.X[0, :]
        y = self.X[1, :]
        theta = self.X[2, :]
        s = self.X[3, :]  # Path parameter
        
        # Extract control variables
        v = self.U[0, :]  # Linear velocity
        omega = self.U[1, :]  # Angular velocity
        
        # Cost function
        cost = 0
        
        for k in range(self.N):
            # Get position on reference path at parameter s[k]
            x_ref, y_ref, psi_ref = self.get_path_point(s[k])
            
            # Contouring error (perpendicular distance to path)
            # For line: ax + by + c = 0, distance = |ax + by + c| / sqrt(a^2 + b^2)
            # Our line: y = mx + c => mx - y + c = 0
            a = self.path_slope
            b = -1.0
            c = self.path_intercept
            norm_factor = ca.sqrt(a**2 + b**2)
            e_contour = (a * x[k] + b * y[k] + c) / norm_factor
            
            # Lag error (distance along the path)
            # Project current position onto the path
            dx = x[k] - x_ref
            dy = y[k] - y_ref
            # Component along path direction
            path_dir_x = ca.cos(psi_ref)
            path_dir_y = ca.sin(psi_ref)
            e_lag = -(dx * path_dir_x + dy * path_dir_y)
            
            # Stage cost
            cost += self.Q_contour * e_contour**2
            cost += self.Q_lag * e_lag**2
            cost += self.R_v * v[k]**2
            cost += self.R_omega * omega[k]**2
            
            # Progress incentive (encourage moving forward along path)
            if k < self.N - 1:
                cost -= self.Q_progress * (s[k+1] - s[k])
        
        # Terminal cost
        x_ref_final, y_ref_final, _ = self.get_path_point(s[-1])
        a = self.path_slope
        b = -1.0
        c = self.path_intercept
        norm_factor = ca.sqrt(a**2 + b**2)
        e_contour_final = (a * x[-1] + b * y[-1] + c) / norm_factor
        cost += 10 * self.Q_contour * e_contour_final**2
        
        self.opti.minimize(cost)
        
        # Dynamics constraints
        for k in range(self.N):
            x_next = x[k] + v[k] * ca.cos(theta[k]) * self.dt
            y_next = y[k] + v[k] * ca.sin(theta[k]) * self.dt
            theta_next = theta[k] + omega[k] * self.dt
            s_next = s[k] + v[k] * ca.cos(theta[k] - self.get_path_heading(s[k])) * self.dt
            
            self.opti.subject_to(self.X[0, k+1] == x_next)
            self.opti.subject_to(self.X[1, k+1] == y_next)
            self.opti.subject_to(self.X[2, k+1] == theta_next)
            self.opti.subject_to(self.X[3, k+1] == s_next)
        
        # Initial condition constraint
        self.opti.subject_to(self.X[:, 0] == self.X0)
        
        # State constraints
        self.opti.subject_to(self.opti.bounded(self.x_min, x, self.x_max))
        self.opti.subject_to(self.opti.bounded(self.y_min, y, self.y_max))
        self.opti.subject_to(s >= 0)  # Path parameter must be positive
        
        # Control constraints
        self.opti.subject_to(self.opti.bounded(-self.v_max, v, self.v_max))
        self.opti.subject_to(self.opti.bounded(-self.omega_max, omega, self.omega_max))
        
        # Solver options
        opts = {
            'ipopt.print_level': 0,
            'print_time': 0,
            'ipopt.max_iter': 100,
            'ipopt.tol': 1e-4,
            'ipopt.acceptable_tol': 1e-3
        }
        self.opti.solver('ipopt', opts)
        
    def get_path_point(self, s):
        """Get point on straight line path at parameter s"""
        # Parametric form: (x, y) = (s, m*s + c)
        x_ref = s
        y_ref = self.path_slope * s + self.path_intercept
        psi_ref = ca.arctan(self.path_slope)  # Path heading (constant for straight line)
        return x_ref, y_ref, psi_ref
    
    def get_path_heading(self, s):
        """Get path heading at parameter s (constant for straight line)"""
        return ca.arctan(self.path_slope)
    
    def solve(self, x0):
        """Solve the MPCC optimization problem"""
        # Set initial state
        self.opti.set_value(self.X0, x0)
        
        # Set initial guess (warm start with previous solution if available)
        if not hasattr(self, 'sol'):
            # First solve - use simple initialization
            x_guess = np.linspace(x0[0], x0[0] + 5, self.N + 1)
            y_guess = np.linspace(x0[1], x0[1] + 5 * self.path_slope, self.N + 1)
            theta_guess = np.ones(self.N + 1) * np.arctan(self.path_slope)
            s_guess = np.linspace(x0[3], x0[3] + 5, self.N + 1)
            
            self.opti.set_initial(self.X[0, :], x_guess)
            self.opti.set_initial(self.X[1, :], y_guess)
            self.opti.set_initial(self.X[2, :], theta_guess)
            self.opti.set_initial(self.X[3, :], s_guess)
            self.opti.set_initial(self.U[0, :], np.ones(self.N) * 0.5)
            self.opti.set_initial(self.U[1, :], np.zeros(self.N))
        else:
            # Warm start with shifted previous solution
            self.opti.set_initial(self.X, self.sol.value(self.X))
            self.opti.set_initial(self.U, self.sol.value(self.U))
        
        # Solve
        try:
            self.sol = self.opti.solve()
            success = True
        except RuntimeError as e:
            print(f"Solver failed: {e}")
            success = False
            
        if success:
            u_opt = self.sol.value(self.U[:, 0])
            x_pred = self.sol.value(self.X)
            return u_opt, x_pred
        else:
            return np.array([0.0, 0.0]), None


def simulate_mpcc():
    """Simulate the MPCC controller"""
    
    # Create controller
    controller = MPCCUnicycle()
    
    # Initial state [x, y, theta, s]
    x0 = np.array([0.0, 0.0, 0.5, 0.0])
    
    # Simulation parameters
    T_sim = 15.0  # Simulation time (s)
    dt = controller.dt
    n_steps = int(T_sim / dt)
    
    # Storage for trajectory
    x_history = [x0[0]]
    y_history = [x0[1]]
    theta_history = [x0[2]]
    s_history = [x0[3]]
    v_history = []
    omega_history = []
    
    # Path points for plotting
    s_path = np.linspace(-2, 12, 100)
    x_path = s_path
    y_path = controller.path_slope * s_path + controller.path_intercept
    
    # Simulation loop
    x_current = x0.copy()
    
    print("Starting MPCC simulation...")
    for i in range(n_steps):
        # Solve MPCC
        u_opt, x_pred = controller.solve(x_current)
        
        if x_pred is None:
            print(f"Solver failed at step {i}")
            break
        
        v, omega = u_opt
        
        # Apply control (simulate dynamics)
        x_current[0] += v * np.cos(x_current[2]) * dt
        x_current[1] += v * np.sin(x_current[2]) * dt
        x_current[2] += omega * dt
        # Update path parameter based on projection
        x_current[3] = x_current[0]  # For straight line, s ≈ x
        
        # Store history
        x_history.append(x_current[0])
        y_history.append(x_current[1])
        theta_history.append(x_current[2])
        s_history.append(x_current[3])
        v_history.append(v)
        omega_history.append(omega)
        
        if i % 10 == 0:
            # Calculate contouring error
            a = controller.path_slope
            b = -1.0
            c = controller.path_intercept
            e_contour = abs(a * x_current[0] + b * x_current[1] + c) / np.sqrt(a**2 + b**2)
            print(f"Step {i}/{n_steps}: pos=({x_current[0]:.2f}, {x_current[1]:.2f}), "
                  f"e_contour={e_contour:.3f}, v={v:.2f}, omega={omega:.2f}")
    
    print("Simulation complete!")
    
    # Plotting
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Trajectory
    ax = axes[0, 0]
    ax.plot(x_path, y_path, 'k--', linewidth=2, label='Reference Path')
    ax.plot(x_history, y_history, 'b-', linewidth=2, label='Robot Trajectory')
    ax.plot(x_history[0], y_history[0], 'go', markersize=10, label='Start')
    ax.plot(x_history[-1], y_history[-1], 'ro', markersize=10, label='End')
    
    # Plot robot orientation at intervals
    for i in range(0, len(x_history), 15):
        dx = 0.3 * np.cos(theta_history[i])
        dy = 0.3 * np.sin(theta_history[i])
        ax.arrow(x_history[i], y_history[i], dx, dy, 
                head_width=0.2, head_length=0.15, fc='blue', ec='blue', alpha=0.6)
    
    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Y (m)', fontsize=12)
    ax.set_title('MPCC Trajectory Tracking', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    
    # Plot 2: Contouring Error
    ax = axes[0, 1]
    a = controller.path_slope
    b = -1.0
    c = controller.path_intercept
    errors = [abs(a * x + b * y + c) / np.sqrt(a**2 + b**2) 
              for x, y in zip(x_history, y_history)]
    time = np.arange(len(errors)) * dt
    ax.plot(time, errors, 'r-', linewidth=2)
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Contouring Error (m)', fontsize=12)
    ax.set_title('Tracking Error', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Control Inputs - Linear Velocity
    ax = axes[1, 0]
    time_u = np.arange(len(v_history)) * dt
    ax.plot(time_u, v_history, 'b-', linewidth=2, label='v')
    ax.axhline(y=controller.v_max, color='r', linestyle='--', alpha=0.5, label='Max')
    ax.axhline(y=-controller.v_max, color='r', linestyle='--', alpha=0.5)
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Linear Velocity (m/s)', fontsize=12)
    ax.set_title('Control Input: Linear Velocity', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Control Inputs - Angular Velocity
    ax = axes[1, 1]
    ax.plot(time_u, omega_history, 'g-', linewidth=2, label='ω')
    ax.axhline(y=controller.omega_max, color='r', linestyle='--', alpha=0.5, label='Max')
    ax.axhline(y=-controller.omega_max, color='r', linestyle='--', alpha=0.5)
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Angular Velocity (rad/s)', fontsize=12)
    ax.set_title('Control Input: Angular Velocity', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('mpcc_results.png', dpi=150, bbox_inches='tight')
    print("Results saved to mpcc_results.png")
    plt.show()
    
    # Print final statistics
    final_error = errors[-1]
    mean_error = np.mean(errors[20:])  # After initial transient
    max_error = np.max(errors)
    
    print("\n=== Performance Statistics ===")
    print(f"Final contouring error: {final_error:.4f} m")
    print(f"Mean contouring error (steady-state): {mean_error:.4f} m")
    print(f"Maximum contouring error: {max_error:.4f} m")
    print(f"Final position: ({x_history[-1]:.2f}, {y_history[-1]:.2f})")
    print(f"Distance traveled along path: {s_history[-1]:.2f} m")





if __name__ == "__main__":
    simulate_mpcc()