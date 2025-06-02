import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import scipy

def compute_jacobian(y, x):
    """
    Computes the Jacobian of y with respect to x.
    
    Args:
        y: Tensor of shape [B, out_dim]
        x: Tensor of shape [B, in_dim] (requires_grad should be True)
        
    Returns:
        jacobian: Tensor of shape [B, out_dim, in_dim]
    """
    batch_size, out_dim = y.shape
    in_dim = x.shape[1]
    jac_list = []
    for i in range(out_dim):
        # Compute gradient of the i-th output component (summed over the batch)
        grad_y = torch.autograd.grad(
            y[:, i].sum(), x, retain_graph=True, create_graph=True
        )[0]  # shape: [B, in_dim]
        jac_list.append(grad_y.unsqueeze(1))  # shape: [B, 1, in_dim]
    jacobian = torch.cat(jac_list, dim=1)  # shape: [B, out_dim, in_dim]
    return jacobian

class DoubleIntegratorDynamics:
    """
    Double integrator system with dynamics:
    ẋ₁ = x₂
    ẋ₂ = u
    """
    
    def __init__(self):
        self.nx = 2  # State dimension
        self.nu = 1  # Input dimension
        self.ny = 1  # Output dimension
        
    def forward(self, x, u):
        """
        Dynamics. x: state (batch, 2); u: controller input (batch, 1).
        Returns only ẋ₂ since ẋ₁ is just x₂.
        """
        # Second state derivative is simply the input: ẋ₂ = u
        return u
    
    def f_torch(self, x, u):
        """
        Complete dynamics. x: state (batch, 2); u: controller input (batch, 1).
        Returns both state derivatives.
        """
        x2 = x[:, 1].unsqueeze(-1)
        d_x1 = x2
        d_x2 = u
        
        return torch.cat((d_x1, d_x2), dim=1)
    
    def linearized_dynamics(self, x, u):
        """
        Computes the Jacobians A = ∂f/∂x and B = ∂f/∂u using autograd.
        """
        # Make sure x and u require gradients.
        x = x.clone().detach().requires_grad_(True)
        u = u.clone().detach().requires_grad_(True)
        f = self.f_torch(x, u)  # [B, nx]
        A = compute_jacobian(f, x)  # [B, nx, nx]
        B = compute_jacobian(f, u)  # [B, nx, nu]
        return A, B
    
    @property
    def x_equilibrium(self):
        """
        Returns the equilibrium state (origin).
        """
        return torch.zeros((2,))
    
    @property
    def u_equilibrium(self):
        """
        Returns the equilibrium input.
        """
        return torch.zeros((1,))
    
class PendulumDynamics:
    """
    The inverted pendulum, with the upright equilibrium as the state origin.
    """

    def __init__(self, m: float = 1, l: float = 1, beta: float = 1, g: float = 9.81):
        self.nx = 2
        self.nu = 1
        self.ny = 1
        self.m = m  # Mass
        self.l = l  # Length
        self.g = g  # Gravity
        self.beta = beta  # Damping
        self.inertia = self.m * self.l**2

    def forward(self, x, u):
        """
        Dynamics. x: state (batch, 2); u: controller input (batch, 1).
        """
        # States (theta, thete_dot)
        theta, theta_dot = x[:, 0], x[:, 1]
        theta = theta.unsqueeze(-1)
        theta_dot = theta_dot.unsqueeze(-1)
        # Dynamics according to http://underactuated.mit.edu/pend.html
        ml2 = self.m * self.l * self.l
        d_theta = theta_dot
        d_theta_dot = (
            (-self.beta / ml2) * theta_dot
            + (self.g / self.l) * torch.sin(theta)
            + u / ml2
        )
        return d_theta_dot
    
    def f_torch(self, x, u):
        """
        Dynamics. x: state (batch, 2); u: controller input (batch, 1).
        """
        # States (theta, thete_dot)
        theta, theta_dot = x[:, 0], x[:, 1]
        theta = theta.unsqueeze(-1)
        theta_dot = theta_dot.unsqueeze(-1)
        # Dynamics according to http://underactuated.mit.edu/pend.html
        ml2 = self.m * self.l * self.l
        d_theta = theta_dot
        d_theta_dot = (
            (-self.beta / ml2) * theta_dot
            + (self.g / self.l) * torch.sin(theta)
            + u / ml2
        )
        return torch.cat((d_theta, d_theta_dot),dim=1)

    def linearized_dynamics(self, x, u):
        device = x.device
        batch_size = x.shape[0]
        A = torch.zeros((batch_size, self.nx, self.nx))
        B = torch.zeros((batch_size, self.nx, self.nu))
        A[:, 0, 1] = 1
        A[:, 1, 0] = self.g / self.l * torch.sin(x[:, 0])
        A[:, 1, 1] = -self.beta / (self.inertia)
        B[:, 1, 0] = 1 / self.inertia
        return A.to(device), B.to(device)

    def h(self, x):
        return x[:, : self.ny]

    def linearized_observation(self, x):
        batch_size = x.shape[0]
        C = torch.zeros(batch_size, self.ny, self.nx, device=x.device)
        C[:, 0] = 1
        return C

    @property
    def x_equilibrium(self):
        return torch.zeros((2,))

    @property
    def u_equilibrium(self):
        return torch.zeros((1,))
    
class PathTrackingDynamics:
    def __init__(self, speed: float, length: float, radius: float):
        self.nx = 2
        self.nu = 1
        self.speed = speed
        self.length = length
        self.radius = radius

    def forward(self, x: torch.Tensor, u: torch.Tensor):
        """
        x: size is (batch, 2)
        u: size is (batch, 1)
        """

        theta_e = x[:, 1:2]
        sintheta_e = torch.sin(theta_e)
        costheta_e = torch.cos(theta_e)

        d_e_acc = self.speed * sintheta_e
        coef = self.radius / self.speed
        theta_e_acc = (self.speed * u / self.length) - (
            costheta_e / (coef - sintheta_e)
        )
        return torch.cat((d_e_acc, theta_e_acc), dim=1)
    
    def f_torch(self, x: torch.Tensor, u: torch.Tensor):
        """
        x: size is (batch, 2)
        u: size is (batch, 1)
        """

        theta_e = x[:, 1:2]
        sintheta_e = torch.sin(theta_e)
        costheta_e = torch.cos(theta_e)

        d_e_acc = self.speed * sintheta_e
        coef = self.radius / self.speed
        theta_e_acc = (self.speed * u / self.length) - (
            costheta_e / (coef - sintheta_e)
        )
        return torch.cat((d_e_acc, theta_e_acc), dim=1)

    def linearized_dynamics(self, x, u):
        device = x.device
        batch_size = x.shape[0]
        A = torch.zeros((batch_size, self.nx, self.nx), device=device)
        B = torch.zeros((batch_size, self.nx, self.nu), device=device)
        theta_e = x[:, 1:2]
        sintheta_e = torch.sin(theta_e)
        costheta_e = torch.cos(theta_e)
        coef = self.radius / self.speed
        A[:, 0, 1] = self.speed * costheta_e
        A[:, 1, 1] = -(sintheta_e * coef + 1) / ((coef - sintheta_e) ** 2)
        B[:, 1, 0] = self.speed / self.length
        return A.to(device), B.to(device)

    @property
    def x_equilibrium(self):
        return torch.zeros((2,))

    @property
    def u_equilibrium(self):
        return torch.tensor([self.length / self.radius])
    
class VanDerPolDynamics:
    """
    The Van der Pol oscillator system with control input.
    System equations:
    ẋ₁ = x₂
    ẋ₂ = x₁ - μ(1 - x₁²)x₂ + u
    """
    
    def __init__(self, mu: float = 1.0):
        self.nx = 2  # State dimension
        self.nu = 1  # Input dimension
        self.ny = 1  # Output dimension
        self.mu = mu  # System parameter
    
    def f_torch(self, x, u):
        """
        Complete dynamics. x: state (batch, 2); u: controller input (batch, 1).
        Returns both state derivatives.
        """
        x1, x2 = x[:, 0], x[:, 1]
        x1 = x1.unsqueeze(-1)
        x2 = x2.unsqueeze(-1)
        
        # First state derivative: ẋ₁ = x₂
        d_x1 = x2
        
        # Second state derivative: ẋ₂ = -x₁ + μ(1 - x₁²)x₂ + u
        d_x2 = -x1 + self.mu * (1 - x1**2) * x2 + u
        
        return torch.cat((d_x1, d_x2), dim=1)
    
    def linearized_dynamics(self, x, u):
        # Use autograd to compute Jacobians.
        x = x.clone().detach().requires_grad_(True)
        u = u.clone().detach().requires_grad_(True)
        f = self.f_torch(x, u)
        A = compute_jacobian(f, x)
        B = compute_jacobian(f, u)
        return A, B
    
    @property
    def x_equilibrium(self):
        """
        Returns the equilibrium state (origin).
        """
        return torch.zeros((2,))
    
    @property
    def u_equilibrium(self):
        """
        Returns the equilibrium input.
        """
        return torch.zeros((1,))
    
class CartPoleDynamics():
    """
    The dynamics of a cart-pole with state x = [px, θ, px_dot, θdot]
    """

    def __init__(self, mc=1.0, mp=0.1, l=1.0, gravity=9.81, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mc = mc
        self.mp = mp
        self.l = l
        self.gravity = gravity
        
        # Initialize a flag to track if we've compiled the dynamics
        self._compiled = False

    def forward(self, x, u):
        """
        Refer to https://underactuated.mit.edu/acrobot.html#cart_pole
        """
        px_dot = x[:, 2]
        theta_dot = x[:, 3]
        s = torch.sin(x[:, 1])
        c = torch.cos(x[:, 1])
        px_ddot = (
            u.squeeze(1) + self.mp * s * (self.l * theta_dot**2 - self.gravity * c)
        ) / (self.mp * s**2 + self.mc)
        theta_ddot = (
            -u.squeeze(1) * c
            - self.mp * self.l * theta_dot**2 * c * s
            + (self.mc + self.mp) * self.gravity * s
        ) / (self.l * (self.mc + self.mp * s**2))
        return torch.cat(
            (
                px_dot.unsqueeze(1),
                theta_dot.unsqueeze(1),
                px_ddot.unsqueeze(1),
                theta_ddot.unsqueeze(1),
            ),
            dim=1,
        )
        
    def f_torch(self, x, u):
        return self.forward(x, u)
    
    def linearized_dynamics(self, x, u):
        # Use autograd to compute Jacobians.
        x = x.clone().detach().requires_grad_(True)
        u = u.clone().detach().requires_grad_(True)
        f = self.f_torch(x, u)
        A = compute_jacobian(f, x)
        B = compute_jacobian(f, u)
        return A, B
    
    @property
    def x_equilibrium(self):
        return torch.zeros((4,))

    @property
    def u_equilibrium(self):
        return torch.zeros((1,))
    
class Quadrotor2DDynamics:
    """
    2D Quadrotor dynamics, based on https://github.com/StanfordASL/neural-network-lyapunov/blob/master/neural_network_lyapunov/examples/quadrotor2d/quadrotor_2d.py
    """

    def __init__(
        self, length=0.25, mass=0.486, inertia=0.00383, gravity=9.81, *args, **kwargs
    ):
        self.nx = 6
        self.nu = 2
        # length of the rotor arm.
        self.length = length
        # mass of the quadrotor.
        self.mass = mass
        # moment of inertia
        self.inertia = inertia
        # gravity.
        self.gravity = gravity

    def forward(self, x, u):
        """
        Compute the continuous-time dynamics (batched, pytorch).
        This is the actual computation that will be bounded using auto_LiRPA.
        """
        q = x[:, :3]
        qdot = x[:, 3:]
        qddot1 = (-1.0 / self.mass) * (torch.sin(q[:, 2:]) * (u[:, :1] + u[:, 1:]))
        qddot2 = (1.0 / self.mass) * (
            torch.cos(q[:, 2:]) * (u[:, :1] + u[:, 1:])
        ) - self.gravity
        qddot3 = (self.length / self.inertia) * (u[:, :1] - u[:, 1:])
        return torch.cat((qddot1, qddot2, qddot3), dim=1)
    
    def f_torch(self, x, u):
        """
        Compute the full state derivatives for the quadrotor system.
        Args:
            x: state tensor (batch, 6) [x, y, theta, x_dot, y_dot, theta_dot]
            u: control input tensor (batch, 2)
        Returns:
            Full state derivative tensor (batch, 6)
        """
        # First half is velocity (q_dot)
        q_dot = x[:, 3:]
        
        # Second half is acceleration (q_ddot) from the forward method
        q = x[:, :3]
        qddot1 = (-1.0 / self.mass) * (torch.sin(q[:, 2:]) * (u[:, :1] + u[:, 1:]))
        qddot2 = (1.0 / self.mass) * (torch.cos(q[:, 2:]) * (u[:, :1] + u[:, 1:])) - self.gravity
        qddot3 = (self.length / self.inertia) * (u[:, :1] - u[:, 1:])
        q_ddot = torch.cat((qddot1, qddot2, qddot3), dim=1)

        # Combine into full state derivative
        return torch.cat((q_dot, q_ddot), dim=1)

    def f1(self, x):
        f1_tensor = torch.zeros(x.shape[0], self.nx, device=x.device)
        f1_tensor[:, :3] = x[:, 3:]
        f1_tensor[:, 4] = -self.mass * self.gravity
        return f1_tensor

    def f2(self, x):
        q = x[:, :3]
        f2_tensor = torch.zeros(x.shape[0], self.nx, self.nu, device=x.device)
        f2_tensor[:, 3, :] = (-1.0 / self.mass) * torch.sin(q[:, 2:])
        f2_tensor[:, 4, :] = (1.0 / self.mass) * torch.cos(q[:, 2:])
        f2_tensor[:, 5, 0] = self.length / self.inertia
        f2_tensor[:, 5, 1] = -self.length / self.inertia
        return f2_tensor

    def linearized_dynamics(self, x, u):
        """
        Return ∂ẋ/∂x and ∂ẋ/∂ u
        """
        if isinstance(x, np.ndarray):
            A = np.zeros((6, 6))
            B = np.zeros((6, 2))
            A[:3, 3:6] = np.eye(3)
            theta = x[2]
            A[3, 2] = -np.cos(theta) / self.mass * (u[0] + u[1])
            A[4, 2] = -np.sin(theta) / self.mass * (u[0] + u[1])
            B[3, 0] = -np.sin(theta) / self.mass
            B[3, 1] = B[3, 0]
            B[4, 0] = np.cos(theta) / self.mass
            B[4, 1] = B[4, 0]
            B[5, 0] = self.length / self.inertia
            B[5, 1] = -B[5, 0]
            return A, B
        elif isinstance(x, torch.Tensor):
            dtype = x.dtype
            A = torch.zeros((6, 6), dtype=dtype)
            B = torch.zeros((6, 2), dtype=dtype)
            A[:3, 3:6] = torch.eye(3, dtype=dtype)
            theta = x[2]
            A[3, 2] = -torch.cos(theta) / self.mass * (u[0] + u[1])
            A[4, 2] = -torch.sin(theta) / self.mass * (u[0] + u[1])
            B[3, 0] = -torch.sin(theta) / self.mass
            B[3, 1] = B[3, 0]
            B[4, 0] = torch.cos(theta) / self.mass
            B[4, 1] = B[4, 0]
            B[5, 0] = self.length / self.inertia
            B[5, 1] = -B[5, 0]
            return A, B

    def lqr_control(self, Q, R, x, u):
        """
        The control action should be u = K * (x - x*) + u*
        """
        x_np = x if isinstance(x, np.ndarray) else x.detach().numpy()
        u_np = u if isinstance(u, np.ndarray) else u.detach().numpy()
        A, B = self.linearized_dynamics(x_np, u_np)
        S = scipy.linalg.solve_continuous_are(A, B, Q, R)
        K = -np.linalg.solve(R, B.T @ S)
        return K, S

    @property
    def x_equilibrium(self):
        return torch.zeros((6,))

    @property
    def u_equilibrium(self):
        return torch.full((2,), (self.mass * self.gravity) / 2)

        
class PvtolDynamics(nn.Module):
    def __init__(
        self,
        length=0.25,
        mass=4.0,
        inertia=0.0475,
        gravity=9.8,
        dist=0.25,
        dt=0.05,
        *args,
        **kwargs
    ):
        super().__init__()
        self.nx = 6
        self.nq = 3
        self.nu = 2
        # length of the rotor arm.
        self.length = length
        # mass of the quadrotor.
        self.mass = mass
        # moment of inertia
        self.inertia = inertia
        # gravity.
        self.gravity = gravity
        self.dist = dist
        self.dt = dt

    def forward(self, state, u):
        """
        Compute the continuous-time dynamics (batched, pytorch).
        This is the actual computation that will be bounded using auto_LiRPA.
        """

        x = state[:, 0:1]
        y = state[:, 1:2]
        theta = state[:, 2:3]
        x_d = state[:, 3:4]
        y_d = state[:, 4:5]
        theta_d = state[:, 5:6]

        u_1 = u[:, 0:1]
        u_2 = u[:, 1:2]

        sin_theta = torch.sin(theta)
        cos_theta = torch.cos(theta)
        x_change = x_d * cos_theta - y_d * sin_theta
        y_change = x_d * sin_theta + y_d * cos_theta
        theta_change = theta_d
        x_d_change = y_d * theta_d - self.gravity * sin_theta
        y_d_change = -x_d * theta_d - self.gravity * cos_theta + (u_1 + u_2) / self.mass
        theta_d_change = (u_1 - u_2) * self.dist / self.inertia

        state_next = torch.concat(
            [x_change, y_change, theta_change, x_d_change, y_d_change, theta_d_change], dim=-1
        )

        return state_next
    
    def f_torch(self, x, u):
        return self.forward(x, u)

    def linearized_dynamics(self, x, u):
        # Appendix A.2 in "Neural Lyapunov Control for Discrete-Time Systems"
        A = np.zeros((6, 6))
        B = np.zeros((6, 2))
        A[0, 3] = A[1, 4] = A[2, 5] = 1
        A[3, 2] = -self.gravity
        B[4, :] = 1.0 / self.mass
        B[5, 0] = self.length / self.inertia
        B[5, 1] = -B[5, 0]
        return A, B

    def lqr_control(self, Q, R, x, u):
        """
        The control action should be u = K * (x - x*) + u*
        """
        x_np = x if isinstance(x, np.ndarray) else x.detach().numpy()
        u_np = u if isinstance(u, np.ndarray) else u.detach().numpy()
        A, B = self.linearized_dynamics(x_np, u_np)
        S = scipy.linalg.solve_continuous_are(A, B, Q, R)
        K = -np.linalg.solve(R, B.T @ S)
        return K, S

    def h(self, x):
        return x[:, :0]

    @property
    def x_equilibrium(self):
        return torch.zeros((6,))

    @property
    def u_equilibrium(self):
        return torch.full((2,), self.mass * self.gravity / 2)
    
class DuctedFanDynamics(nn.Module):
    def __init__(self, g=0.28, m=11.2, I=0.0462, r=0.156, d=0.1):
        super().__init__()
        self.nx = 6
        self.nu = 2
        self.g  = g
        self.m  = m
        self.I  = I
        self.r  = r
        self.d  = d

    def f_torch(self, x, u):
        """
        x: [B,6] = [x, y, θ, ẋ, ẏ, θ̇]
        u: [B,2] = [u₀, u₁]
        returns [B,6] = [ẋ, ẏ, θ̇, ẍ, ÿ, θ̈]
        """
        # split state
        q    = x[:, :3]    # [x, y, θ]
        qdot = x[:, 3:]    # [ẋ, ẏ, θ̇]

        θ = q[:, 2:3]
        # accelerations
        qddot_x     = (
            -self.d * qdot[:, 0:1]
            + u[:, 0:1] * torch.cos(θ)
            - u[:, 1:2] * torch.sin(θ)
        ) / self.m

        qddot_y     = (
            -self.d * qdot[:, 1:2]
            + u[:, 0:1] * torch.sin(θ)
            + u[:, 1:2] * torch.cos(θ)
        ) / self.m - self.g

        qddot_theta = (self.r * u[:, 0:1]) / self.I
        qddot = torch.cat((qddot_x, qddot_y, qddot_theta), dim=1)

        return torch.cat((qdot, qddot), dim=1)

    @property
    def x_equilibrium(self):
        return torch.zeros((6,))

    @property
    def u_equilibrium(self):
        return torch.tensor([0.0, self.m * self.g])
    
class Quadrotor3DDynamics:
    """
    3D Quadrotor dynamics, based on https://raw.githubusercontent.com/StanfordASL/neural-network-lyapunov/master/neural_network_lyapunov/examples/quadrotor3d/quadrotor.py

    Reference:
    https://github.com/huanzhang12/neural_lyapunov_training/blob/32665a946a6edae25be0a819fba07d011df6f9c7/quadrotor3d_training.py
    https://github.com/huanzhang12/neural_lyapunov_training/blob/32665a946a6edae25be0a819fba07d011df6f9c7/models.py

    A quadrotor that directly commands the thrusts.
    The state is [pos_x, pos_y, pos_z, roll, pitch, yaw, pos_xdot, pos_ydot,
    pos_zdot, angular_vel_x, angular_vel_y, angular_vel_z], where
    angular_vel_x/y/z are the angular velocity measured in the body frame.
    Notice that unlike many models where uses the linear velocity in the body
    frame as the state, we use the linear velocit in the world frame as the
    state. The reason is that the update from linear velocity to next position
    is a linear constraint, and we don't need to use a neural network to encode
    this update.
    """

    def __init__(
        self, length=0.225, mass=0.486, gravity=9.81,
        z_torque_to_force_factor=1.1 / 29, version='default',
        *args, **kwargs
    ):
        self.version = version
        self.nx = 12
        self.nq = 0
        self.nu = 4
        self.arm_length = 0.225
        self.mass = 0.486
        self.gravity = 9.81
        self.z_torque_to_force_factor = 1.1 / 29
        self.hover_thrust = self.mass * self.gravity / 4
        self.inertia = torch.tensor([4.9E-3, 4.9E-3, 8.8E-3])
        self.plant_input_w = torch.tensor(
            [
                [1, 1, 1, 1],
                [0, self.arm_length, 0, -self.arm_length],
                [-self.arm_length, 0, self.arm_length, 0],
                [
                    self.z_torque_to_force_factor,
                    -self.z_torque_to_force_factor,
                    self.z_torque_to_force_factor,
                    -self.z_torque_to_force_factor
                ]
            ]
        )
        self.pos_ddot_bias = torch.tensor([0, 0, -self.gravity])
        self.x_dim = 12

    def forward(self, x, u):
        self.inertia = self.inertia.to(x)
        self.plant_input_w = self.plant_input_w.to(x)
        self.pos_ddot_bias = self.pos_ddot_bias.to(x)

        rpy = x[:, 3:6]
        pos_dot = x[:, 6:9]
        omega = x[:, 9:12]

        if self.version == 'default':
            cos_rpy = torch.cos(rpy)
            sin_rpy = torch.sin(rpy)
            cos_roll = cos_rpy[:, 0:1]
            sin_roll = sin_rpy[:, 0:1]
            cos_pitch = cos_rpy[:, 1:2]
            sin_pitch = sin_rpy[:, 1:2]
            tan_pitch = torch.tan(rpy[:, 1:2])
            cos_yaw = cos_rpy[:, 2:3]
            sin_yaw = sin_rpy[:, 2:3]

            R_last_col = torch.concat([
                sin_yaw * sin_roll + cos_yaw * sin_pitch * cos_roll,
                -cos_yaw * sin_roll + sin_yaw * sin_pitch * cos_roll,
                cos_pitch * cos_roll,
            ], dim=-1)

            omega_0 = omega[:, 0:1]
            omega_1 = omega[:, 1:2]
            omega_2 = omega[:, 2:3]
            sin_cos_roll_omega = sin_roll * omega_1 + cos_roll * omega_2
            rpy_dot_0 = omega_0 + tan_pitch * sin_cos_roll_omega
            rpy_dot_1 = cos_roll * omega_1 - sin_roll * omega_2
            rpy_dot_2 = sin_cos_roll_omega / cos_pitch
        elif self.version == '0915-v1':
            # https://zh.wikipedia.org/wiki/%E4%B8%89%E8%A7%92%E6%81%92%E7%AD%89%E5%BC%8F
            # sin(a)sin(b)=1/2 (cos(a-b) - cos(a+b))
            # cos(a)cos(b)=1/2 (cos(a-b) + cos(a+b))
            # sin(a)cos(b)=1/2 (sin(a+b) + sin(a-b))

            cos_rpy = torch.cos(rpy)
            sin_rpy = torch.sin(rpy)
            cos_roll = cos_rpy[:, 0:1]
            sin_roll = sin_rpy[:, 0:1]
            cos_pitch = cos_rpy[:, 1:2]
            sin_pitch = sin_rpy[:, 1:2]
            tan_pitch = torch.tan(rpy[:, 1:2])
            cos_yaw = cos_rpy[:, 2:3]
            sin_yaw = sin_rpy[:, 2:3]

            roll = rpy[:, 0:1]
            pitch = rpy[:, 1:2]
            yaw = rpy[:, 2:3]

            yaw_minus_roll = yaw - roll
            cos_yaw_minus_roll = torch.cos(yaw_minus_roll)
            sin_yaw_minus_roll = torch.sin(yaw_minus_roll)

            yaw_plus_roll = yaw + roll
            cos_yaw_plus_roll = torch.cos(yaw_plus_roll)
            sin_yaw_plus_roll = torch.sin(yaw_plus_roll)

            pitch_minus_roll = pitch - roll
            cos_pitch_minus_roll = torch.cos(pitch_minus_roll)

            pitch_plus_roll = pitch + roll
            cos_pitch_plus_roll = torch.cos(pitch_plus_roll)

            R_last_col = torch.concat([
                # sin_yaw * sin_roll
                0.5 * (cos_yaw_minus_roll - cos_yaw_plus_roll)
                # cos_yaw * sin_pitch * cos_roll
                + 0.5 * (cos_yaw_minus_roll + cos_yaw_plus_roll) * sin_pitch,

                # sin(a)cos(b)=1/2 (sin(a+b) + sin(a-b))
                # -sin_roll * cos_yaw
                -0.5 * (sin_yaw_plus_roll - sin_yaw_minus_roll)
                # sin_yaw * cos_roll * sin_pitch
                + 0.5 * (sin_yaw_plus_roll + sin_yaw_minus_roll) * sin_pitch,

                # cos_pitch * cos_roll
                0.5 * (cos_pitch_minus_roll + cos_pitch_plus_roll),
            ], dim=-1)

            omega_0 = omega[:, 0:1]
            omega_1 = omega[:, 1:2]
            omega_2 = omega[:, 2:3]
            sin_cos_roll_omega = sin_roll * omega_1 + cos_roll * omega_2
            rpy_dot_0 = omega_0 + tan_pitch * sin_cos_roll_omega
            rpy_dot_1 = cos_roll * omega_1 - sin_roll * omega_2
            rpy_dot_2 = sin_cos_roll_omega / cos_pitch
        else:
            raise NotImplementedError

        # plant_input is [total_thrust, torque_x, torque_y, torque_z]
        plant_input = u.matmul(self.plant_input_w.t())
        pos_ddot = (self.pos_ddot_bias
                    + R_last_col * plant_input[:, 0:1] / self.mass)
        # Here we exploit the fact that the inertia matrix is diagonal.
        omega_dot = (torch.linalg.cross(-omega, self.inertia * omega) +
                     plant_input[:, 1:]) / self.inertia

        ret = torch.cat([
            pos_dot,
            rpy_dot_0, rpy_dot_1, rpy_dot_2,
            pos_ddot,
            omega_dot
        ], dim=-1)

        return ret
    
    def f_torch(self, x, u):
        return self.forward(x,u)

    @property
    def x_equilibrium(self):
        return torch.zeros((12,))

    @property
    def u_equilibrium(self):
        return torch.full((4,), self.hover_thrust)
