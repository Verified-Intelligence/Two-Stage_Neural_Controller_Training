from dataclasses import dataclass
import torch
from typing import Any, List
import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import src.loss as loss
import src.dynamics as dynamical_sys
    
@dataclass
class SystemConfig:
    dynamics: Any
    controller: Any
    lyapunov: Any
    factor_list: List[float]
    scale_vector: List[float]
    loss_weights: List[float]
    p: float
    mu: float
    dt: float
    steps: float
    lower_bound: torch.Tensor
    upper_bound: torch.Tensor

configs = {}

# ─── Van der Pol ───────────────────────────────────────────────────────────────
_d = dynamical_sys.VanDerPolDynamics()
configs['van'] = SystemConfig(
    dynamics     = _d,
    controller   = loss.Controller([2, 10, 10, 1],_d.x_equilibrium,_d.u_equilibrium,scale=1.0),
    lyapunov     = loss.Lyapunov([2, 40, 40, 1]),
    factor_list  = [1],
    scale_vector = [1, 1],
    loss_weights = [-0.5, -1.0, -0.0, -1.0, -1.0],
    p = 2.5,
    mu = 0.1,
    dt = 0.01,
    steps = 50,
    lower_bound = torch.full((_d.x_equilibrium.size(0),), float('-inf')),
    upper_bound = torch.full((_d.x_equilibrium.size(0),), float('inf')),
)

# ─── Double Integrator ────────────────────────────────────────────────────────
_d = dynamical_sys.DoubleIntegratorDynamics()
configs['double'] = SystemConfig(
    dynamics      = _d,
    controller    = loss.Controller([2, 10, 10, 1], _d.x_equilibrium, _d.u_equilibrium, scale=1.0),
    lyapunov      = loss.Lyapunov([2, 40, 40, 1]),
    factor_list   = [1],
    scale_vector  = [1, 2],
    loss_weights = [-0.5, -1.0, -0.0, -1.0, -1.0],
    p = 1,
    mu = 0.05,
    dt = 0.01,
    steps = 50,
    lower_bound = torch.full((_d.x_equilibrium.size(0),), float('-inf')),
    upper_bound = torch.full((_d.x_equilibrium.size(0),), float('inf')),
)

# ─── Pendulum ─────────────────────────────────────────────────────────────────
_d = dynamical_sys.PendulumDynamics(m=0.15, l=0.5, beta=0.1)
configs['pendulum_bigtorque'] = SystemConfig(
    dynamics      = _d,
    controller    = loss.Controller([2, 10, 10, 1], _d.x_equilibrium, _d.u_equilibrium, scale=6.0),
    lyapunov      = loss.Lyapunov([2, 40, 40, 1]),
    factor_list   = [1],
    scale_vector  = [1, 2],
    loss_weights  = [-0.5, -1.0, -0.0, -1.0, -1.0],
    p = 1,
    mu = 0.2,
    dt = 0.001,
    steps = 50,
    lower_bound = torch.full((_d.x_equilibrium.size(0),), float('-inf')),
    upper_bound = torch.full((_d.x_equilibrium.size(0),), float('inf')),
)

_d = dynamical_sys.PendulumDynamics(m=0.15, l=0.5, beta=0.1)
configs['pendulum_smalltorque'] = SystemConfig(
    dynamics      = _d,
    controller    = loss.Controller([2, 10, 10, 1], _d.x_equilibrium, _d.u_equilibrium, scale=0.75),
    lyapunov      = loss.Lyapunov([2, 40, 40, 1]),
    factor_list   = [1],
    scale_vector  = [1, 2],
    loss_weights = [-0.5, -1.0, -0.0, -1.0, -1.0],
    p = 1,
    mu = 0.2,
    dt = 0.001,
    steps = 50,
    lower_bound = torch.full((_d.x_equilibrium.size(0),), float('-inf')),
    upper_bound = torch.full((_d.x_equilibrium.size(0),), float('inf')),
)

# ─── Path Tracking ────────────────────────────────────────────────────────────
_d = dynamical_sys.PathTrackingDynamics(speed=2.0, length=1.0, radius=10.0)
configs['path_tracking_bigtorque'] = SystemConfig(
    dynamics      = _d,
    controller    = loss.Controller([2, 10, 10, 1], _d.x_equilibrium, _d.u_equilibrium, scale=0.84),
    lyapunov      = loss.Lyapunov([2, 40, 40, 1]),
    factor_list   = [2],
    scale_vector  = [1, 1],
    loss_weights = [-0.5, -1.0, -0.0, -1.0, -1.0],
    p = 1.5,
    mu = 0.1,
    dt = 0.01,
    steps = 50,
    lower_bound = torch.full((_d.x_equilibrium.size(0),), float('-inf')),
    upper_bound = torch.full((_d.x_equilibrium.size(0),), float('inf')),
)

_d = dynamical_sys.PathTrackingDynamics(speed=2.0, length=1.0, radius=10.0)
configs['path_tracking_smalltorque'] = SystemConfig(
    dynamics      = _d,
    controller    = loss.Controller([2, 10, 10, 1], _d.x_equilibrium, _d.u_equilibrium, scale=0.5),
    lyapunov      = loss.Lyapunov([2, 40, 40, 1]),
    factor_list   = [2],
    scale_vector  = [1, 1],
    loss_weights = [-0.5, -1.0, -0.0, -1.0, -1.0],
    p = 1.5,
    mu = 0.1,
    dt = 0.01,
    steps = 50,
    lower_bound = torch.full((_d.x_equilibrium.size(0),), float('-inf')),
    upper_bound = torch.full((_d.x_equilibrium.size(0),), float('inf')),
)

# ─── Cartpole ─────────────────────────────────────────────────────────────────
_d = dynamical_sys.CartPoleDynamics(mc=1.0, mp=0.1, l=1.0, gravity=9.81)
configs['cartpole'] = SystemConfig(
    dynamics      = _d,
    controller    = loss.Controller([4, 32, 32, 1], _d.x_equilibrium, _d.u_equilibrium, scale=30.0),
    lyapunov      = loss.Lyapunov([4, 128, 128, 1]),
    factor_list = [1.0],
    scale_vector = [2.4, 2.4, 12.0, 12.0],
    loss_weights = [1.5, -1.0, -0.0, -1.0, -1.0],
    p = 1,
    mu = 0.8,
    dt = 0.001,
    steps = 50,
    lower_bound = torch.full((_d.x_equilibrium.size(0),), float('-inf')),
    upper_bound = torch.full((_d.x_equilibrium.size(0),), float('inf')),
)

# ─── Ducted Fan ─────────────────────────────────────────────────────────────
_d = dynamical_sys.DuctedFanDynamics()
configs['ducted_fan'] = SystemConfig(
    dynamics      = _d,
    controller    = loss.MixController(
                        dims=[6, 64, 64, 64, 2],                # example architecture
                        x_equilibrium=_d.x_equilibrium,
                        u_equilibrium=_d.u_equilibrium,
                        scale=10.0,
                        pos_idx=1
                    ),
    lyapunov      = loss.Lyapunov([6, 120, 120, 120, 1]),
    factor_list   = [0.4],
    scale_vector  = [1, 1, 1, 1, 1, 1],
    loss_weights = [1.5, -1.0, -0.0, -1.0, -1.0],
    p = 1,
    mu = 0.2,
    dt = 0.0005,
    steps = 50,
    lower_bound = torch.full((_d.x_equilibrium.size(0),), float('-inf')),
    upper_bound = torch.full((_d.x_equilibrium.size(0),), float('inf')),
)

# ─── 2D Quadrotor ─────────────────────────────────────────────────────────────
_d = dynamical_sys.Quadrotor2DDynamics()
configs['2dquadrotor'] = SystemConfig(
    dynamics      = _d,
    controller    = loss.PosController([6, 60, 60, 60, 2], _d.x_equilibrium, _d.u_equilibrium, scale=6.0),
    lyapunov      = loss.Lyapunov([6, 120, 120, 120, 1]),
    factor_list   = [0.4],
    scale_vector  = [0.75, 0.75, np.pi/2, 4, 4, 3],
    loss_weights = [1.5, -1.0, -0.0, -1.0, -1.0],
    p = 1,
    mu = 0.2,
    dt = 0.0005,
    steps = 50,
    lower_bound = torch.full((_d.x_equilibrium.size(0),), float('-inf')),
    upper_bound = torch.full((_d.x_equilibrium.size(0),), float('inf')),
)

# ─── PVTOL ────────────────────────────────────────────────────────────────────
_d = dynamical_sys.PvtolDynamics()
configs['pvtol'] = SystemConfig(
    dynamics      = _d,
    controller    = loss.PosController([6, 60, 60, 60, 2], _d.x_equilibrium, _d.u_equilibrium, scale=39.2),
    lyapunov      = loss.Lyapunov([6, 120, 120, 120, 1]),
    factor_list   = [0.4],
    scale_vector  = [1, 1, 1, 1, 1, 1],
    loss_weights = [1.5, -1.0, -0.0, -1.0, -1.0],
    p = 1,
    mu = 0.8,
    dt = 0.0005,
    steps = 50,
    lower_bound = torch.full((_d.x_equilibrium.size(0),), float('-inf')),
    upper_bound = torch.full((_d.x_equilibrium.size(0),), float('inf')),
)

# ─── 3D Quadrotor ─────────────────────────────────────────────────────────────
_d = dynamical_sys.Quadrotor3DDynamics()
configs['3dquadrotor'] = SystemConfig(
    dynamics      = _d,
    controller    = loss.PosController([12, 80, 80, 80, 4], _d.x_equilibrium, _d.u_equilibrium, scale=3.6),
    lyapunov      = loss.Lyapunov([12, 256, 256, 256, 1]),
    factor_list   = [0.4],
    scale_vector = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    loss_weights = [1.5, -1.0, -0.0, -1.0, -1.0],
    p = 1,
    mu = 0.2,
    dt = 0.0005,
    steps = 50,
    lower_bound = -torch.tensor([5.0, 5.0, 5.0, 1.2, 1.2, np.pi,7.2, 7.2, 7.2,10.8, 10.8, 10.8]),
    upper_bound = torch.tensor([5.0, 5.0, 5.0, 1.2, 1.2, np.pi,7.2, 7.2, 7.2,10.8, 10.8, 10.8])
)