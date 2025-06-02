import torch.nn.functional as F
from torch import nn
import torch
import os
import sys
import time
import torch._dynamo
torch._dynamo.config.suppress_errors = True
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
sys.path.append("../torchdyn")
from auto_LiRPA.jacobian import JacobianOP
from training_config import configs


class Verification_Loss(nn.Module):
    def __init__(self, dynamics, controller, lyapunov):
        super().__init__()
        self.controller = controller
        self.dynamics = dynamics
        self.lyapunov = lyapunov
    
    def forward(self, x):
        V_x = self.lyapunov(x)
        u = self.controller(x)
        x_dot = self.dynamics.f_torch(x, u)
        dVdx = JacobianOP.apply(self.lyapunov(x), x).squeeze(1)
        V_dot = torch.sum(dVdx * x_dot, dim=1, keepdim=True)
        return torch.cat((V_x, V_dot), dim=1)


def create_verification_loss(dynamics_name):
    if dynamics_name not in configs:
        raise ValueError(f"Unknown dynamics name: {dynamics_name}")
    
    config = configs[dynamics_name]
    dynamics = config.dynamics
    controller = config.controller
    lyapunov = config.lyapunov
    
    return Verification_Loss(dynamics, controller, lyapunov)