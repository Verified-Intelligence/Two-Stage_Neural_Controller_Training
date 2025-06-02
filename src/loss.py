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

class Lyapunov(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.dims = dims
        layers = []
        for i in range(len(dims)-2):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            layers.append(nn.Tanh())
        
        layers.append(nn.Linear(dims[-2], dims[-1]))
        layers.append(nn.Sigmoid())
        self.layers = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.layers(x)
    
class Controller(nn.Module):
    def __init__(self, dims, x_equilibrium, u_equilibrium, scale=1.0):
        """
        Range [-scale, scale]
        """
        super(Controller, self).__init__()
        self.dims = dims
        self.scale = scale
        layers = []
        for i in range(len(dims) - 2):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            layers.append(nn.Tanh())
            
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.layers = nn.Sequential(*layers)
        self.register_buffer('x_equilibrium', x_equilibrium)
        self.register_buffer('u_equilibrium', u_equilibrium)
        self.register_buffer('b', torch.atanh(self.u_equilibrium / self.scale))

    def forward(self, x):
        u_x = self.layers(x)
        x_eq = self.x_equilibrium.unsqueeze(0)  
        u_x_star = self.layers(x_eq)[0]
        u_diff = u_x - u_x_star
        return self.scale * torch.tanh(u_diff + self.b)
    
class PosController(nn.Module):
    def __init__(self, dims, x_equilibrium, u_equilibrium, scale=1.0):
        """
        Range [0,scale]
        """
        super(PosController, self).__init__()
        self.dims = dims
        layers = []
        for i in range(len(dims) - 2):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.layers = nn.Sequential(*layers)
        self.scale = scale
        self.register_buffer('x_equilibrium', x_equilibrium)
        self.register_buffer('u_equilibrium', u_equilibrium)
        self.register_buffer('b', torch.atanh(self.u_equilibrium / self.scale))

    def forward(self, x):
        u_x = self.layers(x)
        x_eq = self.x_equilibrium.unsqueeze(0)  # reshape if necessary
        u_x_star = self.layers(x_eq)
        u_diff = u_x - u_x_star[0]
        return self.scale * torch.relu(torch.tanh(u_diff + self.b))
    
class MixController(nn.Module):
    def __init__(self, dims, x_equilibrium, u_equilibrium, scale=1.0, pos_idx=None):
        """
        At pos_idx, range is [0,scale]. Otherwise range is [-scale, scale]
        """
        super(MixController, self).__init__()
        self.dims = dims
        layers = []
        for i in range(len(dims) - 2):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.layers = nn.Sequential(*layers)
        self.scale = scale
        self.pos_idx = pos_idx
        self.register_buffer('x_equilibrium', x_equilibrium)
        self.register_buffer('u_equilibrium', u_equilibrium)
        self.register_buffer('b', torch.atanh(self.u_equilibrium / self.scale))

    def forward(self, x):
        u_x = self.layers(x)
        x_eq = self.x_equilibrium.unsqueeze(0)  
        u_x_star = self.layers(x_eq)
        u_diff = u_x - u_x_star[0]
        control = self.scale * torch.tanh(u_diff + self.b)
        if self.pos_idx:                              
            control[:, self.pos_idx] = torch.relu(control[:, self.pos_idx])
        return control


class loss_return:
    def __init__(self, data_loss, pde_loss, controller_loss, v_zero, bdry_loss, loss, num_cex):
        self.data_loss = data_loss
        self.pde_loss = pde_loss
        self.controller_loss = controller_loss
        self.zero = v_zero
        self.bdry_loss = bdry_loss
        self.loss = loss
        self.num_cex = num_cex

class V_Train_Loss(nn.Module):
    def __init__(
        self,
        dynamics,
        controller,
        lyap_nn,
        mu,
        loss_weights,
        upper_limit,
        p,
        mode='learnable',
        use_data_loss=True,
        rollout_steps=50,
        dt=0.01,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.dynamics = dynamics
        self.controller = controller
        self.lyapunov = lyap_nn
        self.mu = mu
        self.loss_weights = loss_weights
        self.upper_limit = upper_limit
        self.p = p
        self.mode = mode
        self.use_data_loss = use_data_loss
        self.rollout_steps = rollout_steps
        self.dt = dt
        self.time = 0 
        
        if self.mode == 'learnable':
            self.log_sigma_pde = nn.Parameter(torch.tensor(loss_weights[0], dtype=torch.float32))
            self.log_sigma_controller = nn.Parameter(torch.tensor(loss_weights[1], dtype=torch.float32))
            self.log_sigma_data = nn.Parameter(torch.tensor(loss_weights[2], dtype=torch.float32))
            self.log_sigma_zero = nn.Parameter(torch.tensor(loss_weights[3], dtype=torch.float32))
            self.log_sigma_bdry = nn.Parameter(torch.tensor(loss_weights[4], dtype=torch.float32))

    def compute_td_target(self, x_batch):
        if not hasattr(self, '_compiled_euler_step'):
            def euler_step(x):
                x = x.clone()
                
                dt = self.dt
                u1 = self.controller(x)
                k1 = self.dynamics.f_torch(x, u1)
                return (x + dt * k1).clone()
                        
            # Try to compile the function, with fallback if torch.compile isn't available
            self._compiled_euler_step = torch.compile(
                euler_step, 
                mode='reduce-overhead'
            )
            print("Using compiled euler step function")
        
        with torch.no_grad():
            x_current = x_batch.clone()
            equilibrium = self.dynamics.x_equilibrium.to(x_current.device)
            integral = torch.zeros(x_batch.size(0), 1, device=x_batch.device)

            for step in range(self.rollout_steps):                
                x_next = self._compiled_euler_step(x_current)
                x_next = x_next.detach().clone()
                integral += torch.linalg.norm((x_current - equilibrium), dim=1, keepdim=True).pow(self.p) * self.dt
                x_current = x_next

            epsilon = 1e-8
            V_end = self.lyapunov(x_current)
            V_end_safe = torch.clamp(V_end, 0, 1 - epsilon)
            tail_estimate = torch.atanh(V_end_safe)
            return torch.tanh(self.mu * integral + tail_estimate)

    def forward(self, x, bdry_points):  
        zero_tensor = self.dynamics.x_equilibrium.unsqueeze(0).to(x.device)
        zero_tensor.requires_grad = True
        V_zero = self.lyapunov(zero_tensor)
        
        x_batch = x.clone().detach().requires_grad_(True)
        
        ## Data Loss
        start_time = time.time()
        if self.use_data_loss:
            td_target = self.compute_td_target(x_batch)
            V_pred = self.lyapunov(x_batch)
            data_loss = F.mse_loss(V_pred, td_target)
        else:
            data_loss = torch.tensor(0.0).to(x.device)
        end_time = time.time()
        self.time = end_time - start_time
        
        # Lyapunov function on x_batch
        W_vals = self.lyapunov(x_batch)  # shape: [B, 1]
        sum_W = W_vals.sum()
        gradW = torch.autograd.grad(sum_W, x_batch, create_graph=True)[0]
        
        # Compute dynamics and controller output
        u_vals = self.controller(x_batch)  # shape: [B, u_dim]
        f_vals = self.dynamics.f_torch(x_batch, u_vals)
        f_vals_u_detach = self.dynamics.f_torch(x_batch, u_vals.detach())
        
        # PDE residual: r(x) = grad(W)·f + mu*(1-W)*(1+W)*||x||^p
        dot_term = torch.sum(gradW * f_vals_u_detach, dim=1, keepdim=True)
        xnorm = torch.linalg.norm(x_batch - zero_tensor, dim=1, keepdim=True).pow(self.p)
        pde_residual = dot_term + self.mu * (1.0 - W_vals) * (1.0 + W_vals) * xnorm
        pde_loss = torch.mean(pde_residual ** 2) 
        
        # --- Modified Controller Loss ---
        grad_dot_f = torch.sum(gradW.detach() * f_vals, dim=1) 
        controller_loss = torch.mean(grad_dot_f)

        v_zero = torch.maximum(torch.relu(torch.square(V_zero) - 1e-4), torch.relu(1e-6 - torch.square(V_zero)))
        
        # Boundary Condition Loss: enforce lyapunov(bdry_points)=1
        W_vals_bdry = self.lyapunov(bdry_points)
        bdry_loss = F.mse_loss(W_vals_bdry, torch.full_like(W_vals_bdry, 1.0))
        
        default_weights = [self.loss_weights[0], self.loss_weights[1], 
                           self.loss_weights[2], self.loss_weights[3], self.loss_weights[4]]
        
        if self.mode == 'learnable':
            weighted_pde = (1.0/(2 * torch.exp(2 * self.log_sigma_pde))) * pde_loss + self.log_sigma_pde
            weighted_controller = (1.0/(2 * torch.exp(2 * self.log_sigma_controller))) * controller_loss + self.log_sigma_controller
            weighted_data = (1.0/(2 * torch.exp(2 * self.log_sigma_data))) * data_loss + self.log_sigma_data
            weighted_zero = (1.0/(2 * torch.exp(2 * self.log_sigma_zero))) * v_zero + self.log_sigma_zero
            weighted_bdry = (1.0/(2 * torch.exp(2 * self.log_sigma_bdry))) * bdry_loss + self.log_sigma_bdry
            final_loss = (weighted_pde + weighted_controller + weighted_data +
                          weighted_zero + weighted_bdry)
        else:
            final_loss = (
                default_weights[0] * pde_loss +
                default_weights[1] * controller_loss +
                default_weights[2] * data_loss +
                default_weights[3] * v_zero +
                default_weights[4] * bdry_loss
            )
        
        cex_condition = grad_dot_f > 0
        n_cex = int(cex_condition.sum().item())
        
        return loss_return(data_loss, pde_loss, controller_loss, v_zero, bdry_loss, final_loss, n_cex)
    