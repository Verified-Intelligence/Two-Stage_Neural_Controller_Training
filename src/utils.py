import torch
import matplotlib.pyplot as plt
from matplotlib import cm
import torch.nn.functional as F
import torch.nn as nn
from torchdiffeq import odeint
from typing import Optional, Tuple
from tqdm import tqdm

def generate_limits(factor_list, scale_vector, device, dtype):
    lower_limits = [
        torch.tensor([-f * s for s in scale_vector], dtype=dtype, device=device)
        for f in factor_list
    ]
    upper_limits = [
        torch.tensor([f * s for s in scale_vector], dtype=dtype, device=device)
        for f in factor_list
    ]
    return lower_limits, upper_limits

def pgd_attack_boundary(
    x_init: torch.Tensor,
    face_index: int,
    fixed_value: float,
    V: nn.Module,
    controller: nn.Module,
    dynamics,
    rho1: float,
    rho2: float,
    pgd_steps: int = 500,
    step_size: float = 0.01,
    free_limits: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    device: str = "cuda"
):
    x_attack = x_init.clone().to(device).requires_grad_(True)
    optimizer = torch.optim.Adam([x_attack], lr=step_size)
    n_dim = x_init.shape[1]
    
    # Set the outward normal for the face.
    n_vec = torch.zeros(n_dim, device=device)
    n_vec[face_index] = 1.0 if fixed_value > 0 else -1.0

    for step in range(pgd_steps):
        optimizer.zero_grad()
        with torch.no_grad():
            x_attack[:, face_index] = fixed_value
        
        V_vals = V(x_attack).squeeze()
        level_penalty = torch.relu(V_vals - rho2) + torch.relu(rho1 - V_vals)
        
        u_vals = controller(x_attack)
        f_vals = dynamics.f_torch(x_attack, u_vals)
        dot_vals = torch.matmul(f_vals, n_vec)
        
        objective = torch.mean(dot_vals - 10.0 * level_penalty)
        loss = -objective  # gradient ascent
        loss.backward()
        optimizer.step()
        
        if free_limits is not None:
            lb, ub = free_limits
            free_indices = [i for i in range(n_dim) if i != face_index]
            with torch.no_grad():
                x_attack[:, free_indices] = torch.max(torch.min(x_attack[:, free_indices], ub), lb)
    
    with torch.no_grad():
        x_attack[:, face_index] = fixed_value  
        u_final = controller(x_attack)
        f_final = dynamics.f_torch(x_attack, u_final)
        dot_final = torch.matmul(f_final, n_vec)
        V_final = V(x_attack).squeeze()
        in_level = (V_final >= rho1) & (V_final <= rho2)
        violation_mask = in_level & (dot_final >= 0)
    
    return x_attack.detach(), dot_final.detach(), violation_mask.detach()


def attack_all_boundaries_with_info(lower_limit: torch.Tensor, 
                                    upper_limit: torch.Tensor, 
                                    V: nn.Module, 
                                    controller: nn.Module, 
                                    dynamics, 
                                    rho1: float, 
                                    rho2: float, 
                                    pgd_steps: int = 1000,
                                    step_size: float = 0.01,
                                    num_samples: int = 10000,
                                    device: str = "cuda",
                                    dtype=torch.float32):
    '''
    Attack boundary condition.
    '''
    results = []
    n_dim = lower_limit.shape[0]
    
    for dim in range(n_dim):
        for bound_type in ['lower', 'upper']:
            fixed_value = lower_limit[dim].item() if bound_type == 'lower' else upper_limit[dim].item()
            free_indices = [i for i in range(n_dim) if i != dim]
            free_lower = lower_limit[free_indices]
            free_upper = upper_limit[free_indices]
            
            x_free = torch.rand(num_samples, len(free_indices), device=device, dtype=dtype)
            x_free = free_lower + (free_upper - free_lower) * x_free
            x_init_list = []
            for i in range(num_samples):
                x_full = []
                free_iter = iter(x_free[i])
                for j in range(n_dim):
                    if j == dim:
                        x_full.append(torch.tensor(fixed_value, device=device, dtype=dtype))
                    else:
                        x_full.append(next(free_iter))
                x_init_list.append(torch.stack(x_full))
            x_init_face = torch.stack(x_init_list, dim=0)
            free_limits = (free_lower, free_upper)
            
            x_attack, dot_final, violation_mask = pgd_attack_boundary(
                x_init=x_init_face,
                face_index=dim,
                fixed_value=fixed_value,
                V=V,
                controller=controller,
                dynamics=dynamics,
                rho1=rho1,
                rho2=rho2,
                pgd_steps=pgd_steps,
                step_size=step_size,
                free_limits=free_limits,
                device=device
            )
            
            indices = violation_mask.nonzero(as_tuple=False).flatten()
            if indices.numel() > 0:
                x_violations = x_attack[indices].detach()
                with torch.no_grad():
                    W_vals = V(x_violations)
                    if W_vals.ndim == 2 and W_vals.shape[1] == 1:
                        W_vals = W_vals.squeeze(1)
                    u_vals = controller(x_violations)
                    f_vals = dynamics.f_torch(x_violations, u_vals)
                
                with torch.enable_grad():
                    x_violations_grad = x_violations.clone().detach().requires_grad_(True)
                    W_vals_grad = V(x_violations_grad)
                    sum_W = W_vals_grad.sum()
                    gradW = torch.autograd.grad(sum_W, x_violations_grad, create_graph=False)[0]
                    dot_grad = torch.sum(gradW * f_vals, dim=1)
                
                violation_info = {
                    'states': x_violations,
                    'W_values': W_vals,
                    'u_values': u_vals,
                    'f_values': f_vals,
                    'derivatives': dot_grad,
                    'indices': indices,
                    'total_violations': indices.numel()
                }
            else:
                violation_info = None
            
            # Print detailed information for this face.
            print(f"\n=== Face: Dimension {dim} ({bound_type} boundary fixed at {fixed_value}) ===")
            if violation_info is not None:
                print(f"Found {violation_info['total_violations']} violations!")
                try:
                    mean_u = violation_info['u_values'].mean()
                    print(f"Mean u(x): {mean_u.item()}")
                except Exception as e:
                    print("Error computing mean u(x):", e)
                print("\nFirst 10 violations:")
                for i in range(min(10, violation_info['indices'].numel())):
                    print(f"\nViolation {i+1}:")
                    print(f"State x: {violation_info['states'][i].cpu().numpy()}")
                    print(f"W(x): {violation_info['W_values'][i].item():.6f}")
                    u_val = violation_info['u_values'][i]
                    if u_val.numel() == 1:
                        print(f"u(x): {u_val.item():.6f}")
                    else:
                        print(f"u(x): {u_val.cpu().numpy()}")
                    # print(f"f(x): {violation_info['f_values'][i].cpu().numpy()}")
                    # print(f"∇W(x): {violation_info['grad_values'][i].detach().cpu().numpy()}")
                    # print(f"∇W·f: {violation_info['derivatives'][i].detach().item():.6f}")
            else:
                print("No violations found on this face.")
            
            result = {
                'dimension': dim,
                'bound_type': bound_type,
                'fixed_value': fixed_value,
                'x_attack': x_attack,
                'dot_final': dot_final,
                'violation_mask': violation_mask,
                'violation_info': violation_info
            }
            results.append(result)
    return results


def pgd_attack(
    x_init: torch.Tensor,
    lyapunov_nn,        # A model with forward(x)-> W(x)
    controller,         # A model with forward(x)-> u(x)
    dynamics,           # A model with f_torch(x,u)-> f(x)
    rho1: float,        # Lower bound for the band
    rho2: float,        # Upper bound for the band
    pgd_steps: int = 50,
    step_size: float = 1e-2,
    clamp_box: torch.Tensor = None,
    device="cuda",
):
    x_adv = x_init.clone().to(device).requires_grad_(True)
    optimizer = torch.optim.Adam([x_adv], lr=step_size)

    for step in range(pgd_steps):
        optimizer.zero_grad()
        W_vals = lyapunov_nn(x_adv)  # shape: (batch_size,1)
        
        sum_W = W_vals.sum()
        gradW = torch.autograd.grad(sum_W, x_adv, create_graph=True)[0]  # (bs, x_dim)

        u_vals = controller(x_adv)                      # (batch_size, u_dim)
        f_vals = dynamics.f_torch(x_adv, u_vals)        # (batch_size, x_dim)

        dot_term = torch.sum(gradW * f_vals, dim=1, keepdim=True)  # (batch_size,1)
        stacked = torch.cat([dot_term, 
                             rho2 - W_vals, 
                             W_vals - rho1], dim=1)  # shape: (batch_size, 3)
        min_vals, _ = torch.min(stacked, dim=1, keepdim=True)      # shape: (batch_size,1)

        loss = -torch.mean(min_vals)
        loss.backward()
        optimizer.step()

        if clamp_box is not None:
            lb, ub = clamp_box
            with torch.no_grad():
                x_adv.data = torch.max(torch.min(x_adv, ub), lb)

    # ===========================
    # Final check of the violation
    # ===========================
    x_adv_for_grad = x_adv.clone().requires_grad_(True)
    W_final = lyapunov_nn(x_adv_for_grad)
    sum_W_final = W_final.sum()
    gradW_final = torch.autograd.grad(sum_W_final, x_adv_for_grad, create_graph=True)[0]
    u_vals_final = controller(x_adv_for_grad)
    f_vals_final = dynamics.f_torch(x_adv_for_grad, u_vals_final)
    dot_term_final = torch.sum(gradW_final * f_vals_final, dim=1)

    with torch.no_grad():
        band_mask = (W_final.squeeze() > rho1) & (W_final.squeeze() < rho2)
        violation_mask = band_mask & (dot_term_final >= 0)

        final_obj = torch.zeros_like(dot_term_final)
        final_obj[violation_mask] = dot_term_final[violation_mask]

        if violation_mask.any():
            violation_indices = violation_mask.nonzero(as_tuple=False).flatten()
            violation_states = x_adv_for_grad[violation_indices]
            violation_dots = dot_term_final[violation_indices]
            violation_W = W_final[violation_indices]
            violation_u = u_vals_final[violation_indices]
            violation_f = f_vals_final[violation_indices]
            violation_grad = gradW_final[violation_indices]

            violation_info = {
                'states': violation_states,
                'derivatives': violation_dots,
                'W_values': violation_W,
                'u_values': violation_u,
                'f_values': violation_f,
                'grad_values': violation_grad,
                'indices': violation_indices,
                'total_violations': violation_indices.shape[0]
            }
        else:
            violation_info = None

    return x_adv.detach(), final_obj, violation_info

def estimate_roa_size(V, lower_bound, upper_bound, rho1, rho2, grid_points_per_dim=50, 
                        device='cuda', batch_size=10000):
    lower_bound = lower_bound.to(device=device, dtype=torch.float32)
    upper_bound = upper_bound.to(device=device, dtype=torch.float32)
    dims = lower_bound.shape[0]
    
    grids = [torch.linspace(lower_bound[i], upper_bound[i], grid_points_per_dim, device=device)
             for i in range(dims)]
    
    mesh = torch.meshgrid(*grids, indexing='ij')
    grid_points = torch.stack(mesh, dim=-1).reshape(-1, dims)
    total_points = grid_points.shape[0]
    
    valid_count = 0
    for i in range(0, total_points, batch_size):
        batch_points = grid_points[i:i+batch_size]
        V_values = V(batch_points)
        if V_values.ndim == 2 and V_values.shape[1] == 1:
            V_values = V_values.squeeze(1)
        valid = (V_values > rho1) & (V_values < rho2)
        valid_count += valid.float().sum().item()
    
    proportion = valid_count / total_points
    
    box_volume = torch.prod(upper_bound - lower_bound).item()
    estimated_volume = proportion * box_volume
    
    return estimated_volume, proportion

def estimate_roa_size_memory_efficient(
    V, lower_bound, upper_bound, rho1, rho2, 
    num_samples=1000000, batch_size=5000, 
    device='cuda', seed=None):
    """
    Estimate the ROA size using random sampling, which is more memory-efficient 
    for high-dimensional systems.
    
    Args:
        V: Lyapunov function
        lower_bound: Lower bounds of the state space
        upper_bound: Upper bounds of the state space
        rho1: Lower threshold for ROA
        rho2: Upper threshold for ROA
        num_samples: Total number of random samples to use
        batch_size: Number of samples to process at once
        device: Device to use for computation
        seed: Random seed for reproducibility
        
    Returns:
        estimated_volume: Estimated volume of the ROA
        proportion: Proportion of samples in the ROA
    """
    # Set random seed if provided
    if seed is not None:
        torch.manual_seed(seed)
    
    lower_bound = lower_bound.to(device=device, dtype=torch.float32)
    upper_bound = upper_bound.to(device=device, dtype=torch.float32)
    dims = lower_bound.shape[0]
    
    # Compute box volume
    box_volume = torch.prod(upper_bound - lower_bound).item()
    print("box_volume", box_volume)
    
    valid_count = 0
    total_processed = 0
    
    # Process in batches
    for i in range(0, num_samples, batch_size):
        current_batch_size = min(batch_size, num_samples - total_processed)
        if current_batch_size <= 0:
            break
            
        random_points = torch.rand(current_batch_size, dims, device=device)
        scaled_points = random_points * (upper_bound - lower_bound) + lower_bound
        
        with torch.no_grad():
            V_values = V(scaled_points)
            if V_values.ndim == 2 and V_values.shape[1] == 1:
                V_values = V_values.squeeze(1)
                
            # Count points in ROA
            valid = (V_values > rho1) & (V_values < rho2)
            valid_count += valid.float().sum().item()
        
        total_processed += current_batch_size
        
        # Optional progress tracking for large sample sizes
        if total_processed % (num_samples // 10) == 0 or total_processed == num_samples:
            print(f"Processed {total_processed}/{num_samples} samples")
    
    proportion = valid_count / total_processed
    estimated_volume = proportion * box_volume
    
    return estimated_volume, proportion

def plot_V_heatmap(
    fig,
    V,
    rho,
    lower_limit,
    upper_limit,
    nx,
    x_boundary,
    plot_idx=[0, 1],
    mode=0.0,
    contour_colors=['black', 'blue'],
    V_color="k",
    V_lqr=None,
):
    device = lower_limit.device
    x_ticks = torch.linspace(
        lower_limit[plot_idx[0]], upper_limit[plot_idx[0]], 500, device=device
    )
    y_ticks = torch.linspace(
        lower_limit[plot_idx[1]], upper_limit[plot_idx[1]], 500, device=device
    )
    grid_x, grid_y = torch.meshgrid(x_ticks, y_ticks)
    if mode == "boundary":
        X = torch.ones(250000, nx, device=device) * x_boundary
    elif isinstance(mode, float):
        X = torch.ones(250000, nx, device=device) * upper_limit * mode
    X[:, plot_idx[0]] = grid_x.flatten()
    X[:, plot_idx[1]] = grid_y.flatten()

    with torch.no_grad():
        V_val = V(X)

    V_val = V_val.cpu().reshape(500, 500)
    grid_x = grid_x.cpu()
    grid_y = grid_y.cpu()

    ax = fig.add_subplot(111)
    im = ax.pcolor(grid_x, grid_y, V_val, cmap=cm.coolwarm)

    # Handle multiple rho values for contour plotting
    contour_handles = []
    if isinstance(rho, (list, tuple)):
        for i, r in enumerate(rho):
            color = contour_colors[i] if contour_colors and i < len(contour_colors) else V_color
            contour = ax.contour(grid_x, grid_y, V_val, levels=[r], colors=color, linewidths=3, zorder=3)
            contour.collections[0].set_label(f"rho={r:.3f}") 
    else:
        contour = ax.contour(grid_x, grid_y, V_val, levels=[rho], colors=V_color, linewidths=3, zorder=3)
        contour_handles.append(contour.collections[0])
        contour.collections[0].set_label(f"rho={rho:.3f}")

    lower_limit = lower_limit.cpu()
    upper_limit = upper_limit.cpu()
    ax.set_xlim(lower_limit[plot_idx[0]], upper_limit[plot_idx[0]])
    ax.set_ylim(lower_limit[plot_idx[1]], upper_limit[plot_idx[1]])
    
    cbar = fig.colorbar(im, ax=ax)

    return ax, cbar

def plot_V_and_grad_heatmap(fig, W, rho, lower_bound, upper_bound, dims, plot_idx=[0,1], grid_points_per_dim=150, device='cuda'):
    """
    Plots two heatmaps side-by-side:
      - Left: The value of V(x) computed by W on a grid.
      - Right: The norm of the gradient of V(x) computed on the same grid.
    
    This version uses pcolormesh with the same grid (created using torch.meshgrid with indexing='ij'),
    ensuring that the orientation is the same as in plot_V_heatmap.
    
    Args:
        fig: A matplotlib figure.
        W (callable): A model mapping [N, dims] -> [N, 1] (e.g., the Lyapunov network).
        rho (list): Two values defining a band (e.g., [rho_low, rho_high]) for reference.
        lower_bound (torch.Tensor): 1D tensor with lower bounds for each dimension.
        upper_bound (torch.Tensor): 1D tensor with upper bounds for each dimension.
        dims (int): Total dimension of the state.
        plot_idx (list): Two indices to select which two dimensions to plot.
        grid_points_per_dim (int): Number of grid points per plotted dimension.
        device (str): Device to run computations on.
    
    Returns:
        Two axes objects: one for V(x) and one for ||grad(V)||.
    """
    import matplotlib.pyplot as plt

    idx0, idx1 = plot_idx
    grid0 = torch.linspace(lower_bound[idx0].item(), upper_bound[idx0].item(), grid_points_per_dim, device=device)
    grid1 = torch.linspace(lower_bound[idx1].item(), upper_bound[idx1].item(), grid_points_per_dim, device=device)
    mesh0, mesh1 = torch.meshgrid(grid0, grid1, indexing='ij')

    total_points = grid_points_per_dim * grid_points_per_dim
    midpoints = (lower_bound + upper_bound) / 2.0 
    full_state = midpoints.repeat(total_points, 1)  
    full_state[:, idx0] = mesh0.reshape(-1)
    full_state[:, idx1] = mesh1.reshape(-1)
    full_state = full_state.to(device=device, dtype=torch.float32)
    full_state.requires_grad_(True)
    
    V_values = W(full_state)
    if V_values.ndim == 2 and V_values.shape[1] == 1:
        V_values = V_values.squeeze(1)
    
    sum_V = V_values.sum()
    grad_V = torch.autograd.grad(sum_V, full_state, create_graph=False)[0]
    grad_norm = grad_V.norm(dim=1)
    
    V_map = V_values.reshape(grid_points_per_dim, grid_points_per_dim).detach().cpu().numpy()
    grad_norm_map = grad_norm.reshape(grid_points_per_dim, grid_points_per_dim).detach().cpu().numpy()
    
    ax1 = fig.add_subplot(1, 2, 1)
    im1 = ax1.pcolormesh(mesh0.cpu(), mesh1.cpu(), V_map, cmap='viridis', shading='auto')
    ax1.set_title("V(x)")
    ax1.set_xlabel(f"x[{idx0}]")
    ax1.set_ylabel(f"x[{idx1}]")
    fig.colorbar(im1, ax=ax1)

    ax2 = fig.add_subplot(1, 2, 2)
    im2 = ax2.pcolormesh(mesh0.cpu(), mesh1.cpu(), grad_norm_map, cmap='inferno', shading='auto')
    ax2.set_title("||grad V(x)||")
    ax2.set_xlabel(f"x[{idx0}]")
    ax2.set_ylabel(f"x[{idx1}]")
    fig.colorbar(im2, ax=ax2)

    return ax1, ax2

def plot_u_heatmap(
    fig,
    controller,
    lower_bound,
    upper_bound,
    dims,
    plot_idx=[0,1],
    grid_points_per_dim=500,  # matching the 500 grid points as in plot_V_heatmap
    device='cuda'
):
    """
    Plots a heatmap of the controller output u(x) on a 2D slice of the state space,
    formatted to match the style of plot_V_heatmap.

    Args:
        fig: A matplotlib figure object.
        controller (callable): A model that maps [B, dims] -> [B, 1] or [B, u_dim].
        lower_bound (torch.Tensor): 1D tensor of shape [dims] with lower limits.
        upper_bound (torch.Tensor): 1D tensor of shape [dims] with upper limits.
        dims (int): Total state dimension.
        plot_idx (list): Two indices to select which two dimensions to plot.
        grid_points_per_dim (int): Number of grid points per dimension.
        device (str): Device for computation.
    
    Returns:
        ax: The matplotlib axis with the heatmap.
    """
    idx0, idx1 = plot_idx

    x_ticks = torch.linspace(lower_bound[idx0].item(), upper_bound[idx0].item(), grid_points_per_dim, device=device)
    y_ticks = torch.linspace(lower_bound[idx1].item(), upper_bound[idx1].item(), grid_points_per_dim, device=device)
    grid_x, grid_y = torch.meshgrid(x_ticks, y_ticks)
    
    total_points = grid_points_per_dim * grid_points_per_dim
    
    midpoints = (lower_bound + upper_bound) / 2.0  # shape: [dims]
    full_state = midpoints.repeat(total_points, 1)  # shape: [total_points, dims]
    full_state[:, idx0] = grid_x.flatten()
    full_state[:, idx1] = grid_y.flatten()
    full_state = full_state.to(device=device, dtype=torch.float32)
    
    # Evaluate the controller on the full state grid.
    with torch.no_grad():
        u_vals = controller(full_state) 
    
    # If control is one-dimensional, reshape; otherwise, use the norm.
    if u_vals.shape[1] == 1:
        u_map = u_vals.reshape(grid_points_per_dim, grid_points_per_dim).cpu().numpy()
    else:
        u_norm = u_vals.norm(dim=1)
        u_map = u_norm.reshape(grid_points_per_dim, grid_points_per_dim).cpu().numpy()
    
    ax = fig.add_subplot(111)
    im = ax.pcolor(grid_x.cpu(), grid_y.cpu(), u_map, cmap=cm.coolwarm, shading='auto')
    ax.set_title("u(x)")
    ax.set_xlabel(f"x[{idx0}]")
    ax.set_ylabel(f"x[{idx1}]")
    fig.colorbar(im, ax=ax)
    return ax

def plot_pde_loss_heatmap(fig, lyaloss, lower_bound, upper_bound, dims, plot_idx=[0, 1], grid_points_per_dim=500, device='cuda'):
    idx0, idx1 = plot_idx

    x_ticks = torch.linspace(lower_bound[idx0].item(), upper_bound[idx0].item(), grid_points_per_dim, device=device)
    y_ticks = torch.linspace(lower_bound[idx1].item(), upper_bound[idx1].item(), grid_points_per_dim, device=device)
    grid_x, grid_y = torch.meshgrid(x_ticks, y_ticks, indexing='ij')
    total_points = grid_points_per_dim * grid_points_per_dim

    midpoints = (lower_bound + upper_bound) / 2.0  
    full_state = midpoints.repeat(total_points, 1)   
    full_state[:, idx0] = grid_x.flatten()
    full_state[:, idx1] = grid_y.flatten()
    full_state = full_state.to(device=device, dtype=torch.float32)
    full_state.requires_grad_(True)

    W_vals = lyaloss.lyapunov(full_state)  # Shape: [B, 1]
    sum_W = W_vals.sum()
    gradW = torch.autograd.grad(sum_W, full_state, create_graph=True)[0]  # Shape: [B, state_dim]
    
    # Evaluate the controller and dynamics.
    u_vals = lyaloss.controller(full_state)             # Shape: [B, u_dim]
    f_vals = lyaloss.dynamics.f_torch(full_state, u_vals)  # Shape: [B, state_dim]
    dot_term = torch.sum(gradW * f_vals, dim=1, keepdim=True)
    xnorm = torch.norm(full_state, dim=1, keepdim=True).pow(lyaloss.p)
    pde_loss = (dot_term + lyaloss.mu * (1.0 - W_vals) * (1.0 + W_vals) * xnorm) ** 2

    # Reshape the PDE loss to a 2D grid.
    pde_loss_map = pde_loss.reshape(grid_points_per_dim, grid_points_per_dim).detach().cpu().numpy()

    # Plot the heatmap.
    ax = fig.add_subplot(111)
    im = ax.pcolormesh(grid_x.cpu(), grid_y.cpu(), pde_loss_map, cmap='coolwarm', shading='auto')
    ax.set_title("PDE Loss Heatmap")
    ax.set_xlabel(f"x[{idx0}]")
    ax.set_ylabel(f"x[{idx1}]")
    fig.colorbar(im, ax=ax)
    
    return ax

def simulate_trajectories(
    dynamics,
    controller,
    x0: torch.Tensor,
    dt: float = 0.01,
    steps: int = 1000,
    method: str = "rk4",
    num_full: int = 100,
    device="cuda",
):
    """
    Simulate a batch of trajectories under `dynamics`+`controller`.
    
    - Only keeps ALL timesteps for a random subset of `num_full` trajectories.
    - For the other trajectories, only returns the final state.
    
    Returns:
      full_idx: indices of the trajectories stored in full_trajs
      full_trajs: Tensor of shape (num_full, steps+1, nx)
      final_states: Tensor of shape (batch_size, nx)
    """
    x0 = x0.to(device)
    batch_size, nx = x0.shape
    
    Nf = min(num_full, batch_size)
    perm = torch.randperm(batch_size, device=device)
    full_idx = perm[:Nf]
    
    full_trajs = torch.zeros(Nf, steps+1, nx, device=device, dtype=x0.dtype)
    final_states = torch.zeros(batch_size, nx, device=device, dtype=x0.dtype)
    x_current = x0.clone()
    full_trajs[:, 0, :] = x_current[full_idx]
    
    with torch.no_grad():
        for t in tqdm(range(steps)):
            if method.lower() == "euler":
                u = controller(x_current)
                f_val = dynamics.f_torch(x_current, u)
                x_next = x_current + dt * f_val

            elif method.lower() == "rk4":
                u1 = controller(x_current)
                k1 = dynamics.f_torch(x_current, u1)
                x2 = x_current + 0.5 * dt * k1
                u2 = controller(x2)
                k2 = dynamics.f_torch(x2, u2)
                x3 = x_current + 0.5 * dt * k2
                u3 = controller(x3)
                k3 = dynamics.f_torch(x3, u3)
                x4 = x_current + dt * k3
                u4 = controller(x4)
                k4 = dynamics.f_torch(x4, u4)
                x_next = x_current + (dt/6.0) * (k1 + 2*k2 + 2*k3 + k4)
            else:
                raise ValueError(f"Unknown method '{method}'")

            full_trajs[:, t+1, :] = x_next[full_idx]
            x_current = x_next

        final_states[:] = x_current

    return full_idx, full_trajs, final_states

def check_convergence(
    x_trajs: torch.Tensor,
    x_equilibrium: torch.Tensor,
    tolerance: float = 1e-4,
):
    final_states = x_trajs[:, -1, :]
    dist = torch.norm(final_states - x_equilibrium.unsqueeze(0), dim=1, p=float('inf'))
    success_mask = dist < tolerance
    success_rate = success_mask.float().mean().item()
    return success_mask, success_rate
