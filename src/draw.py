import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import torch.nn.functional as F
import torch
import os
import sys
from typing import OrderedDict
import argparse
import numpy as np

sys.path.append(os.path.dirname(os.path.realpath(__file__)))
sys.path.append("../torchdyn")

import src.utils as utils
from src.training_config import configs

device = torch.device("cuda")
dtype = torch.float32

def parse_args():
    """
    Parses command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Train a neural Lyapunov function for the Van der Podl dynamics."
    )
    parser.add_argument("--system", type=str, help="Dynamical System.")
    parser.add_argument("--c1", type=float, default=0.01, help="c1.")
    parser.add_argument("--c2", type=float, default=0.95, help="c2.")
    parser.add_argument("--attack", action='store_true', help="Whether to do pgd attack or just visualize")
    parser.add_argument("--compute_volume", action='store_true', help="Whether to compute volume of ROA")
    parser.add_argument("--pgd_steps", type=int, default=1000, help="PGD attack steps.")
    parser.add_argument("--pgd_step_size", type=float, default=0.001, help="PGD attack step size.")
    parser.add_argument("--num_points", type=int, default=2000000, help="Number of pgd reinitializations.")
    parser.add_argument("--dt", type=float, default=0.003, help="Discretization Timestep.")
    parser.add_argument("--simulate_steps", type=int, default=10000, help="Simulate steps.")
    parser.add_argument('--box', type=float, nargs='+', default=[1.0], help='Finetuning Box')
    parser.add_argument('--plot_idx', type=int, nargs='+', default=[1], help='Two dimensions to plot for high dim system')
    parser.add_argument("--load_dir", type=str, help='Pretrained model path')
    parser.add_argument("--save_dir", type=str, help='Save the visualizations to which directory')
    return parser.parse_args()

def sample_points_in_band_single(lyapunov, init_points, band_lower, band_upper,
                                 pgd_steps=100, step_size=1e-2, num_points=100,
                                 limits=None, device='cuda'):
    """
    PGD adjust points so that their Lyapunov values lie in [band_lower, band_upper].
    If limits is provided, it should be a tuple (lower_limit, upper_limit) for clamping.
    """
    x_adv = init_points.clone().to(device).detach().requires_grad_(True)
    optimizer = torch.optim.Adam([x_adv], lr=step_size)
    
    for step in range(pgd_steps):
        optimizer.zero_grad()
        V_vals = lyapunov(x_adv).squeeze(1)
        loss_band = torch.max(F.relu(V_vals - band_upper), F.relu(band_lower - V_vals))
        loss = loss_band.mean()
        loss.backward()
        optimizer.step()
        
        if limits is not None:
            lower_lim, upper_lim = limits
            lower_lim = lower_lim.to(x_adv.device)
            upper_lim = upper_lim.to(x_adv.device)
            x_adv.data = torch.max(torch.min(x_adv.data, upper_lim), lower_lim)
    
    with torch.no_grad():
        V_final = lyapunov(x_adv).squeeze(1)
        mask = (V_final >= band_lower) & (V_final <= band_upper)
        x_in_band = x_adv[mask]
    
    if x_in_band.shape[0] > num_points:
        indices = torch.randperm(x_in_band.shape[0])[:num_points]
        x_in_band = x_in_band[indices]
    
    return x_in_band.detach()

def load_model(controller, lyapunov_nn, load_dir):
    controller_state = OrderedDict()
    lyapunov_state = OrderedDict()
    model = torch.load(load_dir)
    if 'state_dict' in model.keys():
        new_model = model['state_dict']
    else:
        new_model = model
    
    model = new_model
    for k, v in model.items():
        if k.startswith("controller."):
            new_key = k.replace("controller.", "")
            controller_state[new_key] = v
        elif k.startswith("lyapunov."):
            new_key = k.replace("lyapunov.", "")
            lyapunov_state[new_key] = v

    controller.load_state_dict(controller_state)
    lyapunov_nn.load_state_dict(lyapunov_state)
    controller.to(device).to(dtype)
    lyapunov_nn.to(device).to(dtype)
    
    return controller, lyapunov_nn

if __name__ == "__main__":  
    args = parse_args()
    
    cfg = configs.get(args.system)
    if cfg is None:
        raise ValueError(f"Unknown system {args.system!r}")
    
    dynamics = cfg.dynamics
    controller = cfg.controller.to(device).to(dtype)
    lyapunov_nn = cfg.lyapunov.to(device).to(dtype)
    
    mu = cfg.mu
    p = cfg.p
    dt = cfg.dt
    rollout_steps = cfg.steps
    loss_weights = cfg.loss_weights
    
    controller, lyapunov_nn = load_model(controller, lyapunov_nn, args.load_dir)
    lower_limit = -torch.tensor(args.box, device=device, dtype=dtype)
    upper_limit = torch.tensor(args.box, device=device, dtype=dtype)
    
    zero_tensor = torch.zeros_like(lower_limit, device=device).unsqueeze(0)
    nx = dynamics.x_equilibrium.size(0)
    print(zero_tensor)
    print(lyapunov_nn(zero_tensor))
    print(controller(zero_tensor))
    
    c1 = args.c1
    c2 = args.c2
    
    if args.attack:
        x_init = lower_limit.unsqueeze(0) + (upper_limit - lower_limit).unsqueeze(0) * torch.rand((args.num_points, nx), device=device, dtype=dtype)
        x_attack, final_obj, violation_info = utils.pgd_attack(
            x_init=x_init, lyapunov_nn=lyapunov_nn, controller=controller, dynamics=dynamics, 
            rho1=c1, rho2=c2, pgd_steps=args.pgd_steps, step_size=args.pgd_step_size, clamp_box=(lower_limit, upper_limit), device=device
        )

        if violation_info is not None:
            print(f"Found {violation_info['total_violations']} violations!")
            print(f"Mean u(x): {violation_info['u_values'].mean()}")
            print("\nFirst 10 violations:")
            for i in range(min(10, len(violation_info['indices']))):
                print(f"\nViolation {i+1}:")
                print(f"State x: {violation_info['states'][i].cpu().numpy()}")
                print(f"W(x): {violation_info['W_values'][i].item():.6f}")
                print(f"u(x): {violation_info['u_values'][i]}")
                print(f"f(x): {violation_info['f_values'][i].cpu().numpy()}")
                print(f"∇W(x): {violation_info['grad_values'][i].cpu().numpy()}")
                print(f"∇W·f: {violation_info['derivatives'][i].item():.6f}")
        else:
            print("No violations found in this batch.")
            
        all_boundary_results = utils.attack_all_boundaries_with_info(
            lower_limit=lower_limit, upper_limit=upper_limit, V=lyapunov_nn, controller=controller, dynamics=dynamics,
            rho1=c1, rho2=c2, pgd_steps=1000, step_size=0.01, num_samples=50000, device=device, dtype=dtype
        )
        
    os.makedirs(os.path.dirname(args.save_dir), exist_ok=True)
    ## Plot V
    fig = plt.figure()
    utils.plot_V_heatmap(
        fig, lyapunov_nn, [c1, c2], lower_limit, upper_limit, nx, None, plot_idx=args.plot_idx,
    )
    plt.title(f'{args.system}')
    plt.tight_layout()
    plt.savefig(os.path.join(args.save_dir, 'V_heatmap.png'), dpi=300)
    
    ## Plot V grad
    fig = plt.figure(figsize=(12, 5))
    ax_V, ax_grad = utils.plot_V_and_grad_heatmap(fig, lyapunov_nn, [c1,c2], lower_limit, upper_limit, dims=nx, plot_idx=args.plot_idx, grid_points_per_dim=150, device=device)
    plt.suptitle(f"V(x) and ||grad V(x)|| for {args.system}")
    plt.tight_layout()
    plt.savefig(os.path.join(args.save_dir, f'{args.system}_V_grad.png'), dpi=300)
    
    ## Plot u
    fig = plt.figure(figsize=(6,5))
    utils.plot_u_heatmap(fig, controller, lower_limit, upper_limit, dims=nx, plot_idx=args.plot_idx, grid_points_per_dim=150, device=device)
    plt.title(f"Controller output u(x) for {args.system}")
    plt.tight_layout()
    plt.savefig(os.path.join(args.save_dir, f'{args.system}_u.png'), dpi=300)
    
    # # make sure the bounds we use for plotting live on the CPU
    lower_limit = lower_limit.cuda()     # ← added
    upper_limit = upper_limit.cuda()     # ← added
    batch_size = 50000 # how many random initial states we test
    nx = dynamics.x_equilibrium.size(0)
    x0_box = lower_limit.unsqueeze(0) + (upper_limit - lower_limit).unsqueeze(0) * torch.rand((batch_size, nx), device=device, dtype=dtype)
    x0_box = sample_points_in_band_single(
        lyapunov_nn, x0_box, 0.0, c2, pgd_steps=100, step_size=1e-1, num_points=5000, limits=(lower_limit, upper_limit), device=device
    )
    
    with torch.no_grad():
        lyapunov_values = lyapunov_nn(x0_box)
        valid_mask = (lyapunov_values < c2).squeeze()
        filtered_x0 = x0_box[valid_mask]
        
    full_idx, x_trajs, final_states = utils.simulate_trajectories(dynamics, controller, filtered_x0, dt=args.dt, steps=args.simulate_steps, method="euler", num_full=30, device=device)
    success_mask, success_rate = utils.check_convergence(final_states.unsqueeze(1), dynamics.x_equilibrium.to(device), tolerance=1e-3)
    print(f"Convergence success rate: {success_rate*100:.1f}% ({success_mask.sum().item()}/{len(filtered_x0)})")
    
    i_traj = 0     
    time_axis = dt * torch.arange(args.simulate_steps+1)
    plt.figure()
    for dim in range(nx):
        plt.plot(time_axis.detach().cpu(), x_trajs[i_traj, :, dim].detach().cpu(), label=f'x[{dim}]')
    plt.title(f"Trajectory for initial state {i_traj}, converged={success_mask[i_traj].item()}")
    plt.xlabel("Time (s)")
    plt.ylabel("State")
    plt.savefig(os.path.join(args.save_dir, f'{args.system}_traj.png'), dpi=300)
    
    ## Compute Volume
    if args.compute_volume:
        estimated_volume, proportion = utils.estimate_roa_size_memory_efficient(lyapunov_nn, lower_limit, upper_limit, 0, c2, num_samples=50000000, device=device)
        print(f"Estimated ROA volume: {estimated_volume:.4f} (proportion = {proportion:.4f})")

        estimated_volume, proportion = utils.estimate_roa_size_memory_efficient(lyapunov_nn, lower_limit, upper_limit, 0, c1, num_samples=50000000, device=device)
        print(f"Estimated ROA volume with cex: {estimated_volume:.4f} (proportion = {proportion:.4f})")
