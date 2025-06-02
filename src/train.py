import torch.nn.functional as F
import torch
import numpy as np
import os
import sys
import logging
import time
from torch.optim.lr_scheduler import StepLR
import argparse
import numpy as np
from torchdiffeq import odeint
import random

sys.path.append(os.path.dirname(os.path.realpath(__file__)))
sys.path.append("../torchdyn")

import src.loss as loss
import src.utils as utils
from src.training_config import configs

device = 'cuda'
dtype = torch.float32

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def parse_args():
    """
    Parses command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Train a controller jointly with Lyapunov function on continuous-time systems."
    )
    parser.add_argument("--system", type=str, help="Dynamical System.")
    parser.add_argument("--max_iter", type=int, default=5000, help="Maximum number of iterations.")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs per iteration.")
    parser.add_argument("--samples_per_iter", type=int, default=64, help="Samples per iteration.")
    parser.add_argument("--learning_rate", type=float, default=0.003, help="Learning rate.")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay.")
    parser.add_argument("--l1_reg", type=float, default=0.0, help="L1 regularization factor.")
    parser.add_argument("--mode", type=str, default='learnable', help="Reweighting Stretegy.")
    parser.add_argument("--use_data_loss", type=bool, default=True, help="Whether to use data loss during training.")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch Size in Training")
    parser.add_argument("--bdry_ratio", type=float, default=2, help="Which domain to select bdry points")
    parser.add_argument("--update_interval", type=int, default=1000, help="Interval to Update Box")
    parser.add_argument("--lyapunov_thershold", type=float, default=0.9, help="Levelset to consider trajectories")
    parser.add_argument("--num_trajectories", type=int, default=2000, help="Number of Trajectories Simulated when Updating Box")
    parser.add_argument("--traj_dt", type=float, default=0.0005, help="Dt when simulate trajectory.")
    parser.add_argument("--traj_steps", type=int, default=80000, help="Steps when simulate trajectory.")
    parser.add_argument("--traj_pgd_steps", type=int, default=100, help="PGD Steps when simulate trajectory.")
    parser.add_argument("--adaptive", type=bool, default=False, help="Whether to use an adpative solver when simulate trajectories.")
    parser.add_argument("--l1_lambda", type=float, default=0.0, help="weight of L1 regularization on Lyapunov net parameters")
    parser.add_argument("--max_growth", type=float, default=10, help="Maximum value can the box enlarge per update.")
    parser.add_argument("--convergence_threshold", type=float, default=0.01, help="Convergence Threshold.")
    parser.add_argument("--candidate_size", type=int, default=640, help="Num points consider.")
    parser.add_argument("--pgd_steps", type=int, default=30, help="PGD steps in sampling collocation points.")
    parser.add_argument("--save_dir", type=str, help='Save to which directory')
    parser.add_argument("--log_interval", type=int, default=100, help='How often we print the training log')
    parser.add_argument("--save_interval", type=int, default=5000, help='How often we save model checkpoint')
    parser.add_argument("--ablation", action="store_true", help='Abaltion study')
    parser.add_argument("--seed", type=int, default=0, help="Seed")
    return parser.parse_args()

def sample_out_of_domain(limits, n, scale=1., device='cpu', dtype=torch.float32):
    lower_limit, upper_limit = limits  # each of shape [nx]
    nx = lower_limit.shape[0]
    center = (lower_limit + upper_limit) / 2.
    half_range = (upper_limit - lower_limit) / 2.

    scaled_lower = center - scale * half_range
    scaled_upper = center + scale * half_range

    x = torch.rand(n, nx, device=device, dtype=dtype) * (scaled_upper - scaled_lower).unsqueeze(0) + scaled_lower.unsqueeze(0)
    indices = torch.randint(0, nx, (n,), device=device)
    choices = torch.randint(0, 2, (n,), device=device)  

    lower_vals = scaled_lower[indices]
    upper_vals = scaled_upper[indices]
    boundary_vals = torch.where(choices.bool(), upper_vals, lower_vals)

    x_out = x.clone()
    x_out[torch.arange(n, device=device), indices] = boundary_vals

    noise = torch.rand(n, nx, device=device, dtype=dtype) * 0.1
    x_out = x_out + torch.sign(x_out) * noise

    return x_out

def split_levelset(upper_bound: float, num_splits: int):
    bands = []
    step = upper_bound / num_splits
    for i in range(num_splits):
        band_lower = i * step
        band_upper = (i + 1) * step
        bands.append((band_lower, band_upper))
    return bands

def sample_points_in_band_single(lyaloss, init_points, band_lower, band_upper,
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
        V_vals = lyaloss.lyapunov(x_adv).squeeze(1)
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
        V_final = lyaloss.lyapunov(x_adv).squeeze(1)
        mask = (V_final >= band_lower) & (V_final <= band_upper)
        x_in_band = x_adv[mask]

    if x_in_band.shape[0] > num_points:
        indices = torch.randperm(x_in_band.shape[0])[:num_points]
        x_in_band = x_in_band[indices]

    return x_in_band.detach()

def sample_points_in_bands(lyaloss, init_points, band_list, 
                        pgd_steps=100, step_size=1e-2, num_points=100, limits=None, device='cuda'):
    band_points = []
    for (band_lower, band_upper) in band_list:
        pts = sample_points_in_band_single(
            lyaloss, init_points, band_lower, band_upper,
            pgd_steps=pgd_steps, step_size=step_size,
            num_points=num_points, limits=limits, device=device
        )
        if pts is not None and pts.size(0) > 0:
            band_points.append(pts)
    if len(band_points) > 0:
        return torch.cat(band_points, dim=0)
    else:
        print('No points in band.')
        return torch.empty((0, init_points.size(1)), device=device)

def sample_points_near_boundary(lyaloss, init_points, target=1.0, pgd_steps=30, step_size=1e-2, 
                                num_points=100, limits=None, device='cuda'):
    """
    Adjust points using PGD so that their Lyapunov values approach target.
    When there are more than num_points, select those closest to target.
    For our purpose, just keep target equals 1.
    """
    x_adv = init_points.clone().to(device).detach().requires_grad_(True)
    optimizer = torch.optim.Adam([x_adv], lr=step_size)

    target_tensor = torch.full((x_adv.size(0),), target, device=device)
    for step in range(pgd_steps):
        optimizer.zero_grad()
        V_vals = lyaloss.lyapunov(x_adv).squeeze(1)
        loss = F.mse_loss(V_vals, target_tensor)
        loss.backward()
        optimizer.step()

        if limits is not None:
            lower_lim, upper_lim = limits
            lower_lim = lower_lim.to(x_adv.device)
            upper_lim = upper_lim.to(x_adv.device)
            x_adv.data = torch.max(torch.min(x_adv.data, upper_lim), lower_lim)

        out_x = x_adv
        if out_x.shape[0] > num_points:
            with torch.no_grad():
                V_vals = lyaloss.lyapunov(out_x).squeeze(1)
                diff = (V_vals - target).abs()
                _, sorted_indices = torch.sort(diff, descending=False)
                selected_indices = sorted_indices[:num_points]
                out_x = out_x[selected_indices]

        return out_x.detach()
    
def closed_loop_dynamics(t, x, dynamics, controller):
    u = controller(x)
    return dynamics.f_torch(x, u)

def update_box_from_trajectories(controller, dynamics, x0, steps, dt, round_to=1.0, 
                                current_lower=None, current_upper=None, convergence_thresh=0.1, 
                                expansion_factor=1.2, adaptive=False, hard_lower=None, hard_upper=None):
    device = x0.device
    B, n = x0.shape

    x = x0.clone()
    min_traj = x.clone()
    max_traj = x.clone()

    # simulate
    if adaptive:
        T = dt * steps
        t_span = torch.tensor([0.0, T], device=device)
        with torch.no_grad():
            traj = odeint(lambda t, x: closed_loop_dynamics(t, x, dynamics, controller),
                          x0, t_span, method='dopri5')
        x_start, x_end = traj[0], traj[1]
        min_traj = torch.min(min_traj, x_end)
        max_traj = torch.max(max_traj, x_end)
        x = x_end
    else:
        with torch.no_grad():
            for _ in range(steps):
                u = controller(x)
                f = dynamics.f_torch(x, u)
                x = x + dt * f
                min_traj = torch.min(min_traj, x)
                max_traj = torch.max(max_traj, x)

    eq = dynamics.x_equilibrium.to(device)
    final_mask = (x.sub(eq).abs() < convergence_thresh).all(dim=1)
    num_converged = int(final_mask.sum().item())

    if num_converged == 0 or num_converged <= 0.01 * B:
        print("No trajectories converged. Returning current training box.")
        return (current_lower, current_upper), 0

    # filter extrema to only those trajectories that converged
    min_c = min_traj[final_mask]
    max_c = max_traj[final_mask]

    # global min/max across all converged trajectories
    global_min = min_c.min(dim=0)[0]
    global_max = max_c.max(dim=0)[0]

    # round and expand
    new_lower = torch.floor(global_min / round_to) * round_to * expansion_factor
    new_upper = torch.ceil (global_max / round_to) * round_to * expansion_factor
    
    # ---------- clamp to hard limits ----------
    if hard_lower is not None:
        new_lower = torch.maximum(new_lower, hard_lower.to(new_lower.device))
    if hard_upper is not None:
        new_upper = torch.minimum(new_upper, hard_upper.to(new_upper.device))

    return (new_lower, new_upper), num_converged

def cap_box_growth(curr_lower, curr_upper, prop_lower, prop_upper, max_growth=1.5):
    new_upper = torch.minimum(prop_upper, curr_upper * max_growth)
    new_lower = torch.maximum(prop_lower, curr_lower * max_growth)
    return new_lower, new_upper
 
def train_full(
    args,
    lyaloss,
    lower_limit,  
    upper_limit,  
    box,
    hard_lower,
    hard_upper,
    logger=None,
):
    device = lower_limit.device
    dtype = lower_limit.dtype
    nx = lower_limit.size(0)
    l1_lambda = args.l1_lambda

    if logger is None:
        logger = logging.getLogger(__name__)

    start_time = time.time()
    main, meta = [], []
    for n,p in lyaloss.named_parameters():
        (meta if 'log_sigma' in n else main).append(p)

    optimizer = torch.optim.AdamW(
            [{'params': main, 'lr': args.learning_rate},
            {'params': meta, 'lr': args.learning_rate}] )
    scheduler = StepLR(optimizer, step_size=500, gamma=0.95)

    def sample_uniform(n):
        # Sample uniformly in [lower_limit, upper_limit]
        return torch.rand((n, nx), device=device, dtype=dtype) * (upper_limit - lower_limit) + lower_limit

    no_convergence_counter = 0
    for i in range(args.max_iter):
        limits = (lower_limit, upper_limit)
        if ((i+1) % args.update_interval == 0):
            x0_samples_full = sample_uniform(args.num_trajectories * 2)
            x0_samples_cand = sample_points_in_band_single(
                lyaloss, x0_samples_full, 0.0, args.lyapunov_thershold, pgd_steps=args.traj_pgd_steps, num_points=args.num_trajectories, limits=limits, device=device
            )
            with torch.no_grad():
                V_sampled = lyaloss.lyapunov(x0_samples_cand).squeeze(1)
                valid_mask = (V_sampled < args.lyapunov_thershold)
                x0_samples = x0_samples_cand[valid_mask]

            # Update box limits (both lower and upper) based on trajectories.
            (new_lower, new_upper), num_converged = update_box_from_trajectories(
                controller, dynamics, x0_samples,
                steps=args.traj_steps, dt=args.traj_dt, round_to=1.0,
                current_lower=lower_limit, current_upper=upper_limit, convergence_thresh=args.convergence_threshold,
                hard_lower=hard_lower, hard_upper=hard_upper
            )

            if num_converged == 0:
                no_convergence_counter += 1
            else:
                new_lower, new_upper = cap_box_growth(lower_limit, upper_limit, new_lower, new_upper, max_growth=args.max_growth)
                print(num_converged / x0_samples.size(0))
                no_convergence_counter = 0
                lower_limit = new_lower
                upper_limit = new_upper
                lyaloss.upper_limit = new_upper
                l1_lambda = min(l1_lambda, 0.005)
                print("Updated training box:")
                print("  Lower limit:", lower_limit)
                print("  Upper limit:", upper_limit)
                print(f"Current Bdry Ratio: {args.bdry_ratio}")

            if no_convergence_counter >= 5:
                # Expand the box if trajectories fail to converge for many iterations.
                lower_limit = lower_limit - 0.5 * (upper_limit - lower_limit)
                upper_limit = upper_limit + 0.5 * (upper_limit - lower_limit)
                no_convergence_counter = 0

        candidate_x = sample_uniform(10 * args.candidate_size)

        band_list = split_levelset(args.lyapunov_thershold, 5)
        num_points_per_band = int(args.candidate_size / len(band_list))
        inbox_x = sample_points_in_bands(lyaloss, candidate_x, band_list, pgd_steps=args.pgd_steps, 
                                        num_points=num_points_per_band, limits=limits, device=device)
        out_x = sample_points_near_boundary(lyaloss, inbox_x, target=1.0, pgd_steps=args.pgd_steps, num_points=inbox_x.size(0), limits=limits)
        adv_x = torch.cat([inbox_x, out_x], dim=0)

        bdry_points = sample_out_of_domain(limits, args.batch_size, args.bdry_ratio, device=device)
        dataset = torch.utils.data.TensorDataset(adv_x)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

        for _ in range(args.epochs):
            for adv_batch in dataloader:
                bdry_batch = sample_out_of_domain(limits, args.batch_size, args.bdry_ratio, device=device)
                ret = lyaloss(adv_batch[0], bdry_batch)
                loss = ret.loss
                if args.l1_lambda > 0:
                    l1_norm = torch.stack([p.abs().sum() for p in main]).sum()
                    total_params = sum(p.numel() for p in main)
                    loss = loss + l1_lambda * (l1_norm / total_params)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
        scheduler.step()
        
        if i % args.log_interval == 0:
            ret = lyaloss(adv_x, bdry_points)
            print(f"Iter {i}, pde_loss={ret.pde_loss.item():.4f}, controller_loss={ret.controller_loss.item():.4f}, "
                    f"data_loss={ret.data_loss.item():.4f}, zero={ret.zero.item():.4f}, bdry_loss={ret.bdry_loss.item():.4f}, "
                    f"cex={ret.num_cex}")

            elapsed_time = time.time() - start_time
            print(f"elapsed time = {elapsed_time}")

        if ((i+1) % args.save_interval == 0):
            save_dir = f'{args.save_dir}'
            os.makedirs(save_dir, exist_ok=True)
            torch.save(lyaloss.state_dict(), f'{save_dir}/{args.system}_{i+1}.pth')

    return ret

if __name__ == "__main__":  
    args = parse_args()
    set_seed(args.seed)
    
    cfg = configs.get(args.system)
    if cfg is None:
        raise ValueError(f"Unknown system {args.system!r}")
    
    dynamics = cfg.dynamics
    controller = cfg.controller.to(device).to(dtype)
    lyapunov_nn = cfg.lyapunov.to(device).to(dtype)
    
    lower_limit, upper_limit = utils.generate_limits(
        cfg.factor_list, cfg.scale_vector, device, dtype
    )
    
    mu = cfg.mu
    p = cfg.p
    dt = cfg.dt
    rollout_steps = cfg.steps
    loss_weights = cfg.loss_weights
    hard_lower = cfg.lower_bound
    hard_upper = cfg.upper_bound

    for i in range(len(lower_limit)):          
        lyaloss = loss.V_Train_Loss(
            dynamics,
            controller,
            lyapunov_nn,
            mu,
            loss_weights,
            upper_limit[i],
            p,
            args.mode,
            args.use_data_loss,
            rollout_steps,
            dt,
        )

        lyaloss.to(device)
        train_full(
            args,
            lyaloss=lyaloss,
            lower_limit=lower_limit[i],
            upper_limit=upper_limit[i],
            box=i,
            hard_lower=hard_lower,
            hard_upper=hard_upper
        )