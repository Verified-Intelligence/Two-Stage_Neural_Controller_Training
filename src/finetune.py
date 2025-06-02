import torch.nn.functional as F
from torch import nn
from typing import Optional, Tuple
import torch
import numpy as np
import os
import sys
import time
import typing
import argparse
import numpy as np

sys.path.append(os.path.dirname(os.path.realpath(__file__)))
sys.path.append("../torchdyn")

import src.loss as loss
import src.utils as utils
from src.training_config import configs

device = 'cuda'
dtype = torch.float32

def parse_args():
    """
    Parses command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Train a neural Lyapunov function for the Van der Podl dynamics."
    )
    parser.add_argument("--system", type=str, help="Dynamical System.")
    parser.add_argument("--max_iter", type=int, default=100000, help="Maximum number of iterations.")
    parser.add_argument("--lower_threshold", type=float, default=0.01, help="c1.")
    parser.add_argument("--upper_threshold", type=float, default=0.95, help="c2.")
    parser.add_argument("--lyap_lr", type=float, default=1e-4, help="Lyapunov learning rate.")
    parser.add_argument("--controller_lr", type=float, default=1e-4, help="Controller learning rate.")
    parser.add_argument("--pgd_steps", type=int, default=100, help="PGD attack steps.")
    parser.add_argument("--pgd_step_size", type=float, default=0.01, help="PGD attack step size.")
    parser.add_argument("--consecutive_success", type=int, default=500, help="Number of consecutive success before stop.")
    parser.add_argument("--num_points", type=int, default=50000, help="Number of points per iteration.")
    parser.add_argument("--buffer_size", type=int, default=1000000, help="CEX buffer size.")
    parser.add_argument("--finetune_epochs", type=int, default=10, help="Finetune Epochs.")
    parser.add_argument("--batch_size", type=int, default=4096, help="Batch Size.")
    parser.add_argument("--bdry_scale", type=float, default=1.2, help="Which domain to select bdry points")
    parser.add_argument('--box', type=float, nargs='+', default=[1.0], help='Finetuning Box')
    parser.add_argument('--l1_weight', type=float, default=0.0, help='L1 reg')
    parser.add_argument("--load_dir", type=str, help='Pretrained model path')
    parser.add_argument("--save_dir", type=str, help='Save to which directory')
    return parser.parse_args()

def pgd_attack_band_ROA(x_init, lyaloss, pgd_steps=100, step_size=1e-2,
                       clamp_box=None, fraction=0.05, device="cuda",
                       lower_threshold=0.01, upper_threshold=0.8):
    original_requires = {name: param.requires_grad for name, param in lyaloss.named_parameters()}
    for param in lyaloss.parameters():
        param.requires_grad_(False)
    
    # Use a simple optimizer for PGD
    x_adv = x_init.clone().to(device).detach().requires_grad_(True)
    optimizer = torch.optim.Adam([x_adv], lr=step_size)
    
    for step in range(pgd_steps):  
        optimizer.zero_grad()
        
        W_vals = lyaloss.lyapunov(x_adv)   
        sum_W = W_vals.sum()
        gradW = torch.autograd.grad(sum_W, x_adv, create_graph=True)[0]   
        u_vals = lyaloss.controller(x_adv)
            
        f_vals = lyaloss.dynamics.f_torch(x_adv, u_vals)
        dot_term = torch.sum(gradW * f_vals, dim=1, keepdim=True)
        
        stacked = torch.cat([
            dot_term,
            upper_threshold - W_vals,
            W_vals - lower_threshold
        ], dim=1)  # [B, 3]
        
        min_vals, _ = torch.min(stacked, dim=1, keepdim=True)
        loss = -torch.mean(min_vals)
        loss.backward()
        optimizer.step()
        
        if clamp_box is not None:
            lb, ub = clamp_box
            with torch.no_grad():
                x_adv.data = torch.max(torch.min(x_adv, ub), lb)
            x_adv.requires_grad_(True)
        
    x_adv.requires_grad_(False)
    final_W = lyaloss.lyapunov(x_adv)
    valid_mask = ((final_W > lower_threshold) & (final_W < upper_threshold)).squeeze(1)
    x_adv_valid = x_adv[valid_mask].clone()  # Clone to avoid CUDA graph issues
    
    for name, param in lyaloss.named_parameters():
        param.requires_grad_(original_requires[name])
    
    return x_adv_valid

def sample_out_of_domain(upper_limit, nx, n, scale=1.):
    x = (torch.rand(n, nx, device='cuda') * 2 - 1)
    ratios = torch.abs(x) / (upper_limit.unsqueeze(0) * scale)  # Shape: [n, nx]
    xnorm, _ = ratios.max(dim=1, keepdim=True)  # Shape: [n, 1]
    x = x / xnorm
    noise = torch.rand(n, nx, device=x.device) * 0.1
    x = x + torch.sign(x) * noise
    return x

def compute_full_violation(x: torch.Tensor, lyapunov_nn, controller, dynamics):
    x = x.clone()
    x = x.requires_grad_(True)
    
    W_vals = lyapunov_nn(x)
    sum_W = W_vals.sum()
    gradW = torch.autograd.grad(sum_W, x, create_graph=False)[0]

    u_vals = controller(x)
    f_vals = dynamics.f_torch(x, u_vals)

    interior_violation = torch.relu(torch.sum(gradW * f_vals, dim=1))  # [n]
    full_violation = interior_violation
    
    return full_violation

def update_adv_dataset(
    old_buffer: torch.Tensor, new_adv_x: torch.Tensor, loss, buffer_size: int
) -> torch.Tensor:
    new_buffer = torch.cat((new_adv_x, old_buffer), dim=0)
    if new_buffer.shape[0] > buffer_size:            
        violation_on_buffer = compute_full_violation(
            new_buffer, 
            loss.lyapunov,
            loss.controller,
            loss.dynamics,
        )
        _, sorted_indices = torch.sort(violation_on_buffer, descending=True)
        new_buffer = new_buffer[sorted_indices[:buffer_size]]
        
    return new_buffer

def finetune_roa_adaptive_cex_nobdry(
    lyaloss, 
    upper_limit,              # tensor, e.g. shape [nx]
    nx=None,                  # dimensionality of the state (optional, can be inferred)
    lower_threshold=0.05,
    initial_target_roa=0.7,   # Starting ROA target
    max_target_roa=0.95,      # Maximum ROA target to attempt
    expansion_factor=1.1,     # Factor to increase ROA target when no counterexamples found
    consecutive_success=10,   # Number of consecutive iterations without counterexamples before expansion
    max_success_at_max_roa=20, # Consecutive successes at max ROA before stopping
    buffer_size=100000,
    num_candidates=20000,
    sample_fraction=1,
    finetune_epochs=10,
    batch_size=4096,
    lyapunov_lr=1e-4,        
    controller_lr=1e-4,       
    pgd_steps=200,
    pgd_step_size=1e-2,
    max_iter=500,
    bdry_scale=2,
    device='cuda',
    dtype=torch.float32,
    verbose=True
):
    print(dtype)
    if nx is None:
        nx = upper_limit.size(0)
    
    if not torch.is_tensor(upper_limit):
        upper_limit = torch.tensor(upper_limit, device=device, dtype=dtype)
    
    limit = upper_limit
    params_dict = [{"params": list(lyaloss.lyapunov.parameters()), "lr": lyapunov_lr}]
    params_dict.append({
                "params": list(lyaloss.controller.parameters()),
                "lr": controller_lr,
            })
    
    optimizer = torch.optim.Adam(params_dict)
    eq_point = lyaloss.dynamics.x_equilibrium.unsqueeze(0).to(device, dtype=dtype)
    x_adv_buffer = torch.empty((0, nx), device=device)
    
    lower_bound = -limit
    upper_bound = limit
    current_target_roa = initial_target_roa
    success_counter = 0
    max_roa_success_counter = 0
    
    persistent_counter = 0
    
    V_zero_before = lyaloss.lyapunov(eq_point)
    print(V_zero_before)

    # Main iteration loop
    for i in range(max_iter):
        if verbose:
            print(f"Iteration {i+1}/{max_iter} - Target ROA: {current_target_roa:.4f}")
        
        candidate_points = (torch.rand((num_candidates, nx), device=device, dtype=dtype) - 0.5) * 2 * limit
        
        before = time.time()
        potential_adv = pgd_attack_band_ROA(
            candidate_points, 
            lyaloss, 
            pgd_steps=pgd_steps, 
            step_size=pgd_step_size, 
            clamp_box=(lower_bound, upper_bound),
            fraction=sample_fraction, 
            device=device, 
            lower_threshold=lower_threshold,
            upper_threshold=current_target_roa
        )
        print(f"PGD attack time: {time.time() - before:.2f}s")
        
        with torch.enable_grad():
            x_check = potential_adv.clone().detach().requires_grad_(True)
            if x_check.shape[0] == 0:
                num_cex = 0
                x_adv = x_check  # Empty tensor
            else:
                V_check = lyaloss.lyapunov(x_check)
                band_mask = (V_check.squeeze(1) >= lower_threshold) & (V_check.squeeze(1) <= current_target_roa)
                sum_V = V_check.sum()
                gradV = torch.autograd.grad(sum_V, x_check, create_graph=False)[0]
                
                u_check = lyaloss.controller(x_check)
                f_check = lyaloss.dynamics.f_torch(x_check, u_check)
                
                dot_products = torch.sum(gradV * f_check, dim=1)
                cex_mask = (dot_products > 0) & band_mask
                x_adv = x_check[cex_mask]
                num_cex = cex_mask.sum().item()
            
            if verbose:
                print(f"Verified counterexamples: {num_cex}")
        
        if num_cex > 0:
            persistent_counter += 1
        else:
            persistent_counter = 0

        x_adv_buffer = update_adv_dataset(x_adv_buffer, x_adv, lyaloss, buffer_size)
        
        buffer_has_cex = False
        if x_adv_buffer.shape[0] > 0:
            with torch.enable_grad():
                x_buffer_check = x_adv_buffer.clone().detach().requires_grad_(True)
                V_buffer = lyaloss.lyapunov(x_buffer_check)
                band_mask = (V_buffer.squeeze(1) >= lower_threshold) & (V_buffer.squeeze(1) <= current_target_roa)
                sum_V_buffer = V_buffer.sum()
                gradV_buffer = torch.autograd.grad(sum_V_buffer, x_buffer_check, create_graph=False)[0]
                
                u_buffer = lyaloss.controller(x_buffer_check)
                f_buffer = lyaloss.dynamics.f_torch(x_buffer_check, u_buffer)
            
                dot_products_buffer = torch.sum(gradV_buffer * f_buffer, dim=1)
                buffer_violations = (dot_products_buffer > 0) & band_mask
                
                buffer_has_cex = buffer_violations.any().item()
                
                if buffer_has_cex:
                    num_buffer_cex = buffer_violations.sum().item()
                    if verbose:
                        print(f"Buffer still contains {num_buffer_cex} counterexamples.")
        
        if x_adv.shape[0] == 0 and not buffer_has_cex:
            success_counter += 1
            at_max_roa = abs(current_target_roa - max_target_roa) < 1e-5
            
            if at_max_roa:
                max_roa_success_counter += 1
                if verbose:
                    print(f"No counterexamples found at maximum ROA. Counter: {max_roa_success_counter}/{max_success_at_max_roa}")
                
                if max_roa_success_counter >= max_success_at_max_roa:
                    if verbose:
                        print(f"Maximum ROA verified for {max_success_at_max_roa} consecutive iterations. Training complete.")
                    break
            else:
                if verbose:
                    print(f"No counterexamples found. Success counter: {success_counter}/{consecutive_success}")
            
            if success_counter >= consecutive_success:
                old_target = current_target_roa
                current_target_roa = min(current_target_roa * expansion_factor, max_target_roa)
                success_counter = 0  # Reset counter
                
                if verbose:
                    print(f"Target ROA expanded: {old_target:.4f} -> {current_target_roa:.4f}")
                
                if abs(current_target_roa - max_target_roa) < 1e-5:
                    if verbose:
                        print(f"Maximum target ROA {max_target_roa} reached.")

            continue
        else:
            success_counter = 0
            max_roa_success_counter = 0
            
            if verbose:
                print(f"Found counterexamples (new or in buffer). Counters reset.")
        
        x_adv_dataset = torch.utils.data.TensorDataset(x_adv_buffer)
        dataloader = torch.utils.data.DataLoader(
            x_adv_dataset, 
            batch_size=batch_size,
            shuffle=True,
        )
        
        for epoch in range(finetune_epochs):
            epoch_interior_loss = 0.0
            epoch_zero_loss = 0.0
            epoch_bdry_loss = 0.0
            epoch_total_loss = 0.0
            num_batches = 0
            total_cex = 0
            
            for batch in dataloader:
                adv_batch = batch[0].clone().detach().requires_grad_(True)
                bdry_points = sample_out_of_domain(upper_limit, nx, batch_size, scale=bdry_scale)
                
                W_vals = lyaloss.lyapunov(adv_batch)
                band_mask = (W_vals.squeeze(1) >= lower_threshold) & (W_vals.squeeze(1) <= current_target_roa)
                sum_W = W_vals.sum()
                gradW = torch.autograd.grad(sum_W, adv_batch, create_graph=True)[0]
                
                u_vals = lyaloss.controller(adv_batch)
                f_vals = lyaloss.dynamics.f_torch(adv_batch, u_vals)
                
                dot_term = torch.sum(gradW * f_vals, dim=1)
                interior_cex_loss = torch.mean(torch.relu(dot_term))
                
                cex_condition = (dot_term > 0) & band_mask
                n_cex = int(cex_condition.sum().item())
                total_cex += n_cex
                
                if n_cex == 0:
                    continue
                
                zero_tensor = torch.zeros_like(adv_batch[0]).unsqueeze(0).to(device)
                V_zero = lyaloss.lyapunov(zero_tensor)
                zero_loss = torch.maximum(torch.relu(V_zero - lower_threshold), torch.relu(V_zero_before.squeeze().item() - V_zero))
                
                W_vals_bdry = lyaloss.lyapunov(bdry_points)
                bdry_loss = F.mse_loss(W_vals_bdry, torch.full_like(W_vals_bdry, 1.0))
                
                interior_weight = 1.0
                zero_weight = 10.0
                bdry_weight = 0.1
                
                finetune_loss = (
                    interior_weight * interior_cex_loss +
                    zero_weight * zero_loss +
                    bdry_weight * bdry_loss
                )
                
                optimizer.zero_grad()
                finetune_loss.backward()
                torch.nn.utils.clip_grad_norm_(lyaloss.parameters(), 1.0)
                optimizer.step()
                
                epoch_interior_loss += interior_cex_loss.item()
                epoch_zero_loss += zero_loss.item()
                epoch_bdry_loss += bdry_loss.item()
                epoch_total_loss += finetune_loss.item()
                num_batches += 1
            
            if num_batches > 0:
                avg_interior_loss = epoch_interior_loss / num_batches
                avg_zero_loss = epoch_zero_loss / num_batches
                avg_bdry_loss = epoch_bdry_loss / num_batches
                avg_total_loss = epoch_total_loss / num_batches
                
                x_eval = x_adv_buffer.clone().detach().requires_grad_(True)
                V_eval = lyaloss.lyapunov(x_eval)
                band_mask = (V_eval.squeeze(1) >= lower_threshold) & (V_eval.squeeze(1) <= current_target_roa)
                sum_V = V_eval.sum()
                gradV = torch.autograd.grad(sum_V, x_eval, create_graph=True)[0]

                u_eval = lyaloss.controller(x_eval)
                f_eval = lyaloss.dynamics.f_torch(x_eval, u_eval)

                derivative_eval = torch.sum(gradV * f_eval, dim=1)
                violations = (derivative_eval > 0) & band_mask
                num_cex = violations.sum().item()
                cex_percentage = (num_cex / x_eval.size(0)) * 100 if x_eval.size(0) > 0 else 0
                
                avg_W = V_eval[violations].mean().item()
                std_W = V_eval[violations].std().item()

                if verbose:
                    print(f"Epoch {epoch+1}/{finetune_epochs}: Buffer size: {x_eval.size(0)}")
                    print(f"  Current counterexamples: {num_cex} ({cex_percentage:.2f}%)")
                    print(f"  Losses: interior={avg_interior_loss:.6f}, zero={avg_zero_loss:.6f}, "
                         f"bdry={avg_bdry_loss:.6f}, total={avg_total_loss:.6f}, Avg V={avg_W:.4f}, Std V={std_W:.4f}")
            else:
                if verbose:
                    print(f"Epoch {epoch+1}/{finetune_epochs}: No batches processed")
    
    if verbose:
        print(f"\nTraining complete. Final verified target ROA: {current_target_roa:.4f}")
    
    return lyaloss
        
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
    
    lower_limit, upper_limit = utils.generate_limits(
        [1], args.box, device, dtype
    )
        
    for i in range(len(lower_limit)):        
        lyaloss = loss.V_Train_Loss(
            dynamics,
            controller,
            lyapunov_nn,
            mu,
            loss_weights,
            upper_limit[i],
            p,
            'learnable',
            True,
            rollout_steps,
            dt
        )
        
        model = torch.load(args.load_dir)
        for key in list(model.keys()):
            if key in ["log_sigma_cv", "log_sigma_cv_ctrl", "log_sigma_sparse", "log_curvature", "log_u"]:
                model.pop(key)
                
        if 'state_dict' in model.keys():
            model = model['state_dict']
        
        lyaloss.load_state_dict(model)
        lyaloss.to(device).to(dtype)
        
        start_time = time.time()
        new_lyaloss = finetune_roa_adaptive_cex_nobdry(lyaloss, upper_limit=upper_limit[i], nx=dynamics.x_equilibrium.size(0), max_iter=args.max_iter, lower_threshold=args.lower_threshold, initial_target_roa=args.upper_threshold, max_target_roa=args.upper_threshold,
                                                       lyapunov_lr=args.lyap_lr, controller_lr=args.controller_lr, pgd_steps=args.pgd_steps, consecutive_success=args.consecutive_success, max_success_at_max_roa=args.consecutive_success, bdry_scale=args.bdry_scale,
                                                       num_candidates=args.num_points, buffer_size=args.buffer_size, pgd_step_size=args.pgd_step_size, finetune_epochs=args.finetune_epochs, expansion_factor=1.5, batch_size=args.batch_size)
       
        print(f'Time Used for Fintuning: {time.time() - start_time}')
        os.makedirs(os.path.dirname(args.save_dir), exist_ok=True)
        
        # Filter to only save controller and lyapunov state dict entries
        full_state = new_lyaloss.state_dict()
        filtered_state = OrderedDict()

        for k, v in full_state.items():
            if k.startswith("controller.") or k.startswith("lyapunov."):
                filtered_state[k] = v

        # Save only the filtered state dict
        os.makedirs(os.path.dirname(args.save_dir), exist_ok=True)
        torch.save({'state_dict': filtered_state, 'domain': upper_limit[i]}, args.save_dir)
        