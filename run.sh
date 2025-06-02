export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export CUDA_VISIBLE_DEVICES=0

# ------------------------- ROA Estimation ---------------------------------
# Van Der Pol
python src/train.py --system "van" --save_dir './van/pretrain/seed_4' \
--mode 'learnable' --learning_rate 0.0001 --max_iter 3000 --batch_size 128 --candidate_size 640 \
--lyapunov_thershold 0.99 --log_interval 10 --update_interval 100 --save_interval 200 --traj_dt 0.01 --traj_steps 8000 --seed 4

# Cartpole
# python src/train_traj_nonsym.py --system "cartpole" \
# --save_dir './models/final_results/cartpole/pretrain/' \
# --mode 'learnable' --learning_rate 0.0001 --max_iter 5000 --batch_size 256 --bdry_ratio 2 --lyapunov_thershold 0.95 \
# --update_interval 500 --save_interval 500 --log_interval 10 --candidate_size 1500 --num_trajectories 3000

# PVTOL
# python src/train_traj_nonsym.py --system "pvtol" \
# --save_dir './models/final_results/pvtol/pretrain/' \
# --mode 'learnable' --learning_rate 0.0001 --max_iter 12000 --batch_size 128 --bdry_ratio 2 --lyapunov_thershold 0.9 \
# --update_interval 1000 --save_interval 1000 --log_interval 100 --traj_dt 0.0003 --traj_steps 200000 \
# --candidate_size 1500 --num_trajectories 3000

# ------------------------- Finetune ---------------------------------
# python src/finetune.py --system "van" --box 6 9.6 \
# --load_dir './van/pretrain/van_3000.pth' \
# --save_dir './van/finetune/van.pth' --batch_size 4096 --lyap_lr 1e-4 \
# --controller_lr 1e-5 --max_iter 2000 --lower_threshold 0.01 --upper_threshold 0.99 --finetune_epochs 10 \
# --pgd_steps 300 --num_points 60000 --consecutive_success 100 --pgd_step_size 0.1 --bdry_scale 1

# ------------------------- Plotting ---------------------------------
# python src/draw.py --system "van" --box 6 9.6 \
# --load_dir './van/finetune/seed_4.pth' \
# --save_dir './van' --plot_idx 0 1 --c1 0.0101 --c2 0.99 \
# --compute_volume --dt 0.001 --simulate_steps 50000 --attack
