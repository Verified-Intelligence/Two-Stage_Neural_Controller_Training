# Two-Stage Learning of Stabilizing Neural Controllers via Zubov Sampling and Iterative Domain Expansion

Our work studies the synthesis of neural controllers with better stability guarantees for continuous-time systems. We propose a novel two-stage training framework to jointly synthesize the controller and Lyapunov function for continuous-time systems. By leveraging a Zubov‑inspired region of attraction characterization to directly estimate stability boundaries, we propose a novel training data sampling strategy and a domain updating mechanism that significantly reduces the conservatism in training. Moreover, we extend the state-of-the-art neural network verifiers [`alpha-beta-CROWN`](https://abcrown.org) with the capability of efficiently handling the Jacobian of many previous unsupported operators and propose a novel verification scheme that avoid expensive bisection. We show that our training can yield a region of attraction with volume $5 - 1.5\cdot 10^{5}$ times larger compared to the baseline, and our verification on continuous systems can be up to $40-10000$ times faster compared to the traditional SMT solver dReal.

## Setup

The environment can be set up with the following commands.

```bash
conda env create -f environment.yaml --name zubov
conda activate zubov
```

The path can be set up with ```export PYTHONPATH="${PYTHONPATH}:$(pwd)"```

## Training

### ROA Estimation Stage

The ROA estimation stage is implemented in ```train.py```. An example training the Van Der Pol system will be

```bash
python src/train.py --system "van" --save_dir './van/pretrain/' \
--mode 'learnable' --learning_rate 0.0001 --max_iter 3000 --batch_size 128 --candidate_size 640 \
--lyapunov_thershold 0.99 --log_interval 10 --update_interval 100 --save_interval 200 --traj_dt 0.01 --traj_steps 8000
```

For more information regarding the arguments, please refer to ```train.py```. More examples on higher dimensional systems can be found in ```run.sh```. 

### CEGIS Refinement Stage

The CEGIS refinement stage is implemented in ```finetune.py```. An example for the Van Der Pol system model that just gone through the ROA estimation stage with the above command shall be

```bash
python src/finetune.py --system "van" --box 6 9.6 \
--load_dir './van/pretrain/van_3000.pth' \
--save_dir './van/finetune/van.pth' --batch_size 4096 --lyap_lr 1e-4 \
--controller_lr 1e-5 --max_iter 2000 --lower_threshold 0.01 --upper_threshold 0.99 --finetune_epochs 10 \
--pgd_steps 300 --num_points 60000 --consecutive_success 100 --pgd_step_size 0.1 --bdry_scale 1
```

The ```lower_threshold``` and ```upper_threshold``` corresponds to $c_1, c_2$ in paper and needs to be adjusted based on the actual model. Typically we choose it to be around $V(0) + 0.001$. Moreover, the box argument needs to be adjusted to agree with the actual training domain from the ROA estimation stage. For more information on the rest arguments, please refer to ```finetune.py```.

### Adding New System
All dynamical systems currently supported are implemented in ```dynamics.py```, and most training configs can be found in ```training_config.py```. To add a new system, simply add the new system implementation and an corresponding entry in the config file.

### Checkpoints
We provided a series of checkpoints for all the systems, which can be found under ```models```. All models except those for 3D Quadrotor has been finetuned with the CEGIS refinement. 

## Verification
We use a state-of-the-art neural network verifier [`alpha-beta-CROWN`](https://abcrown.org) to formally verify the Lyapunov condition of our trained controllers and Lyapunov functions. To verify the Lyapunov condition within a levelset bounded by $c_1, c_2$, run the following command:
```bash
bash src/verification/run_verify.sh <system_name> <c1> <c2> <path_to_model>
```
For example, to verify the first model trained on the Van Der Pol system, one can run:
```bash
bash src/verification/run_verify.sh van 0.0106 0.989 src/models/van/finetune/seed_0.pth
```
A ```vnnlib``` specification file will be generated in [`src/verification/van`](src/verification/van), and the alpha-beta-CROWN verifier will be executed. The verification result will be appear in the output as:
```
############# Summary #############
Final verified acc: 100.0% (total 1 examples)
Problem instances count: 1 , total verified (safe/unsat): 1 , total falsified (unsafe/sat): 0 , timeout: 0
```
indicating that the property has been successfully verified.

**Note:**
- For the systems discussed in our paper, the corresponding box ranges are predefined in [`run_verify.sh`](src/verification/run_verify.sh) under the ```RADIUS_MAP``` dictionary. If you add new systems, you will also need to specify their box ranges in that script.
- The values of ```c1``` and ```c2``` are estimated based on training results. However, as mentioned in the paper, even if no prior values for ```c1``` and ```c2``` are available (e.g., setting them to 0 and 1 repestively), our verification algorithm can adaptively refine these thresholds and identify the optimal ones.
- Our verification configurations has been tested on a GPU with 32GB memory. If you are using a GPU with less memory, consider reducing the batch size of verification by modifying the ```batch_size``` parameter in the configuration files to avoid out-of-memory errors.



## Check and Plot

The final resulting controller/Lyapunov function can be checked and visualized with ```draw.py```, which provides scripts for trajectory based verification and a PGD attack based verification. An example for the same Van Der Pol system will be

```bash
python src/draw.py --system "van" --box 6 9.6 \
--load_dir './van/finetune/van.pth' \
--save_dir './van' --plot_idx 0 1 --c1 0.0101 --c2 0.99 \
--compute_volume --dt 0.001 --simulate_steps 50000 --attack
```

In this command, ```--attack``` indicates whether a PGD based checking will be conducted. The ```--dt``` and ```--simulate_steps``` indicate the hyperparameter used for trajectory based verification. Moreover, if ```--compute_volume``` is used, the ROA volume will be computed with a sampling based approach.
