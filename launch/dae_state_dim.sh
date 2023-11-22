#!/bin/bash
#PBS -l select=1:ncpus=20:mpiprocs=20:ngpus=2
#PBS -l walltime=24:00:00
#PBS -N dae_state_dim
#PBS -M daniel.ordonez@iit.it
#PBS -m bea
#PBS -q gpu_a100
#PBS -j oe

cd /work/dordonez/Projects/koopman_robotics
conda activate robotics

#python train_observables.py --multirun seed=0,1,2,3 device=0 exp_name=C10-StateDim2 hydra.launcher.n_jobs=-1 model=dae system.n_constraints=1 system.group=C10 system.state_dim=10,70 system.obs_state_ratio=3 system.train_ratio=1.0  &
#python train_observables.py --multirun seed=0,1,2,3 device=1 exp_name=C10-StateDim2 hydra.launcher.n_jobs=4 model=dae system.n_constraints=1 system.group=C10 system.state_dim=30,50 system.obs_state_ratio=3 system.train_ratio=1.0   &
python train_observables.py --multirun seed=0,1,2,3 device=0 exp_name=C10-StateDim2 hydra.launcher.n_jobs=-1 model=dae system.n_constraints=1 system.group=C5 system.state_dim=10,70 system.obs_state_ratio=3 system.train_ratio=1.0  &
python train_observables.py --multirun seed=0,1,2,3 device=1 exp_name=C10-StateDim2 hydra.launcher.n_jobs=4 model=dae system.n_constraints=1 system.group=C5 system.state_dim=30,50 system.obs_state_ratio=3 system.train_ratio=1.0   &
wait