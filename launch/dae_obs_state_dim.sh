#!/bin/bash
#PBS -l select=1:ncpus=16:mpiprocs=16:ngpus=2
#PBS -l walltime=12:00:00
#PBS -N dae_obs_state_dim
#PBS -M daniel.ordonez@iit.it
#PBS -m bea
#PBS -q gpu_a100
#PBS -j oe

cd /work/dordonez/Projects/koopman_robotics
conda activate robotics

python train_observables.py --multirun exp_name=C10-ObsStateDim hydra.launcher.n_jobs=-1 model=dae system.n_constraints=1 system.group=C10 system.state_dim=30 system.obs_state_ratio=1,4 system.train_ratio=1.0 seed=0,1,2,3 device=0 &
python train_observables.py --multirun exp_name=C10-ObsStateDim hydra.launcher.n_jobs=-1 model=dae system.n_constraints=1 system.group=C10 system.state_dim=30 system.obs_state_ratio=2,3 system.train_ratio=1.0 seed=0,1,2,3 device=0 &
wait