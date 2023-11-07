#!/bin/bash
#PBS -l select=1:ncpus=15:mpiprocs=15:ngpus=1
#PBS -l walltime=24:00:00
#PBS -N dae_sample_eff
#PBS -M daniel.ordonez@iit.it
#PBS -m bea
#PBS -q gpu_a100
#PBS -j oe

cd /work/dordonez/Projects/koopman_robotics
conda activate robotics

python train_observables.py --multirun exp_name=C3-Constraints-SampleEff hydra.launcher.n_jobs=15 model=dae system.n_constraints=1 system.group=C3 system.state_dim=30 system.obs_state_ratio=3 system.train_ratio=0.1,0.25,0.5,0.75,1.0 seed=0,1,2,3 device=0