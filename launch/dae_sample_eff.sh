#!/bin/bash
#PBS -l select=1:ncpus=20:mpiprocs=20:ngpus=2
#PBS -l walltime=24:00:00
#PBS -N dae_sample_eff
#PBS -M daniel.ordonez@iit.it
#PBS -m bea
#PBS -q gpu_a100
#PBS -j oe

cd /work/dordonez/Projects/koopman_robotics
conda activate robotics

model="dae"
seeds="0,1,2,3"

shared_params="exp_name=SampleEff model=${model} system=linear_system seed=${seeds} system.n_constraints=1 system.obs_state_ratio=3 system.state_dim=50"
hydra_params="hydra.launcher.n_jobs=10"

python train_observables.py --multirun device=0 system.group=C5 system.train_ratio=0.1,0.25,0.5,0.75,1.0 ${hydra_params} ${shared_params} &
python train_observables.py --multirun device=1 system.group=C10 system.train_ratio=0.1,0.25,0.5,0.75,1.0 ${hydra_params} ${shared_params} &