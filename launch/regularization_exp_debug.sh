#!/bin/bash
#PBS -l select=1:ncpus=10:mpiprocs=10:ngpus=1
#PBS -l walltime=00:10:00
#PBS -N debug_optuna
#PBS -M daniel.ordonez@iit.it
#PBS -m b
#PBS -q debug

cd /work/dordonez/Projects/koopman_robotics
conda activate robotics

python train_observables.py --multirun  exp_name=sampler_test_franklin hydra.sweeper.n_trials=2 debug_loops=True