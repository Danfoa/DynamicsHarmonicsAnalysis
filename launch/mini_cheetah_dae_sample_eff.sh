#!/bin/bash
#PBS -l select=1:ncpus=40:mpiprocs=40:ngpus=4
#PBS -l walltime=24:00:00
#PBS -N mini_sample_eff
#PBS -M daniel.ordonez@iit.it
#PBS -m bea
#PBS -q R143910
#PBS -j oe

cd /work/dordonez/Projects/koopman_robotics
conda activate robotics

seeds="1,2,3,4"
exp_name="mini_sample_eff_terrain"

shared_params="exp_name=${exp_name} system.terrain=flat,uneven_easy model.obs_pred_w=1.0 model.num_hidden_units=256 seed=${seeds} model.activation=ReLU model.batch_size=512"
hydra_params="hydra.launcher.n_jobs=4"

python train_observables.py --multirun device=0 model=dae,edae     system=mini_cheetah    system.train_ratio=0.1,0.25,0.5,0.75,1.0 "${shared_params}" "${hydra_params}" &
python train_observables.py --multirun device=1 model=dae-aug      system=mini_cheetah    system.train_ratio=0.1,0.25,0.5,0.75,1.0 "${shared_params}" "${hydra_params}" &
python train_observables.py --multirun device=2 model=dae-aug,edae system=mini_cheetah-k4 system.train_ratio=0.1,0.25,0.5,0.75,1.0 "${shared_params}" "${hydra_params}" &
python train_observables.py --multirun device=4 model=dae-aug,edae system=mini_cheetah-c2 system.train_ratio=0.1,0.25,0.5,0.75,1.0 "${shared_params}" "${hydra_params}" &

wait