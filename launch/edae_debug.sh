#!/bin/bash
#PBS -l select=1:ncpus=1:mpiprocs=1:ngpus=0
#PBS -l walltime=00:05:00
#PBS -N edae_state_dim
#PBS -M daniel.ordonez@iit.it
#PBS -m bea
#PBS -q debug
#PBS -j oe

cd /work/dordonez/Projects/koopman_robotics
#conda activate robotics

echo "Running edae_state_dim.sh"
