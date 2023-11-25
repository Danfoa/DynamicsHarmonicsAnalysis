#!/bin/bash
#PBS -l select=1:ncpus=20:mpiprocs=20:ngpus=2
#PBS -l walltime=00:01:00
#PBS -N edae_state_dim
#PBS -M daniel.ordonez@iit.it
#PBS -m bea
#PBS -q R143910
#PBS -j oe

echo $CUDA_VISIBLE_DEVICES