#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 13/5/22
# @Author  : Daniel Ordonez
# @email   : daniels.ordonez@gmail.com
import glob
import io
from pathlib import Path
import sys
import time

import morpho_symm
import numpy as np
import pandas as pd
import PIL
import scipy
import sklearn
import torch
import torch.nn.functional as F
from escnn.group import directsum
from omegaconf import DictConfig, OmegaConf, omegaconf
from sklearn.metrics import jaccard_score, precision_score

from morpho_symm.utils.robot_utils import load_symmetric_system

from sklearn.metrics import confusion_matrix
from torch.utils.data._utils.collate import default_collate
from tqdm import tqdm

from data.DynamicsRecording import DynamicsRecording
from utils.mysc import format_si


def mat_to_dynamics_recordings(data_path: Path,
                               partitions_ratio=(0.7, 0.15, 0.15),
                               partitions_name=("train", "val", "test")):
    robot, G = load_symmetric_system(robot_name='mini_cheetah')
    # rep_QJ = G.representations['Q_js']
    rep_TqJ = G.representations['TqQ_js']
    # rep_Ed = G.representations['Ed']
    rep_Rd = G.representations['Rd']
    rep_Rd_pseudo = G.representations['Rd_pseudo']

    data_files = list(data_path.glob("*.mat"))
    assert len(data_files) > 1, f"No .mat files found in {data_path.absolute()}"
    assert sum(partitions_ratio) <= 1.0, "the partitions should add up to less than 100% of the data"

    print(f"Loading data from {data_path}")
    print(f"Dataset .mat files found: {[Path(d).name for d in data_files]}")
    # for all dataset in the folder
    all_samples = 0
    for data_name in data_path.glob('*.mat'):
        # load data
        raw_data = scipy.io.loadmat(data_name)
        # Extract the system's measurements/observations
        contacts = raw_data['contacts']  # 4 dim binary vector: [right_front, left_front, right_hind, left_hind]
        q = raw_data['q']  # Joint positions
        v = raw_data['qd']  # Joint Velocities
        # feet_pos = raw_data['p']  # Feet positions  [x,y,z] w.r.t base frame
        feet_vel = raw_data['v']  # Feet velocities [dx, dy, dz] w.r.t. base frame
        imu_acc = raw_data['imu_acc']
        imu_ang_vel = raw_data['imu_omega']
        imu_ori = raw_data['imu_rpy']  # Roll, pitch, yaw
        torques = raw_data['tau_est']  # Joint torques
        contact_forces = raw_data['F']  # Estimated contact forces
        time = raw_data['control_time'][0]
        dt = np.mean(time[1:] - time[:-1])

        # Define the representations for each measurement/observation.
        # Joint-space configurations
        rep_q = rep_v = rep_torques = rep_TqJ
        # Vectors in 3D.
        rep_imu_acc = rep_Rd
        rep_imu_ori = rep_imu_ang_vel = rep_Rd_pseudo
        rep_feet_vel = rep_contact_forces = directsum([rep_Rd] * 4, name="Legs-3D")  # 4 3D vectors

        # Save recording partitioning trajectories
        num_samples = q.shape[0]
        min_idx = 0
        target_path = data_path.parent / 'recordings' / data_name.stem
        target_path.mkdir(parents=True, exist_ok=True)
        for partition, ratio in zip(partitions_name, partitions_ratio):
            max_idx = min_idx + int(num_samples * ratio)
            idx = range(min_idx, max_idx)
            partition_num_samples = max_idx - min_idx
            min_idx = min_idx + int(num_samples * ratio)
            data = DynamicsRecording(
                description=f"Mini Cheetah {Path(data_name).stem}",
                info=dict(num_traj=1, trajectory_length=q.shape[0]),
                dynamics_parameters=dict(dt=dt, group=dict(group_name=G.name, group_order=G.order())),
                recordings=dict(q=q[None, idx].astype(np.float32),
                                v=v[None, idx].astype(np.float32),
                                torques=torques[None, idx].astype(np.float32),
                                imu_acc=imu_acc[None, idx].astype(np.float32),
                                imu_ori=imu_ori[None, idx].astype(np.float32),
                                imu_ang_vel=imu_ang_vel[None, idx].astype(np.float32),
                                feet_vel=feet_vel[None, idx].astype(np.float32),
                                contact_forces=contact_forces[None, idx].astype(np.float32)),
                state_obs=['q', 'v'],
                action_obs=['torques'],
                obs_representations=dict(q=rep_q, v=rep_v, torques=rep_torques, imu_acc=rep_imu_acc,
                                         imu_ori=rep_imu_ori, imu_ang_vel=rep_imu_ang_vel,
                                         feet_vel=rep_feet_vel, contact_forces=rep_contact_forces))

            data.save_to_file(target_path / f"n_trajs=1-frames={format_si(partition_num_samples)}-{partition}.pkl")
        print(f"Saved recording {data_name.stem} data to {target_path}")

if __name__ == "__main__":
    data_path = Path(__file__).parent.absolute() / 'mat'

    mat_to_dynamics_recordings(data_path=data_path)
