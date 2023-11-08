#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 13/5/22
# @Author  : Daniel Felipe Ordoñez Apraez
# @email   : daniels.ordonez@gmail.com
from pathlib import Path

import numpy as np
import pybullet
import scipy
from escnn.group import directsum
from morpho_symm.utils.robot_utils import load_symmetric_system
from pybullet_utils.bullet_client import BulletClient

from data.DynamicsRecording import DynamicsRecording
from utils.mysc import format_si


def mat_to_dynamics_recordings(data_path: Path,
                               partitions_ratio=(0.7, 0.15, 0.15),
                               partitions_name=("train", "val", "test")):
    robot_name = 'mini_cheetah'
    robot, G = load_symmetric_system(robot_name=robot_name)
    # For God’s sake, we need to avoid using PyBullet.
    bullet_client = BulletClient(connection_mode=pybullet.DIRECT)
    robot.configure_bullet_simulation(bullet_client=bullet_client)
    rep_Q_js = G.representations['Q_js']
    rep_TqQ_js = G.representations['TqQ_js']
    # rep_Ed = G.representations['Ed']
    rep_Rd = G.representations['Rd']
    rep_Rd_pseudo = G.representations['Rd_pseudo']
    # To enable all the symmetries of the system we need to offset the zero reference of the joint positions.
    q0, _ = robot.pin2sim(robot._q0, np.zeros(robot.nv))


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
        q_js_t = raw_data['q']  # Joint positions
        v_js_t = raw_data['qd']  # Joint Velocities
        # Convert the joint positions to a symmetry-enabled reference frame, and the appropriate joint parametrization.
        q_js_ms = q_js_t + q0[7:]  # Add offset to the measurements from UMich
        cos_q_js, sin_q_js = np.cos(q_js_ms), np.sin(q_js_ms)  # convert from angle to unit circle parametrization
        # Define joint positions [q1, q2, ..., qn] -> [cos(q1), sin(q1), ..., cos(qn), sin(qn)] format.
        q_js_unit_circle_t = np.stack([cos_q_js, sin_q_js], axis=2)
        q_js_unit_circle_t = q_js_unit_circle_t.reshape(q_js_unit_circle_t.shape[0], -1)
        assert np.allclose((cos_q_js[0, 0], sin_q_js[0, 0]), q_js_unit_circle_t[0][:2])

        # feet_pos = raw_data['p']  # Feet positions [x, y, z] w.r.t. each leg frame (no idea orientation of these)
        # feet_vel = raw_data['v']  # Feet velocities [dx, dy, dz] w.r.t. each leg frame
        # imu_acc = raw_data['imu_acc']
        imu_ang_vel = raw_data['imu_omega']
        imu_ori = raw_data['imu_rpy']  # Roll, pitch, yaw
        torques = raw_data['tau_est']  # Joint torques
        # contact_forces = raw_data['F']  # Estimated contact forces
        time = raw_data['control_time'][0]
        dt = np.mean(time[1:] - time[:-1])

        # Define the representations for each measurement/observation.
        # Vectors in 3D.
        # rep_imu_acc = rep_Rd
        rep_imu_ori = rep_imu_ang_vel = rep_Rd_pseudo
        # rep_feet_vel = rep_contact_forces = directsum([rep_Rd] * 4, name="Legs-3D")  # 4 3D vectors

        # Save recording partitioning trajectories
        num_samples = q_js_t.shape[0]
        min_idx = 0
        target_path = data_path.parent / 'recordings' / data_name.stem
        target_path.mkdir(parents=True, exist_ok=True)

        train_moments = None
        for partition, ratio in zip(partitions_name, partitions_ratio):
            max_idx = min_idx + int(num_samples * ratio)
            idx = range(min_idx, max_idx)
            partition_num_samples = max_idx - min_idx
            min_idx = min_idx + int(num_samples * ratio)

            data_recording = DynamicsRecording(
                description=f"Mini Cheetah {Path(data_name).stem}",
                info=dict(num_traj=1, trajectory_length=q_js_t.shape[0]),
                dynamics_parameters=dict(dt=dt, group=dict(group_name=G.name, group_order=G.order())),
                recordings=dict(q_js=q_js_unit_circle_t[None, idx].astype(np.float32),
                                v_js=v_js_t[None, idx].astype(np.float32),
                                torques=torques[None, idx].astype(np.float32),
                                # imu_acc=imu_acc[None, idx].astype(np.float32),
                                imu_ori=imu_ori[None, idx].astype(np.float32),
                                imu_ang_vel=imu_ang_vel[None, idx].astype(np.float32),
                                # feet_vel=feet_vel[None, idx].astype(np.float32),
                                # contact_forces=contact_forces[None, idx].astype(np.float32)
                                ),
                state_obs=('q_js', 'v_js', 'imu_ang_vel'),
                action_obs=('torques',),
                obs_representations=dict(q_js=rep_Q_js,
                                         v_js=rep_TqQ_js,
                                         torques=rep_TqQ_js,
                                         # imu_acc=rep_imu_acc,
                                         imu_ori=rep_imu_ori,
                                         imu_ang_vel=rep_imu_ang_vel,
                                         # feet_vel=rep_feet_vel,
                                         # contact_forces=rep_contact_forces
                                         ),
                # Ensure the angles in the unit circle are not disturbed by the normalization.
                obs_moments=dict(q_js=(np.zeros(q_js_unit_circle_t.shape[-1]), np.ones(q_js_unit_circle_t.shape[-1]))),
                )

            if partition == "train":
                for obs_name in data_recording.recordings.keys():
                    if obs_name in data_recording.obs_moments:
                        continue
                    data_recording.compute_obs_moments(obs_name=obs_name)
                train_moments = data_recording.obs_moments
            else:
                data_recording.obs_moments = train_moments
                # Do "Hard" data-augmentation, as we want to evaluate the capacity of the models to predict the
                # physics of the dynamics of the system. Although data comes from a single trajectory, because of the
                # equivariance of Newtonian physics, the models should be able to predict the dynamics of the system
                # for symmetric trajectories.
                for obs_name in data_recording.recordings.keys():
                    obs_rep = data_recording.obs_representations[obs_name]
                    obs_traj = data_recording.recordings[obs_name]
                    orbit = [obs_traj]
                    for g in G.elements:
                        if g == G.identity: continue  # Already added
                        orbit.append(np.einsum('...ij,...j->...i', obs_rep(g), obs_traj))
                    obs_traj_orbit = np.concatenate(orbit, axis=0)
                    data_recording.recordings[obs_name] = obs_traj_orbit

            data_recording.save_to_file(target_path / f"n_trajs=1-frames={format_si(partition_num_samples)}-{partition}.pkl")
        print(f"Saved recording {data_name.stem} data to {target_path}")

if __name__ == "__main__":
    data_path = Path(__file__).parent.absolute() / 'mat'

    mat_to_dynamics_recordings(data_path=data_path)
