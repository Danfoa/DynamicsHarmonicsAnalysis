import copy
import pickle
from collections.abc import Iterable
from math import isnan, ceil
from pathlib import Path
import random
from typing import Union

import torch
from pinocchio import RobotWrapper
from torch.utils.data import DataLoader
from torch.nn import Module

import numpy as np
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, default_collate
from tqdm import tqdm

from nn.DynamicsAutoencoder import UNBOUND_REVOLUTE
# TODO: dependencies to update.
from src.RobotEquivariantNN.utils.utils import dense

import logging
log = logging.getLogger(__name__)

STATES, CTRLS = "states", "ctrls"
STATES_OBS = "z"
FULL_TRAJ = 1.0

class ClosedLoopDynDataset(Dataset):

    def __init__(self, path: Union[Path, Iterable], pred_horizon=1, compute_Q_func=False, input_frames=1,
                 output_frames=1, normalize=True, state_scaler: StandardScaler = None,
                 ctrl_scaler: StandardScaler = None, device='cpu', augment=False, rep_state=None, rep_ctrl=None,
                 robot: RobotWrapper=None):

        self._data = []
        paths = [path] if isinstance(path, Path) else path
        for p in paths:
            assert p.exists(), f"File not found {p.absolute()}"
            with open(p, "rb") as file_handle:
                self._data.append(pickle.load(file_handle))

        self._num_terminal_states = 1  # Num of terminal states without control action.

        self.device = device
        self.pred_horizon = pred_horizon
        self.robot = robot
        self.normalize = normalize
        self.standard_scaler = {STATES: StandardScaler(copy=False), CTRLS: StandardScaler(copy=False)}
        if state_scaler is not None and ctrl_scaler is not None:
            self.standard_scaler[STATES] = copy.deepcopy(state_scaler)
            self.standard_scaler[CTRLS] = copy.deepcopy(ctrl_scaler)

        # Deal with Symmetry representations
        self.augment = augment
        self.rep_state, self.rep_ctrl = rep_state, rep_ctrl
        self.group_actions, self.troch_group_actions = None, None
        if self.rep_state is not None:
            G, G_torch = [], []
            for g_s, g_u in zip(self.rep_state.G.discrete_actions, self.rep_ctrl.G.discrete_actions):
                G_torch.append((torch.tensor(np.asarray(dense(g_s))).to(device),
                                torch.tensor(np.asarray(dense(g_u))).to(device)))
                G.append((np.asarray(dense(g_s)), np.asarray(dense(g_u))))
            self.group_actions = G
            self.torch_group_actions = G_torch

        # Process data
        states, ctrls, costs, sample_ids, trajs_lengths = self.process_trajectory_data()

        # TODO: Switch to hugging face API
        self.states = torch.from_numpy(states).type('torch.FloatTensor').to(device)
        self.ctrls = torch.from_numpy(ctrls).type('torch.FloatTensor').to(device)
        self._sample_ids = sample_ids
        self._trajs_lengths = trajs_lengths

        # TODO:
        self._compute_Q_func = compute_Q_func
        self._in_frames = input_frames
        self._out_frames = output_frames

    def __len__(self):
        return len(self._sample_ids)

    def __getitem__(self, sample_num):
        """
        Return dataset sample
        :param sample_num: Id of trajectory of self.window_size in the history of data
        :return: dict with entries:
            x: State trajectory history. Tensor of dimension (W,D): W: window_size, D: state dimensions.
            x_next: State trajectory history after x. Tensor of dimension (W,D)
            u: Control trajectory history. Tensor of dimension (W,C): W: window_size, C: control dimensions.
            u_next: Control trajectory history after x. Tensor of dimension (W,C).
        """
        pred_horizon = self._trajs_lengths[sample_num]

        sample_idx = self._sample_ids[sample_num]
        return {STATES: self.states[sample_idx:sample_idx + pred_horizon, :],
                CTRLS: self.ctrls[sample_idx:sample_idx + pred_horizon, :],
        }

    def collate_fn(self, batch):
        # Enforce data type in batched array
        # Small hack to do batched augmentation. TODO: Although efficient this should be done somewhere else.
        collated_batch = default_collate(batch)

        if self.augment and self.group_actions is not None:  # Sample uniformly among symmetry actions including identity
            X_batch, U_batch = collated_batch[STATES], collated_batch[CTRLS]
            g_x, g_u = random.choice(self.group_actions)
            g_X_batch = torch.matmul(X_batch.unsqueeze(1), g_x.unsqueeze(0).to(X_batch.dtype)).squeeze()
            g_U_batch = torch.matmul(U_batch.unsqueeze(1), g_u.unsqueeze(0).to(U_batch.dtype)).squeeze()
            return {STATES: g_X_batch, CTRLS: g_U_batch}
        return collated_batch

    def process_trajectory_data(self):
        """
        We assume the data provided is a sequence of trajectories of the close loop dynamics of the system, with
        possible different trajectory lengths. Depending on the window size selected for training, each data sample
        from the dataset is a trajectory of `window_size` taken from a particular timestep of one of the recorded full
        trajectories.
        """
        self._trajs = np.concatenate([data['trajs'] for data in self._data])
        dt = [data['traj_params']['dt'] for data in self._data]
        assert np.unique(dt).size == 1, f"Trajectories should have the same dt"
        self.dt = dt[0]

        num_timesteps = 0
        sample_ids, traj_lengths = [], []
        # state, next state, control, cost.
        states, ctrls, costs = [], [], []
        # Each recorded trajectory is
        for traj in tqdm(self._trajs, desc="Processing trajectories", disable=log.level != logging.DEBUG):
            traj_length = traj[STATES].shape[0] - self._num_terminal_states
            ctrl_length = traj[CTRLS].shape[0]
            # Terminal states do not have control action
            assert traj_length == ctrl_length, f"We assume OC trajectories"

            if isinstance(self.pred_horizon, float):
                pred_horizon = ceil((traj_length - 1) * self.pred_horizon)
            else:
                if self.pred_horizon >= traj_length:
                    log.warning(f"Requested prediction horizon frames {self.pred_horizon} exceed trajectory length "
                                f"{traj_length}. Using {traj_length} instead.")
                pred_horizon = min(self.pred_horizon, traj_length - 1)
            assert pred_horizon <= traj_length, f"Window size should be smaller or equal than trajectory length"

            # Store in dataset the portion of initial states of a trajectory, which are at least `pred_horizon` samples
            # From the end of the trajectory. So we don't gen training samples containing data from two distinct trajs.
            states.append(traj["states"][:traj_length])
            ctrls.append(traj[CTRLS])
            costs.append(traj['cost'][:traj_length])
            # Number of training samples of this trajectory of the dynamical system
            num_samples = traj_length - pred_horizon
            # Store variable to index these trajectory states while querying data points
            sample_ids.append(range(num_timesteps, num_timesteps + num_samples))
            # Store the length of the trajectory associated with each data point.
            traj_lengths.extend([pred_horizon] * num_samples)
            num_timesteps += traj_length
        states = np.vstack(states)
        ctrls = np.vstack(ctrls)
        costs = np.concatenate(costs)
        sample_ids = np.hstack(sample_ids)
        traj_lengths = np.array(traj_lengths)

        assert np.max(sample_ids) < states.shape[0], f"There is an error counting the timesteps or samples per traj"

        # TODO: Allow for delayed coordinates and longer forcasting outputs
        # TODO: Estimate Q and V
        if self.normalize:
            first_call = not hasattr(self.standard_scaler[STATES], "n_samples_seen_")
            if first_call:
                self.standard_scaler[STATES].fit(states)
                self.standard_scaler[CTRLS].fit(ctrls)
                log.info(f"Calculating first order moments of state & control history "
                          f"\n\tState: µ={self.standard_scaler[STATES].mean_},σ={self.standard_scaler[STATES].scale_}"
                          f"\n\tCtrl: µ={self.standard_scaler[CTRLS].mean_},σ={self.standard_scaler[CTRLS].scale_}")

                state_mean, state_std = self.standard_scaler[STATES].mean_, self.standard_scaler[STATES].scale_
                ctrl_mean, ctrl_std = self.standard_scaler[CTRLS].mean_, self.standard_scaler[CTRLS].scale_
                # Process symmetry transformations if given to better estimate the empirical mean and variance
                if self.group_actions is not None:
                    # Empirically compute mean by acting on the dataset data
                    G_state_mean, G_state_var = [], []
                    for g_s, g_u in self.group_actions:
                        G_state_mean.append(np.mean(np.dot(g_s, states[..., None]), axis=1).squeeze())
                    true_state_mean = np.mean(G_state_mean, axis=0)

                    # Use symmetry properties of the mean to compute the expected mean given symmetry
                    G_state_mean = [np.dot(g_s, state_mean[..., None]).squeeze() for g_s, _ in self.group_actions]
                    G_ctrl_mean = [np.dot(g_u, ctrl_mean[..., None]).squeeze() for _, g_u in self.group_actions]

                    expected_state_mean = np.mean(G_state_mean, axis=0)
                    expected_ctrl_mean = np.mean(G_ctrl_mean, axis=0, keepdims=True)

                    # Compute empirically the variance of the dataset
                    G_state_var, G_ctrl_var = [], []
                    for g_s, g_u in self.group_actions:
                        G_state_var.append(np.mean(np.power(np.dot(g_s, states[..., None]).squeeze() -
                                                            expected_state_mean[..., None], 2), axis=-1))
                        G_ctrl_var.append(np.mean(np.power(np.dot(g_u, ctrls[..., None]).squeeze() -
                                                           expected_ctrl_mean, 2), axis=-1))

                    expected_state_var = np.mean(G_state_var, axis=0)
                    expected_ctrl_var = np.mean(G_ctrl_var, axis=0, keepdims=True)

                    # TODO: Check that the trasnformation Sum_g\in G (g Var(x) g^-1) is equivalent.
                    # Use symmetry properties to
                    # Update emprirical first and second momentums of the data distribution considering symmetries.
                    self.standard_scaler[STATES].mean_, self.standard_scaler[STATES].scale_ = expected_state_mean, np.sqrt(expected_state_var)
                    self.standard_scaler[CTRLS].mean_, self.standard_scaler[CTRLS].scale_ = expected_ctrl_mean, np.sqrt(expected_ctrl_var)
                    self.standard_scaler[STATES].var_, self.standard_scaler[CTRLS].var_ = expected_state_var, expected_ctrl_var

                # for i, joint in enumerate(self.robot.joints):
                #     joint_type = joint.shortname()
                #     if joint.idx_q == -1 or joint.idx_v == -1:
                #         log.debug(f"Ignoring joint not in state space {joint_type}")
                #         continue
                #     log.debug(f"{joint_type} nq:{joint.nq} nv:{joint.nv} idx_q:{joint.idx_q} idx_v:{joint.idx_v}")
                #     if UNBOUND_REVOLUTE in joint_type:
                #         # The DoF belongs to the S1: Unit Circle Lie Group
                #         dim_start, dim_end = joint.idx_q, joint.idx_q + joint.nq
                #         self.standard_scaler[STATES].mean_[list(range(dim_start, dim_end))] = 0.
                #         self.standard_scaler[STATES].scale_[list(range(dim_start, dim_end))] = 1.
                #         self.standard_scaler[STATES].var_[list(range(dim_start, dim_end))] = 1.
                #         log.info(f"Avoiding scaling state dims {dim_start} to {dim_end} as these are the coordinates"
                #                  f" of position in the unit circle S1 manifold")

            states = self.standard_scaler[STATES].transform(states)
            ctrls = self.standard_scaler[CTRLS].transform(ctrls)
            log.debug(f"State & Control history normalized "
                     f"\n\tState: µ={np.mean(states, axis=0)},σ={np.std(states, axis=0)}"
                     f"\n\tCtrl: µ={np.mean(ctrls, axis=0)},σ={np.std(ctrls, axis=0)}")
            self._state_scale = torch.from_numpy(self.standard_scaler[STATES].scale_).to(device=self.device)
            self._ctrl_scale = torch.from_numpy(self.standard_scaler[CTRLS].scale_).to(device=self.device)
            self._state_mean = torch.from_numpy(self.standard_scaler[STATES].mean_).to(device=self.device)
            self._ctrl_mean = torch.from_numpy(self.standard_scaler[CTRLS].mean_).to(device=self.device)
        else:
            self._state_scale, self._ctrl_scale, self._state_mean, self._ctrl_mean = 1., 1., 0., 0.

        return states, ctrls, costs, sample_ids, traj_lengths

if __name__ == "__main__":
    import src.RobotEquivariantNN.nn.LightningModel
    robot_name = "pendulum"
    dynamic_regime = "unstable_fix_point"

    dataset_path = Path("data") / robot_name / dynamic_regime

    data_path = list(dataset_path.glob("*.pickle"))

    dataset = ClosedLoopDynDataset(path=data_path[0])

    data_loader = DataLoader(dataset=dataset, batch_size=24, shuffle=True, sampler=None,
                             collate_fn=lambda x: dataset.collate_fn(x))

    for x in data_loader:
        print(x)
        break