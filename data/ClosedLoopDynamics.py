import copy
import pickle
from math import isnan
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


# TODO: dependencies to update.
from src.RobotEquivariantNN.utils.utils import dense

import logging
log = logging.getLogger(__name__)

STATES, CTRLS = "states", "ctrls"
FULL_TRAJ = "full"

class ClosedLoopDynDataset(Dataset):

    def __init__(self, path: Path, window_size=1, compute_Q_func=False, input_frames=1, output_frames=1, normalize=True,
                 state_scaler: StandardScaler = None, ctrl_scaler: StandardScaler = None, device='cpu',
                 augment=False, rep_state=None, rep_ctrl=None, robot: RobotWrapper=None):

        assert path.exists(), path.absolute()
        with open(path, "rb") as file_handle:
            self._data = pickle.load(file_handle)

        self._num_terminal_states = 1  # Num of terminal states without control action.

        self.device = device
        self.window_size = window_size
        self.augment = augment
        self.robot = robot
        self.rep_state, self.rep_ctrl = rep_state, rep_ctrl
        self.normalize = normalize
        self.standard_scaler = {STATES: StandardScaler(copy=False), CTRLS: StandardScaler(copy=False)}
        if state_scaler is not None and ctrl_scaler is not None:
            self.standard_scaler[STATES] = copy.deepcopy(state_scaler)
            self.standard_scaler[CTRLS] = copy.deepcopy(ctrl_scaler)

        states, ctrls, costs, sample_ids = self.process_trajectory_data()

        # TODO: Switch to hugging face API
        self.states = torch.from_numpy(states).type('torch.FloatTensor').to(device)
        self.ctrls = torch.from_numpy(ctrls).type('torch.FloatTensor').to(device)
        self._sample_ids = sample_ids

        # TODO:
        self._compute_Q_func = compute_Q_func
        self._in_frames = input_frames
        self._out_frames = output_frames

        # Deal with Symmetry representations
        self._symmetry_actions = None
        if self.augment and self.rep_state is not None:
            symmetry_actions = []
            for g_x, g_u in zip(self.rep_state.G.discrete_actions, self.rep_ctrl.G.discrete_actions):
                symmetry_actions.append((torch.tensor(np.asarray(dense(g_x))).to(device),
                                         torch.tensor(np.asarray(dense(g_u))).to(device)))
            self._symmetry_actions = symmetry_actions

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
        sample_idx = self._sample_ids[sample_num]
        return {STATES: self.states[sample_idx:sample_idx + self.window_size, :],
                CTRLS: self.ctrls[sample_idx:sample_idx + self.window_size, :],
        }

    def collate_fn(self, batch):
        # Enforce data type in batched array
        # Small hack to do batched augmentation. TODO: Although efficient this should be done somewhere else.
        collated_batch = default_collate(batch)

        if self.augment and self._symmetry_actions is not None:  # Sample uniformly among symmetry actions including identity
            X_batch, U_batch = collated_batch[STATES], collated_batch[CTRLS]
            g_x, g_u = random.choice(self._symmetry_actions)
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
        self._trajs = self._data['trajs']
        self.dt = self._data['traj_params']['dt']
        self.traj_length = self._trajs[0]["states"].shape[0] - self._num_terminal_states
        self.window_size = self.traj_length - 1 if self.window_size == FULL_TRAJ else self.window_size

        num_timesteps = 0
        sample_ids = []
        # state, next state, control, cost.
        states, ctrls, costs = [], [], []
        # Each recorded trajectory is
        for traj in tqdm(self._trajs, desc="Processing trajectories"):
            # Terminal states do not have control action
            assert traj["states"].shape[0] == traj[CTRLS].shape[0] + self._num_terminal_states, f"We assume OC trajectories"
            assert traj["states"].shape[0] == self.traj_length + self._num_terminal_states, f"We assume equal length trajs"
            states.append(traj["states"][:-self._num_terminal_states])
            ctrls.append(traj[CTRLS])
            costs.append(traj['cost'][:-self._num_terminal_states])
            #

            num_samples = self.traj_length - self.window_size
            sample_ids.append(range(num_timesteps, num_timesteps + num_samples))
            num_timesteps += self.traj_length
        states = np.vstack(states)
        ctrls = np.vstack(ctrls)
        costs = np.vstack(costs)
        sample_ids = np.hstack(sample_ids)

        # TODO: Allow for delayed coordinates and longer forcasting outputs
        # TODO: Estimate Q and V
        if self.normalize:
            first_call = not hasattr(self.standard_scaler[STATES], "n_samples_seen_")
            if first_call:
                self.standard_scaler[STATES].fit(states)
                self.standard_scaler[CTRLS].fit(ctrls)
                log.info(f"Calculating first order moments of state & control history "
                         f"\n\tState: µ={self.standard_scaler[STATES].mean_},σ={self.standard_scaler[STATES].var_}"
                         f"\n\tCtrl: µ={self.standard_scaler[CTRLS].mean_},σ={self.standard_scaler[CTRLS].var_}")
            states = self.standard_scaler[STATES].transform(states)
            ctrls = self.standard_scaler[CTRLS].transform(ctrls)
            log.info(f"State & Control history normalized "
                     f"\n\tState: µ={np.mean(states, axis=0)},σ={np.std(states, axis=0)}"
                     f"\n\tCtrl: µ={np.mean(ctrls, axis=0)},σ={np.std(ctrls, axis=0)}")
            self._state_scale = torch.from_numpy(self.standard_scaler[STATES].scale_).to(device=self.device)
            self._ctrl_scale = torch.from_numpy(self.standard_scaler[CTRLS].scale_).to(device=self.device)
            self._state_mean = torch.from_numpy(self.standard_scaler[STATES].mean_).to(device=self.device)
            self._ctrl_mean = torch.from_numpy(self.standard_scaler[CTRLS].mean_).to(device=self.device)
        else:
            self._state_scale, self._ctrl_scale, self._state_mean, self._ctrl_mean = 1., 1., 0., 0.

        return states, ctrls, costs, sample_ids

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