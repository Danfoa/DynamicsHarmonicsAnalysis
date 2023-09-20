import logging
from typing import Any, Optional, Union

import torch.nn
from escnn.group import Representation
from torch import Tensor

from nn.markov_dynamics import MarkovDynamics
from utils.mysc import batched_to_flat_trajectory, flat_to_batched_trajectory, traj_from_states

log = logging.getLogger(__name__)


class LatentMarkovDynamics(MarkovDynamics):

    def __init__(self,
                 obs_fn: torch.nn.Module,
                 inv_obs_fn: torch.nn.Module,
                 obs_state_dynamics: MarkovDynamics,
                 state_dim: Optional[int] = None,
                 state_rep: Optional[Representation] = None,
                 obs_state_dim: Optional[int] = None,
                 obs_state_rep: Optional[Representation] = None,
                 dt: Optional[Union[float, int]] = 1,
                 state_change_of_basis: Optional[Tensor] = None,
                 state_inv_change_of_basis: Optional[Tensor] = None):

        assert obs_state_dim is not None or obs_state_rep is not None, \
            "Either obs_state_dim or obs_state_rep must be provided"

        super().__init__(state_dim=state_dim,
                         state_rep=state_rep,
                         dt=dt,
                         state_change_of_basis=state_change_of_basis,
                         state_inv_change_of_basis=state_inv_change_of_basis)
        self.obs_state_dim = obs_state_dim if obs_state_dim is not None else obs_state_rep.size
        self.obs_state_rep = obs_state_rep
        self.obs_fn = obs_fn
        self.inv_obs_fn = inv_obs_fn
        self.obs_state_dynamics = obs_state_dynamics

    def forward(self, state: Tensor, next_state: Optional[Tensor]) -> [dict[str, Tensor]]:
        """Forward pass of the dynamics model, producing a prediction of the next `n_steps` states.
        Args:
            state: (batch, state_dim) Initial state of the system.
            next_state: (batch, pred_horizon, state_dim) Next states of the system in a prediction horizon window.
        Returns:
            predictions (dict): A dictionary containing the predicted state and observable state trajectory.
                - 'obs_state_traj': (batch, pred_horizon + 1, obs_state_dim)
                - 'pred_obs_state_traj': (batch, pred_horizon + 1, obs_state_dim)
                - 'pred_state_traj': (batch, pred_horizon + 1, state_dim)
        """
        assert state.shape[-1] == next_state.shape[-1], f"Invalid state dimension {state.shape[-1]} != {self.state_dim}"
        assert state.shape[0] == next_state.shape[0], f"Invalid batch size {state.shape[0]} != {next_state.shape[0]}"
        assert len(state.shape) == 2, f"Invalid state shape {state.shape}. Expected (batch, {self.state_dim})"
        if len(next_state.shape) == 2:
            next_state = next_state.unsqueeze(1)

        batch, pred_horizon, _ = next_state.shape
        time_horizon = pred_horizon + 1

        # Apply pre-processing to the initial state and state trajectory
        # obtaining a stare trajectory of shape: (batch * (pred_horizon + 1), state_dim) tensor
        state_traj = self.pre_process_state(state=state, next_state=next_state)

        # Evolution of dynamics ===============================================
        # Compute the projection of the state trajectory in the main and auxiliary observable states
        obs_fn_output = self.obs_fn(state_traj)
        # Post-process observation state trajectories to get (batch, (pred_horizon + 1), obs_state_dim) tensors
        obs_state_trajs = self.post_process_obs_state(*obs_fn_output)

        obs_state_traj = obs_state_trajs.pop('obs_state_traj')

        # Compute the prediction of the observable state trajectory
        pred_obs_state_traj = self.obs_state_dynamics.forcast(state=obs_state_traj[:, 0, :], n_steps=pred_horizon)

        # Predictions in original state space   ==============================
        # Compute the prediction of the state trajectory
        pred_state_traj = self.inv_obs_fn(pred_obs_state_traj)
        # Apply the required post-processing of the state.
        pred_state_traj = self.post_process_state(pred_state_traj)

        # Sanity checks of shapes.
        assert pred_state_traj.shape == (batch, time_horizon, self.state_dim), \
            f"{pred_state_traj.shape}!=({batch}, {time_horizon}, {self.state_dim})"
        assert obs_state_traj.shape == (batch, time_horizon, self.obs_state_dim), \
            f"{obs_state_traj.shape}!=({batch}, {time_horizon}, {self.obs_state_dim})"
        assert pred_obs_state_traj.shape == obs_state_traj.shape, f"{pred_obs_state_traj.shape}!={obs_state_traj.shape}"

        out = dict(obs_state_traj=obs_state_traj,
                   pred_obs_state_traj=pred_obs_state_traj,
                   pred_state_traj=pred_state_traj)
        out.update(obs_state_trajs)
        return out

    def post_process_obs_state(self, obs_state_traj: Tensor, *aux_obs_state_trajs) -> dict[str, Tensor]:
        """ Apply transformations to the observable state trajectory.
        Args:
            obs_state_traj: (batch * time, obs_state_dim) or (batch, time, obs_state_dim)
            aux_obs_state_trajs: iterable of (batch * time, obs_state_dim) or (batch, time, obs_state_dim)
        Returns:
            obs_state_traj: (batch, time, obs_state_dim) tensor
        """
        out = {}
        if len(obs_state_traj.shape) == 2:
            obs_state_traj = flat_to_batched_trajectory(
                obs_state_traj, batch_size=self._batch_size, state_dim=self.obs_state_dim)

        out['obs_state_traj'] = obs_state_traj

        if aux_obs_state_trajs is not None:
            for i, traj in enumerate(aux_obs_state_trajs):
                if len(traj.shape) == 2:
                    traj = flat_to_batched_trajectory(traj, batch_size=self._batch_size, state_dim=self.obs_state_dim)
                out[f'obs_state_traj_aux{"" if i == 0 else f"_{i}"}'] = traj
        return out

    def compute_loss_and_metrics(self,
                                 state_traj: Tensor,
                                 pred_state_traj: Tensor,
                                 obs_state_traj: Tensor,
                                 pred_obs_state_traj: Tensor) -> (Tensor, dict[str, Tensor]):
        """ Compute the loss and metrics for the predicted state trajectory. You should probably override this."""
        obs_pred_loss, obs_pred_metrics = self.obs_state_dynamics.compute_loss_and_metrics(
            state_traj=state_traj, pred_state_traj=pred_obs_state_traj)

        state_pred_loss, state_pred_metrics = super().compute_loss_and_metrics(
            state_traj=state_traj, pred_state_traj=pred_state_traj)

        metrics = obs_pred_metrics
        metrics['obs_pred_loss'] = obs_pred_loss

        shared_metrics = set(state_pred_metrics.keys()).symmetric_difference(set(obs_pred_metrics.keys()))
        if len(shared_metrics) > 0:
            raise RuntimeError(f"Metrics: {shared_metrics} are shared by observation and state space predictions.")

        metrics.update(state_pred_metrics)

        return state_pred_loss, metrics
