import logging
from typing import Any, Optional, Union

import torch.nn
from escnn.group import Representation
from escnn.nn import GeometricTensor
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

        # Observation function evaluation ===============================================
        # Compute the projection of the state trajectory in the main and auxiliary observable states
        obs_fn_output = self.obs_fn(state_traj)
        # Post-process observation state trajectories to get (batch, (pred_horizon + 1), obs_state_dim) tensors
        if not isinstance(obs_fn_output, tuple):
            pre_obs_fn_output = self.pre_process_obs_state(obs_fn_output)
        else:
            pre_obs_fn_output = self.pre_process_obs_state(*obs_fn_output)

        # Extract the observable state trajectory
        assert 'obs_state_traj' in pre_obs_fn_output, f"Missing 'obs_state_traj' in {pre_obs_fn_output}"
        obs_state_traj = pre_obs_fn_output.pop('obs_state_traj')
        # Evolution of observable states ===============================================
        # Feed the pre-processed observable state trajectory to the observable state dynamics
        # pred_obs_state_traj = self.obs_state_dynamics.forcast(state=obs_state_traj[:, 0, :], n_steps=pred_horizon)
        obs_dyn_output = self.obs_state_dynamics(state=obs_state_traj[:, 0, :],
                                                 next_state=obs_state_traj[:, 1:, :],
                                                 **pre_obs_fn_output)
        pred_obs_state_traj = obs_dyn_output.pop('pred_state_traj')
        # Observation function inversion ===============================================
        # Compute the prediction of the state trajectory
        # This post-processing of observables ensures the input to the inverse function is of the correct shape.
        post_obs_dyn_output = self.post_process_obs_state(pred_obs_state_traj, **obs_dyn_output)
        pred_state_traj = self.inv_obs_fn(post_obs_dyn_output.pop('pred_obs_state_traj'))
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
        out.update(pre_obs_fn_output)
        out.update(post_obs_dyn_output)
        return out

    def pre_process_obs_state(self, obs_state_traj: Tensor) -> dict[str, Tensor]:
        """ Apply transformations to the observable state trajectory.
        Args:
            obs_state_traj: (batch * time, obs_state_dim) or (batch, time, obs_state_dim)
        Returns:
            Directory containing
                - obs_state_traj: (batch, time, obs_state_dim) tensor.
                - other: Other observations required for training of the model.
        """
        if len(obs_state_traj.shape) == 2:
            obs_state_traj = flat_to_batched_trajectory(
                obs_state_traj, batch_size=self._batch_size, state_dim=self.obs_state_dim)
        elif len(obs_state_traj.shape) == 3:
            pass
        else:
            raise RuntimeError(f"Invalid observable state trajectory shape {obs_state_traj.shape}. Expected "
                               f"(batch, time, {self.obs_state_dim}) or (batch * time, {self.obs_state_dim})")

        return dict(obs_state_traj=obs_state_traj)

    def post_process_obs_state(self, pred_state_traj: Tensor, **kwargs) -> dict[str, Tensor]:
        """ Post-process the predicted observable state trajectory given by the observable state dynamics.

        Args:
            pred_state_traj: (batch, time, obs_state_dim) Trajectory of the predicted (time -1) observable states
             predicted by the transfer operator.
            **kwargs:
        Returns:
            Dictionary contraining
                - pred_obs_state_traj: (batch, time, obs_state_dim) Trajectory
        """
        flat_pred_obs_state_traj = batched_to_flat_trajectory(pred_state_traj)
        return dict(pred_obs_state_traj=flat_pred_obs_state_traj)

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

    def get_hparams(self):
        hparams = {}
        if hasattr(self.obs_fn, 'get_hparams'):
            obs_fn_params = self.obs_fn.get_hparams()
            hparams['obs_fn'] = obs_fn_params
        if hasattr(self.inv_obs_fn, 'get_hparams'):
            inv_obs_fn_params = self.inv_obs_fn.get_hparams()
            hparams['inv_obs_fn'] = inv_obs_fn_params
        if hasattr(self.obs_state_dynamics, 'get_hparams'):
            obs_state_dyn_params = self.obs_state_dynamics.get_hparams()
            hparams['obs_state_dynamics'] = obs_state_dyn_params
        hparams['state_dim'] = self.state_dim
        hparams['obs_state_dim'] = self.obs_state_dim
        hparams['dt'] = self.dt
        return hparams

    def __repr__(self):
        return f"{self.__class__.__name__}(state_dim={self.state_dim} obs_state_dim={self.obs_state_dim}, dt={self.dt})"