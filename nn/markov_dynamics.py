import logging
from abc import ABC
from typing import Iterable, Optional, Union
import torch
from escnn.group import Representation
from torch import Tensor

from utils.losses_and_metrics import forecasting_loss_and_metrics
from utils.mysc import (batched_to_flat_trajectory, flat_to_batched_trajectory, random_orthogonal_matrix,
                        states_from_traj, traj_from_states)

log = logging.getLogger(__name__)

class MarkovDynamics(torch.nn.Module):

    def __init__(self,
                 state_dim: Optional[int] = None,
                 state_rep: Optional[Representation] = None,
                 dt: Optional[Union[float, int]] = 1,
                 state_change_of_basis: Optional[Tensor] = None,
                 state_inv_change_of_basis: Optional[Tensor] = None):
        super().__init__()
        assert state_dim is not None or state_rep is not None, \
            "Either state_dim or state_rep must be provided"
        self.state_rep: Representation = state_rep
        self.state_dim = state_dim if state_dim is not None else state_rep.size
        self.dt = dt

        # Configure optional change of basis of the state.
        if state_change_of_basis is not None:
            assert state_change_of_basis.shape == (self.state_dim, self.state_dim), \
                f"Expected change of basis ({self.state_dim}, {self.state_dim}), got {state_change_of_basis.shape}"
            assert torch.linalg.matrix_rank(state_change_of_basis) == self.state_dim, \
                f"Defective state_change_of_basis with rank {torch.linalg.matrix_rank(state_change_of_basis)}"
            if state_inv_change_of_basis is None:
                state_inv_change_of_basis = torch.linalg.inv(state_change_of_basis)
            else:
                assert state_inv_change_of_basis.shape == state_change_of_basis.shape

        self.state_change_of_basis = state_change_of_basis
        self.state_inv_change_of_basis = state_inv_change_of_basis

        # Used to reshape the state trajectory from/back (batch * time, state_dim) to (batch, time, state_dim)
        self._batch_size = None
        log.info(f"Instantiating {self.__class__.__name__} with state_dim={self.state_dim} and dt={self.dt}")

    def forward(self, state: Tensor, next_state: Tensor, **kwargs) -> [dict[str, Tensor]]:
        """Forward pass of the dynamics model, producing a prediction of the next `n_steps` states.
        Args:
            state: (batch, state_dim) Initial state of the system.
            next_state: (batch, pred_horizon, state_dim) Next states of the system in a prediction horizon window. or
               (batch, state_dim) if prediction horizon is 1 [step]
            **kwargs: Any auxiliary input required for training the model.
        Returns:
            predictions (dict): A dictionary containing the predicted state and observable state trajectory.
                - 'pred_state_traj': (batch, pred_horizon + 1, state_dim)
                - Any auxiliary output required for training the model.
        """
        # Apply pre-processing to the initial state and state trajectory
        # obtaining a stare trajectory of shape: (batch * (pred_horizon + 1), state_dim) tensor
        state_traj = self.pre_process_state(state=state, next_state=next_state)
        # Evolution of dynamics ===============================================
        pred_state_traj = state_traj
        log.warning(f"MarkovDynamics.forward() is not implemented for {self.__class__.__name__}. "
                    f"This makes this module default to the trivial or identity dynamics.")
        # =====================================================================
        # Apply the required post-processing of the state.
        pred_state_traj = self.post_process_state(pred_state_traj)
        return dict(pred_state_traj=pred_state_traj)

    def forcast(self, state: Tensor, n_steps: int = 1, **kwargs) -> [dict[str, Tensor]]:
        """Forward pass of the dynamics model, producing a prediction of the next `n_steps` states.
        Args:
            state: (batch_dim, state_dim) Initial observable state of the system.
            n_steps: Number of steps to predict.
            **kwargs:
        Returns:
            pred_next_state: (batch_dim, n_steps, state_dim) Predicted observable state.
        """
        raise NotImplementedError()

    def pre_process_state(self, state: Tensor, next_state: Optional[Tensor]=None) -> Tensor:
        """ Apply transformations to the state and next_state tensors before computing observable states.
        Args:
            state: (batch, state_dim) Initial state of the system.
            next_state: (batch, time - 1, state_dim) Next `time-1` states of the system.
        Returns:
            flat_state_trajectory: (batch * time, state_dim) tensor
        """
        self._batch_size = state.shape[0]
        if next_state is None:
            flat_state_trajectory = state
        else:
            state_trajectory = traj_from_states(state=state, next_state=next_state)
            flat_state_trajectory = batched_to_flat_trajectory(state_trajectory)

        # If change of basis of the state is required apply it.
        if self.state_change_of_basis is not None:
            self.state_change_of_basis = self.state_change_of_basis.to(state.device, dtype=state.dtype)
            # Change basis to Isotypic basis.
            state_trajectory_new_basis = torch.einsum(
                'is,...s->...i', self.state_change_of_basis, flat_state_trajectory
                )
            flat_state_trajectory = state_trajectory_new_basis

        return flat_state_trajectory

    def post_process_state(self, state_traj: Tensor) -> Tensor:
        """ Apply required transformations to the predicted state trajectory.

        Args:
            state_traj: (batch * time, state_dim) or (batch, time, state_dim)
        Returns:
            state_traj: (batch, time, state_dim) tensor
        """
        assert state_traj.shape[-1] == self.state_dim, f"State dimension {state_traj.shape[-1]} != {self.state_dim}"

        # If change of basis of the state was applied invert the transformation.
        if self.state_inv_change_of_basis is not None:
            self.state_inv_change_of_basis = self.state_inv_change_of_basis.to(state_traj.device,
                                                                               dtype=state_traj.dtype)
            # Change basis to Isotypic basis.
            state_traj_new_basis = torch.einsum('is,...s->...i', self.state_inv_change_of_basis, state_traj)
            state_traj = state_traj_new_basis

        if len(state_traj.shape) == 2:
            return flat_to_batched_trajectory(state_traj,
                                              batch_size=self._batch_size,
                                              state_dim=self.state_dim)
        elif len(state_traj.shape) == 3:
            return state_traj
        else:
            raise ValueError(f"state_traj must be of shape (batch * time, state_dim) or "
                             f"(batch, time, state_dim) tensor, got {state_traj.shape}")

    def compute_loss_and_metrics(self,
                                 state_traj: Tensor,
                                 pred_state_traj: Tensor,
                                 **kwargs) -> (Tensor, dict[str, Tensor]):
        """ Compute the loss and metrics for the predicted state trajectory. """

        pred_loss, pred_metrics = forecasting_loss_and_metrics(gt=state_traj, pred=pred_state_traj)

        return pred_loss, pred_metrics

    def check_state_traj_shape(self,
                               state_dim: Optional[int] = None,
                               time_horizon: Optional[int] = None,
                               **state_trajectories):
        """ Check shapes of state trajectories in time and batch dimensions. """
        assert len(state_trajectories) > 0, "No state trajectories provided."
        state_dim = self.state_dim if state_dim is None else state_dim
        for traj_name, state_traj in state_trajectories.items():
            assert len(state_traj.shape) == 3 and state_traj.shape[-1] == state_dim, \
                f"Expected {traj_name} (batch, time, {state_dim}), got {state_traj.shape}"
            if time_horizon is None:
                assert state_traj.shape[0] == self._batch_size, \
                    f"Expected {traj_name} ({self._batch_size}, time_horizon, {state_dim}), got {state_traj.shape}"
            else:
                assert state_traj.shape[0] == self._batch_size and state_traj.shape[1] == time_horizon, \
                    f"Expected {traj_name} ({self._batch_size}, {time_horizon}, {state_dim}), got {state_traj.shape}"

    def get_hparams(self):
        hparams = {}
        hparams['state_dim'] = self.state_dim
        hparams['obs_state_dim'] = self.obs_state_dim
        hparams['dt'] = self.dt
        return hparams

    def __repr__(self):
        module_str = super().__repr__()
        return module_str

if __name__ == "__main__":


    state_dim = 100
    n_trajs, time = 2, 5
    change_of_basis = torch.Tensor(random_orthogonal_matrix(state_dim))
    random_state_traj = torch.randn(n_trajs, time, state_dim)

    # Test the change of basis
    state = random_state_traj
    trivial_dyn = MarkovDynamics(state_dim=state_dim, state_change_of_basis=change_of_basis)

    pred_state_traj = trivial_dyn(state=random_state_traj[:, 0, :], next_state=random_state_traj[:, 1:, :])

    assert pred_state_traj.shape == random_state_traj.shape, f"{pred_state_traj.shape} != {random_state_traj.shape}"
    assert torch.allclose(pred_state_traj, random_state_traj, rtol=1e-5, atol=1e-5), f"{pred_state_traj - random_state_traj}"

    state = random_state_traj[:, 0, :]
    pre_state = trivial_dyn.pre_process_state(state=state)
    pred_state = trivial_dyn.post_process_state(pre_state)
    pred_state = pred_state.squeeze(1)
    assert pred_state.shape == state.shape, f"{pred_state.shape} != {state.shape}"
    assert torch.allclose(pred_state, state, rtol=1e-5, atol=1e-5), f"{pred_state - state}"

