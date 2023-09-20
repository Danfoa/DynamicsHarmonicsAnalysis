import logging
from collections import OrderedDict
from typing import Optional, Protocol, Union

import escnn
import numpy as np
import torch
from escnn.group import Representation
from escnn.nn import FieldType, GeometricTensor
from torch import Tensor
from torch.nn import Module

from nn.markov_dynamics import MarkovDynamics
from utils.mysc import full_rank_lstsq
from utils.representation_theory import identify_isotypic_spaces

log = logging.getLogger(__name__)

class DmdSolver(Protocol):
    def __call__(
            self, X: Tensor, X_prime: Tensor, **kwargs
            ) -> Tensor:
        """Compute the least squares solution of the linear system X' = X·A.
        Args:
            X: (|x|, n_samples) Data matrix of the initial states.
            Y: (|y|, n_samples) Data matrix of the next states.
        Returns:
            A: (|y|, |x|) Least squares solution of the linear system `X' = A·X`.
        """
        ...


class LinearDynamics(MarkovDynamics):

    def __init__(self,
                 state_dim: Optional[int] = None,
                 state_rep: Optional[Representation] = None,
                 dmd_algorithm: Optional[DmdSolver] = None,
                 dt: Optional[Union[float, int]] = 1,
                 trainable=False,
                 **markov_dyn_kwargs):

        super().__init__(state_dim=state_dim, state_rep=state_rep, dt=dt, **markov_dyn_kwargs)
        self.is_trainable = trainable

        # Variables for non-training mode
        if not trainable:
            self.transfer_op = None
            self.dmd_algorithm = dmd_algorithm if dmd_algorithm is not None else full_rank_lstsq
        else:
            # Variables for training mode
            self.linear_layer = None    # TODO
            raise NotImplementedError()

    def forcast(self, state: Tensor, n_steps: int = 1, **kwargs) -> Tensor:
        """ Predict the next `n_steps` states of the system.
        Args:
            state: (batch, state_dim) Initial state of the system.
            n_steps: (int) Number of steps to predict.
        Returns:
            pred_state_traj: (batch, n_steps + 1, state_dim)
        """
        batch, state_dim = state.shape
        assert state_dim == self.state_dim

        transfer_op = self.get_transfer_op()

        # Use the transfer operator to compute the maximum likelihood prediction of the trajectory
        pred_state_traj = [state]
        for step in range(n_steps):
            # Compute the next state prediction s_t+1 = K @ s_t
            current_state = pred_state_traj[-1]
            next_obs_state = torch.nn.functional.linear(current_state, transfer_op)
            pred_state_traj.append(next_obs_state)

        pred_state_traj = torch.stack(pred_state_traj, dim=1)
        # a = pred_state_traj.detach().cpu().numpy()
        assert pred_state_traj.shape == (batch, n_steps + 1, state_dim)
        return pred_state_traj

    def get_transfer_op(self):
        if self.is_trainable:
            if isinstance(self.linear_layer, torch.nn.Linear):
                transfer_op = self.linear_layer.weight
            elif isinstance(self.linear_layer, escnn.nn.Linear):
                transfer_op = self.linear_layer.matrix
            else:
                raise NotImplementedError(f"Unknown linear layer type {type(self.linear_layer)}")
        else:
            transfer_op = self.transfer_op
            if transfer_op is None:
                raise RuntimeError("The transfer operator not approximated yet. Call `approximate_transfer_operator`")
        return transfer_op

    def update_transfer_op(self, X: Tensor, X_prime: Tensor) -> dict[str, Tensor]:
        """ Use a DMD algorithm to update the empirical transfer operator
        Args:
            X: (state_dim, n_samples) Data matrix of states at time `t`.
            X_prime: (state_dim, n_samples) Data matrix of the states at time `t + dt`.
        Returns:
            metrics (dict): Dictionary of metrics computed during the update.
        """
        if self.is_trainable:
            raise RuntimeError("This model was initialized as trainable")

        assert X.shape == X_prime.shape, f"X: {X.shape}, X_prime: {X_prime.shape}"
        assert X.shape[0] == self.state_dim, f"Invalid state dimension {X.shape[0]} != {self.state_dim}"

        transfer_op = self.dmd_algorithm(X=X, Y=X_prime)
        assert transfer_op.shape == (self.state_dim, self.state_dim)

        rec_error = torch.nn.functional.mse_loss(transfer_op @ X, X_prime)
        self.transfer_op = transfer_op

        return dict(solution_op_rank=torch.linalg.matrix_rank(transfer_op.detach()).to(torch.float),
                    solution_op_cond_num=torch.linalg.cond(transfer_op.detach()).to(torch.float),
                    solution_op_error=rec_error.detach().to(torch.float))

    def get_hparams(self):
        main_params = dict(state_dim=self.state_dim, trainable=self.is_trainable)
        return main_params

