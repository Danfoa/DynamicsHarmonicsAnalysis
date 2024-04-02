import logging
from collections import OrderedDict
from typing import Optional, Protocol, Union

import escnn
import numpy as np
import torch
from escnn.group import Representation
from torch import Tensor

from nn.markov_dynamics import MarkovDynamics
from utils.linear_algebra import full_rank_lstsq

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
                 bias: bool = True,
                 dmd_algorithm: Optional[DmdSolver] = None,
                 dt: Optional[Union[float, int]] = 1,
                 trainable=False,
                 init_mode: str = "stable",
                 **markov_dyn_kwargs):

        super().__init__(state_dim=state_dim, state_rep=state_rep, dt=dt, **markov_dyn_kwargs)
        self.is_trainable = trainable
        self.bias = bias
        # Variables for non-training mode
        if not trainable:
            self.transfer_op = None
            self.transfer_op_bias = None
            self.dmd_algorithm = dmd_algorithm if dmd_algorithm is not None else full_rank_lstsq
        else:
            self.transfer_op = self.build_linear_map()
            # Initialize weights of the linear layer such that it represents a stable system
            self.reset_parameters(init_mode=init_mode)

    def forward(self, state: Tensor, next_state: Optional[Tensor]=None, **kwargs) -> [dict[str, Tensor]]:
        pred_horizon = next_state.shape[1] if next_state is not None else 1
        pre_processed_state = self.pre_process_state(state=state)

        if self.is_trainable:
            state_traj = self.pre_process_state(state=state, next_state=next_state)
            one_step_evolved_traj = self.transfer_op(state_traj)
            pred_state_traj = self.forcast(state=pre_processed_state, n_steps=pred_horizon)
            out = dict(pred_state_traj=self.post_process_state(pred_state_traj),
                       pred_state_one_step=self.post_process_state(one_step_evolved_traj))
        else:
            pred_state_traj = self.forcast(state=pre_processed_state, n_steps=pred_horizon)
            out = dict(pred_state_traj=self.post_process_state(pred_state_traj))
        return out

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

        # Use the transfer operator to compute the maximum likelihood prediction of the future trajectory
        pred_state_traj = [state]
        for step in range(n_steps):
            # Compute the next state prediction s_t+1 = K @ s_t
            current_state = pred_state_traj[-1]
            if self.is_trainable:
                next_obs_state = self.transfer_op(current_state)
            else:
                transfer_op, bias = self.get_transfer_op()
                if bias is not None:
                    next_obs_state = (transfer_op @ current_state.T + bias).T
                else:
                    next_obs_state = (transfer_op @ current_state.T).T
            pred_state_traj.append(next_obs_state)

        pred_state_traj = torch.stack(pred_state_traj, dim=1)
        assert pred_state_traj.shape == (batch, n_steps + 1, state_dim)
        return pred_state_traj

    def forcast_modes(self, state: Tensor, n_steps: int = 1, rank=-1) -> Tensor:
        batch, state_dim = state.shape

        eig, V, V_inv = self.eigval_decomp()
        if rank < 0:
            eig_tol = torch.finfo(torch.float64).eps * 5
            rank = min(self.state_dim, torch.sum(torch.abs(eig) > eig_tol).item())
        else:
            rank = min(self.state_dim, rank)
        assert rank == -1 or rank >= 1, f"Invalid rank {rank} for state_dim {self.state_dim}"
        # Only consider the first 'rank' eigenvalues and eigenvectors
        eig = eig[:rank]
        V = V[:, :rank]
        V_inv = V_inv[:rank, :]

        # Convert the state tensor to the same dtype as V
        state_cmplx = state.to(dtype=V.dtype)

        # Project the state onto the eigenvectors of the transfer operator
        # Shape of state_modes: [batch, rank]
        state_modes_eigbasis = torch.einsum("ij,...j->...i", V_inv, state_cmplx)

        # Generate a tensor of steps from 1 to n_steps. Shape of steps: [n_steps, 1]
        steps = torch.arange(0, n_steps + 1).to(state.device).unsqueeze(-1)

        # Compute the power of the eigenvalues for each step. Shape of eig_power: [n_steps, rank]
        eig_power = torch.pow(eig.unsqueeze(0), steps)

        # Compute the trajectory of the modes by multiplying the modes by the powered eigenvalues
        # Shape of mode_traj: [..., n_steps, rank]
        mode_traj_eigbasis = torch.einsum('...r,sr->...sr', state_modes_eigbasis, eig_power)

        # Compute the trajectory of the modes in the original coordinates by multiplying the modes by the matrix V
        # Shape of mode_traj_orig_coords: [..., n_steps, rank, state_dim]
        mode_traj_orig_coords = torch.einsum('sr,...tr->...trs', V, mode_traj_eigbasis)

        # Ensure that the recovered modes are approximately real and cast to original dtype
        # mode_traj_orig_coords = mode_traj_orig_coords.to(dtype=state.dtype)

        # Test the above operations by applying the expected evolution of the first mode to the initial state
        T = (V @ torch.diag(eig) @ V_inv).to(dtype=state.dtype)
        step = 2
        T_s = (V @ torch.diag(torch.pow(eig, step)) @ V_inv).to(dtype=state.dtype)
        batch_idx = 2
        x_0 = state[batch_idx, :]
        x_s = T_s @ x_0
        # The sum of the modes should be equal to the state
        x_s_modes = mode_traj_orig_coords[batch_idx, step, :]
        x_s_rec = torch.sum(x_s_modes, dim=0)
        assert torch.allclose(x_s_rec.imag, torch.zeros_like(x_s), atol=1e-3, rtol=1e-3), \
            f"Reconstruction has imaginary component: {torch.max(x_s_rec.imag)}"
        assert torch.allclose(x_s, x_s_rec.real, atol=1e-2, rtol=1e-2), f"Invalid mode evolution: {torch.max(x_s - x_s_rec.real)}"

        return eig, mode_traj_orig_coords

    @torch.no_grad()
    def eigval_decomp(self):
        if self.is_trainable:
            T = self.transfer_op.weight
            # b = self.transfer_op.bias
        else:
            T = self.transfer_op
            # b = self.transfer_op_bias

        # TODO: Remove.... Set random orthogonal matrix
        T = torch.randn_like(T)
        eig, V = torch.linalg.eig(T)  # T = V @ diag(eig) @ V_inv

        # Sort the eigenvalues in descending order of magnitude
        _, idx = torch.sort(torch.abs(eig), descending=True, dim=0, stable=True)
        eig = eig[idx]
        V = V[:, idx]
        V_inv = torch.linalg.inv(V)
        return eig, V, V_inv

    def get_transfer_op(self):
        if self.is_trainable:
            raise RuntimeError("This model was initialized as trainable")
        else:
            transfer_op = self.transfer_op
            bias = self.transfer_op_bias
            if transfer_op is None:
                raise RuntimeError("The transfer operator not approximated yet. Call `approximate_transfer_operator`")
        return transfer_op, bias

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

        A, B = self.dmd_algorithm(X=X, Y=X_prime, bias=self.bias)
        if self.bias:
            rec_error = torch.nn.functional.mse_loss(A @ X + B, X_prime)
        else:
            rec_error = torch.nn.functional.mse_loss(A @ X, X_prime)

        self.transfer_op = A
        self.transfer_op_bias = B
        return dict(solution_op_rank=torch.linalg.matrix_rank(A.detach()).to(torch.float),
                    solution_op_cond_num=torch.linalg.cond(A.detach()).to(torch.float),
                    solution_op_error=rec_error.detach().to(torch.float))

    def build_linear_map(self) -> torch.nn.Linear:
        return torch.nn.Linear(self.state_dim, self.state_dim, bias=self.bias)

    def get_hparams(self):
        main_params = dict(state_dim=self.state_dim, trainable=self.is_trainable)
        return main_params

    def reset_parameters(self, init_mode: str):
        if init_mode == "stable":
            self.transfer_op.weight.data = torch.eye(self.state_dim)
            if self.bias:
                self.transfer_op.bias.data = torch.zeros(self.state_dim)
        else:
            raise NotImplementedError(f"Eival init mode {init_mode} not implemented")
        log.info(f"Eigenvalues initialization to {init_mode}")