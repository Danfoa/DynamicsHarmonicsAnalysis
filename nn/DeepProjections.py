import logging
import time
from typing import Optional, Union

import torch
from morpho_symm.nn.MLP import MLP
from torch import Tensor
from torch.utils.data import DataLoader

from nn.latent_markov_dynamics import LatentMarkovDynamics
from nn.LinearDynamics import LinearDynamics
from nn.markov_dynamics import MarkovDynamics
from nn.ObservableNet import ObservableNet
from utils.linear_algebra import full_rank_lstsq
from utils.losses_and_metrics import (
    obs_state_space_metrics,
)
from utils.mysc import (
    batched_to_flat_trajectory,
    states_from_traj,
)

log = logging.getLogger(__name__)


class DPNet(LatentMarkovDynamics):
    _default_obs_fn_params = dict(
        num_layers=4,
        num_hidden_units=128,
        activation=torch.nn.ReLU,
        batch_norm=True,
        bias=False,
        init_mode="fan_in",
    )

    def __init__(
        self,
        state_dim: int,
        obs_state_dim: int,
        dt: Union[float, int] = 1,
        max_ck_window_length: int = 6,
        ck_w: float = 0.1,
        orth_w: float = 0.1,
        enforce_constant_fn: bool = True,
        use_spectral_score: bool = True,
        explicit_transfer_op: bool = True,
        obs_fn_params: Optional[dict] = None,
        linear_decoder: bool = True,
        **markov_dyn_params,
    ):
        assert ck_w == 0 or max_ck_window_length >= 1, "Minimum window_size of Chapman-Kolmogorov reg is 2 steps"
        self.state_dim, self.obs_state_dim = state_dim, obs_state_dim
        self.dt = dt
        self.max_ck_window_length = max_ck_window_length
        self.ck_w = ck_w
        self.orth_w = orth_w
        self.use_spectral_score = use_spectral_score
        self.explicit_transfer_op = explicit_transfer_op
        self.inverse_projector = None  # if linear decoder is true, this is the map between obs to states.
        self.inverse_projector_bias = None
        self.linear_decoder = linear_decoder
        self.enforce_constant_fn = enforce_constant_fn

        _obs_fn_params = self._default_obs_fn_params.copy()
        if obs_fn_params is not None:
            _obs_fn_params.update(obs_fn_params)
        # Define the linear dynamics of the observable state space
        obs_state_dym = self.build_obs_dyn_module()
        # Build the observation function and its inverse
        obs_fn = self.build_obs_fn(**_obs_fn_params)
        inv_obs_fn = self.build_inv_obs_fn(linear_decoder=linear_decoder, **_obs_fn_params)
        # Variable holding the transfer operator used to evolve the observable state in time.

        # Initialize the base class
        super(DPNet, self).__init__(
            obs_fn=obs_fn,
            inv_obs_fn=inv_obs_fn,
            obs_state_dynamics=obs_state_dym,
            state_dim=state_dim,
            obs_state_dim=obs_state_dim,
            dt=dt,
            **markov_dyn_params,
        )

    def forecast(self, state: Tensor, n_steps: int = 1, **kwargs) -> [dict[str, Tensor]]:
        """Forward pass of the dynamics model, producing a prediction of the next `n_steps` states.

        This function uses the empirical transfer operator to compute forcast the observable state.

        Args:
            state: (batch_dim, obs_state_dim) Initial observable state of the system.
            n_steps: Number of steps to predict.
            **kwargs:
        Returns:
            pred_obs_state_traj: (batch_dim, n_steps, obs_state_dim) Predicted observable state.

        """
        assert len(state.shape) == 2 and state.shape[-1] == self.state_dim, (
            f"Invalid state: {state.shape}. Expected (batch, {self.state_dim})"
        )
        time_horizon = n_steps + 1

        preprocessed_state = self.pre_process_state(state=state)

        obs_state, obs_state_aux = self.obs_fn(preprocessed_state)

        pred_obs_state_traj = self.obs_space_dynamics.forcast(state=obs_state, n_steps=n_steps)

        pred_state_traj = self.inv_obs_fn(batched_to_flat_trajectory(pred_obs_state_traj))

        pred_state_traj = self.post_process_state(state_traj=pred_state_traj)

        assert pred_state_traj.shape == (self._batch_size, time_horizon, self.state_dim), (
            f"{pred_state_traj.shape}!=({self._batch_size}, {time_horizon}, {self.state_dim})"
        )
        assert pred_obs_state_traj.shape == (self._batch_size, time_horizon, self.obs_state_dim), (
            f"{pred_obs_state_traj.shape}!=({self._batch_size}, {time_horizon}, {self.obs_state_dim})"
        )

        return dict(pred_state_traj=pred_state_traj, pred_obs_state_traj=pred_obs_state_traj)

    def pre_process_obs_state(
        self, obs_state_traj: Tensor, obs_state_traj_aux: Optional[Tensor] = None
    ) -> dict[str, Tensor]:
        """Apply transformations to the observable state trajectory.

        Args:
            obs_state_traj: (batch * time, obs_state_dim) or (batch, time, obs_state_dim)
            obs_state_traj_aux: (batch * time, obs_state_dim) or (batch, time, obs_state_dim)

        Returns:
            Directory containing
                - obs_state_traj: (batch, time, obs_state_dim) tensor.
                - obs_state_traj_aux: (batch, time, obs_state_dim) tensor.

        """
        obs_state_traj = super().pre_process_obs_state(obs_state_traj)["obs_state_traj"]
        obs_state_traj_aux = super().pre_process_obs_state(obs_state_traj_aux)["obs_state_traj"]
        return dict(obs_state_traj=obs_state_traj, obs_state_traj_aux=obs_state_traj_aux)

    def compute_loss_and_metrics(
        self,
        obs_state_traj: Tensor,
        obs_state_traj_aux: Tensor,
        pred_obs_state_traj: Optional[Tensor] = None,
        state: Optional[Tensor] = None,
        next_state: Optional[Tensor] = None,
        pred_state_traj: Optional[Tensor] = None,
        rec_state_traj: Optional[Tensor] = None,
    ) -> (Tensor, dict[str, Tensor]):
        obs_space_metrics = self.get_obs_space_metrics(obs_state_traj, obs_state_traj_aux)

        loss = self.compute_loss(
            spectral_score=obs_space_metrics["spectral_score"],
            corr_score=obs_space_metrics["corr_score"],
            ck_reg=obs_space_metrics["ck_reg"],
            orth_reg=obs_space_metrics["orth_reg"],
        )

        _, metrics = super().compute_loss_and_metrics(
            state=state,
            next_state=next_state,
            pred_state_traj=pred_state_traj,
            rec_state_traj=rec_state_traj,
            obs_state_traj=obs_state_traj,
            pred_obs_state_traj=pred_obs_state_traj,
        )
        obs_space_metrics.update(metrics)

        return loss, obs_space_metrics

    def get_obs_space_metrics(self, obs_state_traj: Tensor, obs_state_traj_aux: Optional[Tensor] = None) -> dict:
        if obs_state_traj_aux is None and self.explicit_transfer_op:
            raise ValueError("aux_obs_space is True but obs_state_traj_aux is None")
        # Compute Covariance and Cross-Covariance operators for the observation state space.
        # Spectral and Projection scores, and CK loss terms.
        obs_space_metrics = obs_state_space_metrics(
            obs_state_traj=obs_state_traj,
            obs_state_traj_aux=obs_state_traj_aux,
            max_ck_window_length=self.max_ck_window_length,
        )
        return obs_space_metrics

    def compute_loss(self, spectral_score: Tensor, corr_score: Tensor, ck_reg: Tensor, orth_reg: Tensor):
        """Compute DPNet loss term.

        Args:
            spectral_score: (time_horizon - 1) Tensor containing the average spectral score between time steps separated
             apart by a shift of `dt` [steps/time]. That is:
                spectral_score[dt - 1] = avg(||Cov(x_i, x'_i+dt)||_HS^2/(||Cov(x_i, x_i)||_2*||Cov(x'_i+dt,
                x'_i+dt)||_2))
                 | ∀ i in [0, time_horizon - dt], dt in [1, min(time_horizon - i, window_size)]
            corr_score: (time_horizon - 1) Tensor containing the correlation scores between time steps separated
             apart by a shift of `dt` [steps/time]. That is:
                corr_score[dt - 1] = avg(||Cov(x_i, x_i)^-1 Cov(x_i, x'_i+dt) Cov(x'_i+dt, x'_i+dt)^-1||_HS^2)
                 | ∀ i in [0, time_horizon - dt], dt in [1, min(time_horizon - i, window_size)]
            orth_reg: (time_horizon) Tensor containing the orthonormality regularization term for each time step.
                That is orth_reg[t] = || Cov(t,t) - I ||_2
            ck_reg: (time_horizon - 1,) Average CK error per `dt` time steps. That is:
                ck_error[dt - 2] = avg(|| Cov(t, t+dt) - Cov(t, t+1) Cov(t+1, t+2) ... Cov(t+dt-1, t+dt) ||) |
                ∀ t in [0, time_horizon - 2], dt in [2, min(time_horizon - 2, ck_window_length)]

        Returns:
            loss: Scalar tensor containing the DPNet loss.

        """
        transfer_op_inv_score = spectral_score if self.use_spectral_score else corr_score

        transfer_op_inv_score = torch.mean(transfer_op_inv_score)  # / self.obs_state_dim
        ck_regularization = self.ck_w * torch.mean(ck_reg)  # / self.obs_state_dim
        orth_regularization = (self.orth_w * self.obs_state_dim) * torch.mean(orth_reg)  # / self.obs_state_dim

        score = transfer_op_inv_score - ck_regularization - orth_regularization

        loss = -score  # Change sign to minimize the loss and maximize the score.
        assert not torch.isnan(loss), "Loss is NaN."
        return loss

    @torch.no_grad()
    def approximate_transfer_operator(self, train_data_loader: DataLoader):
        """Compute the empirical transfer operator using the standard full-rank DMD algorithm.

        In which for data matrices X, X': (obs_state_dim, n_samples) the transfer operator is computed as:
        A = min_A(||X' - AX||_F) := X' pinv(X) = X'((X^* X)^-1 X^*)
        To approximate the pseudo inverse of X, we use the SVD decomposition of X = U S V^*, which allow us to express
        pinv(X) = V S^-1 U^*. Where V :(obs_state_dim, obs_state_dim) and U: (n_samples, n_samples)
        Then the empirical transfer operator A is computed as:
        A := X' pinv(X)

        Args:
            train_data_loader: Train data loader containing the state and next state data to construct the
            data matrices X and X', using the current observable functions.

        Returns:
            A: The empirical single-step transfer operator of observable state linear dynamics.

        """
        # We construct the data matrices X and X' using the current observable functions.
        train_data = {}
        for batch in train_data_loader:
            for key, value in batch.items():
                if key not in train_data:
                    train_data[key] = torch.squeeze(value)
                else:
                    train_data[key] = torch.cat([train_data[key], torch.squeeze(value)], dim=0)

        # Apply any pre-processing to the state and next state
        state, next_state = train_data["state"], train_data["next_state"]
        state_traj = self.pre_process_state(state, next_state)
        # Obtain the observable state
        obs_fn_output = self.obs_fn(state_traj)
        # Post process observable state
        obs_state_trajs = self.pre_process_obs_state(*obs_fn_output)
        obs_state_traj = obs_state_trajs.pop("obs_state_traj")

        assert obs_state_traj.shape[1] == 2, f"Expected single step datapoints, got {obs_state_traj.shape[1]} steps."
        obs_state, next_obs_state = states_from_traj(obs_state_traj)
        next_obs_state = next_obs_state.squeeze(1)
        # Generate the data matrices of x(w_t) and x(w_t+1)
        X = obs_state.T  # (obs_state_dim, n_samples)
        X_prime = next_obs_state.T  # (obs_state_dim, n_samples)
        solution_op_metrics = self.obs_space_dynamics.update_transfer_op(X=X, X_prime=X_prime)

        metrics = solution_op_metrics
        metrics["rank_obs_state"] = torch.linalg.matrix_rank(X).detach().to(torch.float)

        if self.linear_decoder:
            # Predict the pre-processed state from the observable state
            # pre_state = self.pre_process_state(state)
            inv_projector, inv_projector_bias, inv_projector_metrics = self.empirical_lin_inverse_projector(
                state, obs_state
            )
            metrics.update(inv_projector_metrics)
            self.inverse_projector = inv_projector
            self.inverse_projector_bias = inv_projector_bias

        return metrics

    def build_obs_fn(self, num_layers, identity=False, **kwargs):
        if identity:
            return lambda x: (x, x)
        num_hidden_units = kwargs["num_hidden_units"]
        # Define the feature extractor used by the observable function.
        encoder = MLP(
            in_dim=self.state_dim, out_dim=num_hidden_units, num_layers=num_layers, head_with_activation=True, **kwargs
        )
        return ObservableNet(
            encoder=encoder, obs_dim=self.obs_state_dim, explicit_transfer_op=self.explicit_transfer_op
        )

    def build_inv_obs_fn(self, num_layers, linear_decoder: bool, identity=False, **kwargs):
        if identity:
            return lambda x: x

        if linear_decoder:

            def decoder(dpnet: DPNet, obs_state: Tensor):
                assert hasattr(dpnet, "inverse_projector"), "DPNet.inverse_projector not initialized."
                return torch.nn.functional.linear(
                    obs_state,
                    weight=dpnet.inverse_projector,
                    bias=dpnet.inverse_projector_bias.T if dpnet.inverse_projector_bias is not None else None,
                )

            return lambda x: decoder(self, x)
        else:
            return MLP(in_dim=self.obs_state_dim, out_dim=self.state_dim, num_layers=num_layers, **kwargs)
            raise NotImplementedError("Need to handle the decoupled optimization of encoder/decoder")

    def build_obs_dyn_module(self) -> MarkovDynamics:
        return LinearDynamics(state_dim=self.obs_state_dim, dt=self.dt, trainable=False, bias=self.enforce_constant_fn)

    def empirical_lin_inverse_projector(self, state: Tensor, obs_state: Tensor):
        """Compute the empirical inverse projector from the observable state to the pre-processed state.

        Args:
            state: (batch, state_dim) Tensor containing the pre-processed state.
            obs_state: (batch, obs_state_dim) Tensor containing the observable state.

        Returns:
            A: (state_dim, obs_state_dim) Tensor containing the empirical inverse projector.
            B: (state_dim, 1) Tensor containing the empirical inverse projector bias.
            rec_error: Scalar tensor containing the reconstruction error or "residual".

        """
        # Inverse projector is computed from the observable state to the pre-processed state
        pre_state = self.pre_process_state(state)
        X = obs_state.T  # (obs_state_dim, n_samples)
        Y = pre_state.T  # (state_dim, n_samples)
        A, B = full_rank_lstsq(X, Y, bias=True)

        rec_error = torch.nn.functional.mse_loss(A @ X + B, Y)

        metrics = dict(
            inverse_projector_rank=torch.linalg.matrix_rank(A.detach()).to(torch.float),
            inverse_projector_cond_num=torch.linalg.cond(A.detach()).to(torch.float),
            inverse_projector_error=rec_error,
        )

        return A, B, metrics

    def __repr__(self):
        str = super().__repr__()
        num_params = sum([param.nelement() for param in self.parameters()])
        num_train_params = sum([param.nelement() for param in self.parameters() if param.requires_grad])
        str += (
            f"\n DPnet-ck_w:{self.ck_w}-orth_w:{self.orth_w} "
            f"\tParameters: {num_params} ({num_train_params} trainable)\n"
            f"\tState Space: \n\t\tdim={self.state_dim}\n"
            f"\tObservation Space: \n\t\tdim={self.obs_state_dim}\n"
        )
        return str

    def get_hparams(self):
        return dict(encoder=self.obs_fn.get_hparams())


if __name__ == "__main__":
    n_trajs = 1
    time = 3
    state_dim = 2
    obs_state_dim = state_dim

    change_of_basis = None  # Tensor(random_orthogonal_matrix(state_dim))

    test_dpnet = DPNet(
        state_dim=state_dim,
        obs_state_dim=obs_state_dim,
        state_change_of_basis=change_of_basis,
        obs_fn_params=dict(identity=True),
    )

    random_state_traj = torch.randn(n_trajs, time, state_dim)

    out = test_dpnet(state=random_state_traj[:, 0, :], next_state=random_state_traj[:, 1:, :])
    pred_state_traj = out["pred_state_traj"]

    assert pred_state_traj.shape == random_state_traj.shape, f"{pred_state_traj.shape} != {random_state_traj.shape}"
    assert torch.allclose(pred_state_traj, random_state_traj, rtol=1e-5, atol=1e-5), (
        f"{pred_state_traj - random_state_traj}"
    )
