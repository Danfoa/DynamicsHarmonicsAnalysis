import cProfile
import logging
import pstats
import time
from pathlib import Path
from typing import Optional, Protocol, Union

import numpy as np
import torch
from lightning import seed_everything
from plotly.graph_objs import Figure
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.DynamicsDataModule import DynamicsDataModule
from nn.TwinMLP import TwinMLP
from nn.mlp import MLP
from utils.losses_and_metrics import (
    forecasting_loss_and_metrics,
    obs_state_space_metrics,
)
from utils.mysc import print_dict, random_orthogonal_matrix, states_from_traj, traj_from_states
from utils.plotting import combine_side_by_side, plot_system_3D, plot_trajectories, plot_two_panel_trajectories

log = logging.getLogger(__name__)


class DmdSolver(Protocol):
    def __call__(
        self, X: Tensor, X_prime: Tensor, **kwargs
    ) -> Tensor:
        """Compute the least squares solution of the linear system X' = X·A.
        Args:
            X: (state_dim, n_samples) Data matrix of the initial states.
            X_prime: (state_dim, n_samples) Data matrix of the next states.
        Returns:
            A: (state_dim, state_dim) Least squares solution of the least square problem min_A ||X_prime - A·X||_F
        """
        ...


class DPNet(torch.nn.Module):
    def __init__(
        self,
        state_dim: int,
        obs_state_dim: int,
        num_layers: int = 4,
        num_hidden_units: int = 128,
        activation=torch.nn.ReLU,
        max_ck_window_length: int = 6,
        ck_w: float = 0.1,
        orth_w: float = 0.1,
        use_spectral_score: bool = True,
        single_obs_space: bool = False,
        dmd_algorithm: Optional[DmdSolver] = None,
        dt: float = 1.0,
        batch_norm: bool = True,
        bias: bool = True,
        init_mode: Optional[str] = 'fan_in',
    ):
        super().__init__()
        self.state_dim = state_dim
        self.dt = dt
        self.obs_state_dim = obs_state_dim
        assert (
            max_ck_window_length >= 2
        ), "Minimum window_size of Chapman-Kolmogorov regularization is 2 steps"
        self.max_ck_window_length = max_ck_window_length
        self.ck_w = ck_w
        self.orth_w = orth_w
        self.dmd_algorithm = dmd_algorithm if dmd_algorithm is not None else self._full_rank_lstsq
        self.use_spectral_score = use_spectral_score
        self.single_obs_space = single_obs_space

        self.obs_state_fn = self.build_obs_fn(num_layers=num_layers,
                                              activation=activation,
                                              num_hidden_units=num_hidden_units,
                                              with_bias=bias,
                                              batch_norm=batch_norm,
                                              init_mode=init_mode)

        # Variable holding the transfer operator used to evolve the observable state in time.
        self.transfer_op = None
        self.inverse_projector = None

    def project(self, state_trajectory: Tensor, **kwargs) -> [dict[str, Tensor]]:

        obs_state_traj, obs_state_traj_prime = self.obs_state_fn(state_trajectory)

        return dict(obs_state_traj=obs_state_traj, obs_state_traj_prime=obs_state_traj_prime)

    def pre_process_state(self, state: Tensor, next_state: Tensor) -> dict[str, Tensor]:
        state_trajectory = traj_from_states(state=state, next_state=next_state)
        return dict(state_trajectory=state_trajectory)

    def post_process_pred(self, obs_state_traj: Tensor, obs_state_traj_prime: Tensor) -> dict:
        # Reshape to (batch, time, obs_state_dim)
        obs_state, next_obs_state = states_from_traj(obs_state_traj)  # The Rest is the observations of the next states,
        obs_state_prime, next_obs_state_prime = states_from_traj(obs_state_traj_prime)
        return dict(
            obs_state=obs_state,
            next_obs_state=next_obs_state,
            obs_state_prime=obs_state_prime,
            next_obs_state_prime=next_obs_state_prime,
        )

    def forward(self, state: Tensor, next_state: Optional[Tensor], n_steps: int = 0) -> [dict[str, Tensor]]:
        """Forward pass of the dynamics model, producing a prediction of the next `n_steps` states.
        Args:
            state: Initial state of the system.
            n_steps: Number of steps to predict.
        Returns:
            predictions (dict): A dictionary containing the predicted states under the key 'state' and
            potentially other auxiliary measurements.
        """
        # Apply any required pre-processing to the initial state `w0` and state trajectory `[w1, w2, ..., wH]`
        input = self.pre_process_state(state=state, next_state=next_state)  # Returns a w:(batch * H, state_dim) tensor
        # Compute the trajectory of observable states x(w)=[x(w0),...] and x'(w)=[x'(w0),...] producing
        projections = self.project(**input)
        # Post-process observation state trajectories
        output = self.post_process_pred(**projections)

        if n_steps > 0 and self.transfer_op is not None:
            obs_state = output["obs_state"]
            pred_next_obs_state = self.forecast(obs_state, n_steps=n_steps)
            output.update(pred_next_obs_state)
        return output

    @torch.no_grad()
    def forecast(
        self, obs_state: Tensor, n_steps: int = 1, **kwargs
    ) -> [dict[str, Tensor]]:
        """This function uses the empirical transfer operator to compute forcast the observable state.

        Because in DP nets the forcasting error is not used in the loss term, this function is by construction
        not generating the computational graph needed for gradient propagation.
        Args:
            obs_state: (batch_dim, obs_state_dim) Initial observable state of the system.
            n_steps: Number of steps to predict.
            **kwargs:
        Returns:
            pred_next_obs_state: (batch_dim, n_steps, obs_state_dim) Predicted observable state.
        """

        if self.transfer_op is None:
            raise RuntimeError(
                "The transfer operator not approximated yet. Call `approximate_transfer_operator`"
            )

        # Use the empirical transfer operator to compute the maximum likelihood prediction of the trajectory
        pred_next_obs_state = [obs_state]
        pred_state = (self.inverse_projector @ obs_state.T).T
        pred_next_state = []
        # The transfer operator of this Isotypic subspace
        for step in range(n_steps):
            # Compute the next state prediction s_t+1 = K @ s_t
            next_obs_state = (self.transfer_op @ pred_next_obs_state[-1].T).T
            pred_next_obs_state.append(next_obs_state)
            pred_next_state.append((self.inverse_projector @ next_obs_state.T).T)

        pred_next_obs_state = torch.stack(pred_next_obs_state[1:], dim=1)
        pred_next_state = torch.stack(pred_next_state, dim=1)
        return dict(
            pred_state=pred_state,
            pred_next_state=pred_next_state,
            pred_next_obs_state=pred_next_obs_state,
        )

    def compute_loss_and_metrics(self,
                                 obs_state: Tensor,
                                 next_obs_state: Tensor,
                                 obs_state_prime: Optional[Tensor] = None,
                                 next_obs_state_prime: Optional[Tensor] = None,
                                 pred_next_obs_state: Optional[Tensor] = None,
                                 state: Optional[Tensor] = None,
                                 pred_state: Optional[Tensor] = None,
                                 next_state: Optional[Tensor] = None,
                                 pred_next_state: Optional[Tensor] = None
                                 ) -> (Tensor, dict[str, Tensor]):

        obs_state_traj = traj_from_states(obs_state, next_obs_state)
        obs_state_traj_prime = traj_from_states(obs_state_prime, next_obs_state_prime)

        if self.single_obs_space:
            obs_state_traj_prime = None
        # Compute Covariance and Cross-Covariance operators for the observation state space.
        # Spectral and Projection scores, and CK loss terms.
        obs_space_metrics = obs_state_space_metrics(obs_state_traj=obs_state_traj,
                                                    obs_state_traj_prime=obs_state_traj_prime,
                                                    max_ck_window_length=self.max_ck_window_length)

        loss = self.compute_loss(spectral_score=obs_space_metrics["spectral_score"],
                                 corr_score=obs_space_metrics["corr_score"],
                                 ck_reg=obs_space_metrics["ck_reg"],
                                 orth_reg=obs_space_metrics["orth_reg"])

        # Include prediction metrics if available
        if pred_next_obs_state is not None:
            assert pred_next_obs_state.shape == next_obs_state.shape
            obs_pred_loss, obs_pred_metrics = forecasting_loss_and_metrics(
                state_gt=next_obs_state, state_pred=pred_next_obs_state)
            obs_pred_metrics["obs_pred_loss"] = obs_pred_loss
            obs_pred_metrics["obs_pred_loss_t"] = obs_pred_metrics.pop("pred_loss_t")
            # Get the state prediction error
            assert pred_next_state.shape == next_state.shape
            pred_loss, pred_metrics = forecasting_loss_and_metrics(state_gt=next_state, state_pred=pred_next_state)
            pred_metrics["pred_loss"] = pred_loss
            pred_metrics["decoder_error"] = torch.nn.functional.mse_loss(state, pred_state)

            cond_num_transfer_op = torch.linalg.cond(self.transfer_op)
            obs_pred_metrics["cond_num_transfer_op"] = cond_num_transfer_op
            obs_space_metrics.update(obs_pred_metrics)
            obs_space_metrics.update(pred_metrics)
            if log.level == logging.DEBUG and cond_num_transfer_op > 100:
                log.warning(f"Condition number of transfer operator: {cond_num_transfer_op:.2f}.")

        return loss, obs_space_metrics

    def compute_loss(self, spectral_score: Tensor, corr_score: Tensor, ck_reg: Tensor, orth_reg: Tensor):
        """ Compute DPNet loss term.

        Args:
            spectral_score: (time_horizon - 1) Tensor containing the average spectral score between time steps separated
             apart by a shift of `dt` [steps/time]. That is:
                spectral_score[dt - 1] = avg(||Cov(x_i, x'_i+dt)||_HS^2/(||Cov(x_i, x_i)||_2*||Cov(x'_i+dt, x'_i+dt)||_2))
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

        transfer_inv_score = spectral_score if self.use_spectral_score else corr_score

        score = torch.mean(transfer_inv_score) - (self.ck_w * torch.mean(ck_reg)) - (self.orth_w * torch.mean(orth_reg))

        # Apply the orthogonal regularization term
        score = score - self.orth_w * torch.mean(orth_reg)
        loss = -score  # Change sign to minimize the loss and maximize the score.
        assert not torch.isnan(loss), f"Loss is NaN."
        return loss

    @torch.no_grad()
    def eval_metrics(self,
                     obs_state: Tensor,
                     next_obs_state: Tensor,
                     pred_next_obs_state: Optional[Tensor] = None,
                     state: Optional[Tensor] = None,
                     pred_state: Optional[Tensor] = None,
                     next_state: Optional[Tensor] = None,
                     pred_next_state: Optional[Tensor] = None,
                     **kwargs
                     ) -> (dict[str, Figure], dict[str, Tensor]):

        obs_state_traj = traj_from_states(obs_state, next_obs_state).detach().cpu().numpy()
        pred_obs_state_traj = traj_from_states(obs_state, pred_next_obs_state).detach().cpu().numpy()

        state_traj = traj_from_states(state, next_state).detach().cpu().numpy()
        pred_state_traj = traj_from_states(pred_state, pred_next_state).detach().cpu().numpy()

        fig = plot_two_panel_trajectories(state_trajs=state_traj,
                                          pred_state_trajs=pred_state_traj,
                                          obs_state_trajs=obs_state_traj,
                                          pred_obs_state_trajs=pred_obs_state_traj,
                                          dt=self.dt,
                                          n_trajs_to_show=5)

        fig_3ds = plot_system_3D(trajectories=state_traj[:20], secondary_trajectories=pred_state_traj[:20], title='state_traj')
        fig_3do = plot_system_3D(trajectories=obs_state_traj[:20], secondary_trajectories=pred_obs_state_traj[:20],
                                 title='obs_state')
        figs = dict(prediction=fig, state=fig_3ds, obs_state=fig_3do)
        metrics = None
        return figs, metrics

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
                    torch.cat([train_data[key], torch.squeeze(value)], dim=0)
        for key, value in train_data.items():
            train_data[key] = value[:6]

        pred = self(**train_data)

        state = train_data["state"]
        obs_state = pred["obs_state"]
        next_obs_state = torch.squeeze(pred["next_obs_state"])

        # Generate the data matrices of x(w_t) and x(w_t+1)
        X = obs_state.T                    # (obs_state_dim, n_samples)
        X_prime = next_obs_state.T         # (obs_state_dim, n_samples)
        empirical_transfer_op = self.dmd_algorithm(X, X_prime)
        # a = empirical_transfer_op.detach().cpu().numpy()
        self.transfer_op = empirical_transfer_op

        # Approximate a linear decoder from "main" observable space to state space.
        X = obs_state.T                      # (obs_state_dim, n_samples)
        Y = state.T                        # (state_dim, n_samples)
        # rank_X = torch.linalg.matrix_rank(X)
        # rank_Y = torch.linalg.matrix_rank(Y)
        self.inverse_projector = self._full_rank_lstsq(X, Y)
        Y_pred = self.inverse_projector @ X
        # error = torch.mean(torch.abs(Y - Y_pred), dim=-1)
        # rank_inv_proj = torch.linalg.matrix_rank(Y)
        # print("Rank of inv Proj: ", rank_inv_proj)


    @staticmethod
    def _full_rank_lstsq(X: Tensor, Y: Tensor, driver='gelsd', **kwargs) -> Tensor:
        """Compute the least squares solution of the linear system `X' = A·X`. Assuming full rank X and A.
        Args:<
            X: (|x|, n_samples) Data matrix of the initial states.
            Y: (|y|, n_samples) Data matrix of the next states.
        Returns:
            A: (|y|, |x|) Least squares solution of the linear system `X' = A·X`.
        """
        assert (
            X.ndim == 2 and Y.ndim == 2 and X.shape[1] == Y.shape[1]
        ), f"X: {X.shape}, Y: {Y.shape}. Expected (|x|, n_samples) and (|y|, n_samples) respectively."

        # Torch convention uses Y:(n_samples, |y|) and X:(n_samples, |x|) to solve the least squares
        # problem for `Y = X·A`, instead of our convention `Y = A·X`. So we have to do the appropriate transpose.
        result = torch.linalg.lstsq(X.T.detach().cpu().to(dtype=torch.double),
                                    Y.T.detach().cpu().to(dtype=torch.double), rcond=None, driver=driver)
        A = result.solution.T.to(device=X.device, dtype=X.dtype)
        # y_hat = A @ X
        return A

    def build_obs_fn(self, num_layers, **kwargs):
        backbone_params = None
        if num_layers > 3:
            num_backbone_layers = max(2, num_layers-2)
            backbone_feat_dim = kwargs.get('num_hidden_units')
            backbone_params = dict(in_dim=self.state_dim, out_dim=backbone_feat_dim,
                                   num_layers=num_backbone_layers, head_with_activation=True, **kwargs)
            obs_fn_params = dict(in_dim=backbone_feat_dim, out_dim=self.obs_state_dim,
                                 num_layers=num_layers-num_backbone_layers, head_with_activation=False, **kwargs)
        else:
            obs_fn_params = dict(in_dim=self.state_dim, out_dim=self.obs_state_dim, num_layers=num_layers,
                                 head_with_activation=False, **kwargs)

        return TwinMLP(net_kwargs=obs_fn_params, backbone_kwargs=backbone_params, equivariant=False)

    def __repr__(self):
        str = super().__repr__()
        num_params = sum([param.nelement() for param in self.parameters()])
        num_train_params = sum(
            [param.nelement() for param in self.parameters() if param.requires_grad]
        )
        str += (
            f"\n DPnet-ck_w:{self.ck_w}-orth_w:{self.orth_w} "
            f"\tParameters: {num_params} ({num_train_params} trainable)\n"
            f"\tState Space: \n\t\tdim={self.state_dim}\n"
            f"\tObservation Space: \n\t\tdim={self.obs_state_dim}\n"
        )
        return str

    def get_hparams(self):
        return dict(encoder=self.obs_state_fn.get_hparams())


if __name__ == "__main__":
    torch.set_printoptions(precision=3)
    seed_everything(10)
    path_to_data = Path("data")
    assert path_to_data.exists(), f"Invalid Dataset path {path_to_data.absolute()}"

    log.setLevel(logging.DEBUG)
    log.level = logging.DEBUG
    # Find all dynamic systems recordings
    path_to_data /= "linear_system"
    path_to_dyn_sys_data = set(
        [a.parent for a in list(path_to_data.rglob("*train.pkl"))]
    )
    # Select a dynamical system
    mock_path = path_to_dyn_sys_data.pop()

    pred_horizon = 50
    batch_size = 100
    device = torch.device("cuda:0")
    data_module = DynamicsDataModule(
        data_path=mock_path,
        pred_horizon=pred_horizon,
        eval_pred_horizon=100,
        frames_per_step=1,
        num_workers=0,
        batch_size=batch_size,
        augment=True,
        device=device,
    )
    data_module.prepare_data()

    dt = data_module.dt
    num_encoder_layers = 2

    state_type = data_module.state_field_type
    obs_state_dimension = state_type.size * 1
    num_encoder_hidden_neurons = obs_state_dimension * 2
    max_ck_window_length = pred_horizon

    dp_net = DPNet(
        state_dim=data_module.state_field_type.size,
        obs_state_dim=obs_state_dimension,
        num_layers=num_encoder_layers,
        num_hidden_units=num_encoder_hidden_neurons,
        max_ck_window_length=max_ck_window_length,
        activation=torch.nn.Identity,
        bias=True,
        batch_norm=False,
        # init_mode='normal.1',
    )
    print(dp_net)
    dp_net.to(device)

    dp_net.approximate_transfer_operator(data_module.predict_dataloader())

    start_time = time.time()
    profiler = cProfile.Profile()
    profiler.enable()
    for i, batch in tqdm(enumerate(data_module.train_dataloader())):
        for k, v in batch.items():
            batch[k] = v.to(device)
        state, next_state = batch["state"], batch["next_state"]
        n_steps = batch["next_state"].shape[1]

        out = dp_net(state=state, next_state=next_state, n_steps=n_steps)

        # Test loss and metrics
        loss, metrics = dp_net.compute_loss_and_metrics(**batch, **out)
        if i > 1:
            break
    profiler.disable()

    figs, val_metrics = dp_net.eval_metrics(**batch, **out)

    figs["prediction"].show()
    figs["state"].show()
    figs["obs_state"].show()

    print(metrics.get("pred_loss", None))

    # print(f"Computing forward pass and loss/metrics for {id} batches took {time.time() - start_time:.2f}[s]"
    #       f"({(time.time() - start_time) / i:.2f} seconds per batch for {pred_horizon} steps in pred horizon)")

    # Create a pstats object
    stats = pstats.Stats(profiler)

    # Sort stats by the cumulative time spent in the function
    stats.sort_stats("cumulative")

    # Print only the info for the functions defined in your script
    # Assuming your script's name is 'your_script.py'
    stats.print_stats("koopman_robotics")
