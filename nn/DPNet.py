import cProfile
import logging
import pstats
import time
from pathlib import Path
from typing import Callable, Optional, Protocol, Union

import escnn
import numpy as np
import torch
from kooplearn.models import DMD
from lightning import seed_everything
from plotly.graph_objs import Figure
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.DynamicsDataModule import DynamicsDataModule
from nn.mlp import MLP
from utils.losses_and_metrics import (forecasting_loss_and_metrics, obs_state_space_metrics)
from utils.mysc import print_dict
from utils.plotting import plot_trajectories, plot_two_panel_trajectories

log = logging.getLogger(__name__)


class DmdSolver(Protocol):
    def __call__(self, X: torch.Tensor, X_prime: torch.Tensor, **kwargs) -> torch.Tensor:
        """ Compute the least squares solution of the linear system X' = X·A.
        Args:
            X: (state_dim, n_samples) Data matrix of the initial states.
            X_prime: (state_dim, n_samples) Data matrix of the next states.
        Returns:
            A: (state_dim, state_dim) Least squares solution of the least square problem min_A ||X_prime - A·X||_F
        """
        ...


class DPNet(torch.nn.Module):

    def __init__(self,
                 state_dim: int,
                 obs_state_dim: int,
                 num_encoder_layers: int = 4,
                 num_encoder_hidden_neurons: int = 128,
                 max_ck_window_length: int = 6,
                 ck_w: float = 0.1,
                 orthonormal_w: float = 0.1,
                 dmd_algorithm: Optional[DmdSolver] = None,
                 activation=torch.nn.ReLU,
                 dt: float = 1.0,
                 batch_norm: bool = True,
                 bias: bool = True):
        super().__init__()
        self.state_dim = state_dim
        self.dt = dt
        self.obs_state_dim = obs_state_dim
        assert max_ck_window_length >= 2, "Minimum window_size of Chapman-Kolmogorov regularization is 2 steps"
        self.max_ck_window_length = max_ck_window_length
        self.ck_w = ck_w
        self.orthonormal_w = orthonormal_w
        self.dmd_algorithm = dmd_algorithm if dmd_algorithm is not None else self._full_rank_lstsq

        # Define the observable network producing the observation state
        self.projection_backbone = MLP(in_dim=self.state_dim,
                                       out_dim=num_encoder_hidden_neurons,
                                       num_hidden_units=num_encoder_hidden_neurons,
                                       num_layers=num_encoder_layers - 2,
                                       head_with_activation=True,
                                       batch_norm=batch_norm,
                                       activation=activation,
                                       with_bias=True)

        # The two observable state functions (two non-linear changes of coordinates) are parameterized by the
        # last two layers of a parallel MLP.

        self.obs_state_fn = MLP(in_dim=num_encoder_hidden_neurons,
                                out_dim=self.obs_state_dim,
                                num_hidden_units=num_encoder_hidden_neurons,
                                num_layers=2,
                                batch_norm=batch_norm,
                                activation=activation,
                                with_bias=False)

        self.obs_state_fn_prime = MLP(in_dim=num_encoder_hidden_neurons,
                                      out_dim=self.obs_state_dim,
                                      num_hidden_units=num_encoder_hidden_neurons,
                                      num_layers=2,
                                      batch_norm=batch_norm,
                                      activation=activation,
                                      with_bias=False)

        # Private variables
        self._batch_dim = None  # Used to convert back and forward between Tensor and GeometricTensor.

        self.transfer_op = None

        num_params = sum([param.nelement() for param in self.parameters()])
        num_train_params = sum([param.nelement() for param in self.parameters() if param.requires_grad])
        log.info(f"DPnet Num. Parameters: {num_params} ({num_train_params} trainable)\n"
                 f"\tObservation Space: \n\t\tdim={self.obs_state_dim}\n"
                 f"\tState Space: \n\t\tdim={self.state_dim}")

    # def configure_projection(self):

    def project(self, state_trajectory: torch.Tensor, **kwargs) -> [dict[str, torch.Tensor]]:

        obs_backbone = self.projection_backbone(state_trajectory)
        obs_state_traj = self.obs_state_fn(obs_backbone)
        obs_state_traj_prime = self.obs_state_fn_prime(obs_backbone)

        return dict(obs_state_traj=obs_state_traj,
                    obs_state_traj_prime=obs_state_traj_prime)

    def pre_process_state(self,
                          state: torch.Tensor,
                          next_state=torch.Tensor, **kwargs) -> Union[torch.Tensor, dict[str, torch.Tensor]]:

        self._batch_dim = state.shape[0]

        if next_state.shape == state.shape:  # next_state : (batch_dim, state_dim)
            state_trajectory = torch.cat([state, next_state], dim=0)
        else:  # next_state : (batch_dim, pred_horizon, state_dim)
            state_trajectory = torch.cat([torch.unsqueeze(state, dim=1), next_state], dim=1)
        # Combine initial state and next states into a state trajectory.
        state_trajectory = state_trajectory.reshape(-1, self.state_dim)

        return dict(state_trajectory=state_trajectory, **kwargs)

    def post_process_pred(self,
                          obs_state_traj: torch.Tensor,
                          obs_state_traj_prime: torch.Tensor) -> dict[str, torch.Tensor]:
        # Reshape to (batch, time, obs_state_dim)
        obs_traj = torch.reshape(obs_state_traj, (self._batch_dim, -1, self.obs_state_dim))
        obs_state = obs_traj[:, 0, ...]  # First "time step" is the initial state observation
        next_obs_state = obs_traj[:, 1:, ...]  # The Rest is the observations of the next states,

        obs_traj_prime = torch.reshape(obs_state_traj_prime, (self._batch_dim, -1, self.obs_state_dim))
        obs_state_prime = obs_traj_prime[:, 0, ...]  # First "time step" is the initial state observation
        next_obs_state_prime = obs_traj_prime[:, 1:, ...]  # The Rest is the observations of the next states,

        return dict(obs_state=obs_state, next_obs_state=next_obs_state,
                    obs_state_prime=obs_state_prime, next_obs_state_prime=next_obs_state_prime)

    def forward(self,
                state: torch.Tensor, n_steps: int = 0, **kwargs) -> [dict[str, torch.Tensor]]:
        """ Forward pass of the dynamics model, producing a prediction of the next `n_steps` states.
        Args:
            state: Initial state of the system.
            n_steps: Number of steps to predict.
            **kwargs: Auxiliary arguments

        Returns:
            predictions (dict): A dictionary containing the predicted states under the key 'state' and
            potentially other auxiliary measurements.
        """
        # Apply any required pre-processing to the initial state `w0` and state trajectory `[w1, w2, ..., wH]`
        input = self.pre_process_state(state, **kwargs)  # Returns a w:(batch * H, state_dim) tensor
        # Compute the trajectory of observable states x(w)=[x(w0),...] and x'(w)=[x'(w0),...] producing
        projections = self.project(**input)
        # Post-process observation state trajectories
        output = self.post_process_pred(**projections)

        if n_steps > 0 and self.transfer_op is not None:
            obs_state = output['obs_state']
            pred_next_obs_state = self.forcast(obs_state, n_steps=n_steps)
            output.update(pred_next_obs_state)
        return output

    @torch.no_grad()
    def forcast(self,
                obs_state: torch.Tensor, n_steps: int = 1, **kwargs) -> [dict[str, torch.Tensor]]:
        """ This function uses the empirical transfer operator to compute forcast the observable state.

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
            raise RuntimeError("The transfer operator not approximated yet. Call `approximate_transfer_operator`")

        # Use the empirical transfer operator to compute the maximum likelihood prediction of the trajectory
        pred_next_obs_state = [obs_state]
        # The transfer operator of this Isotypic subspace
        for step in range(n_steps):
            # Compute the next state prediction s_t+1 = K @ s_t
            next_state = (self.transfer_op @ pred_next_obs_state[-1].T).T
            pred_next_obs_state.append(next_state)

        pred_next_obs_state = torch.stack(pred_next_obs_state[1:], dim=1)
        return dict(pred_next_obs_state=pred_next_obs_state)

    def compute_loss_and_metrics(self,
                                 obs_state: torch.Tensor,
                                 next_obs_state: torch.Tensor,
                                 obs_state_prime: Optional[torch.Tensor] = None,
                                 next_obs_state_prime: Optional[torch.Tensor] = None,
                                 pred_next_obs_state: Optional[torch.Tensor] = None,
                                 **kwargs
                                 ) -> (torch.Tensor, dict[str, torch.Tensor]):

        # Compute Covariance and Cross-Covariance operators for the observation state space.
        # Spectral and Projection scores, and CK loss terms.
        obs_space_metrics = obs_state_space_metrics(obs_state=obs_state,
                                                    next_obs_state=next_obs_state,
                                                    representation=None,
                                                    max_ck_window_length=self.max_ck_window_length,
                                                    ck_w=self.ck_w)
        non_nans = lambda x: torch.logical_not(torch.isnan(x))
        for metric, vals in obs_space_metrics.items():
            obs_space_metrics[metric] = vals[non_nans(vals)]

        loss = self.compute_loss(spectral_score=torch.mean(obs_space_metrics['spectral_score']),
                                 ck_score=torch.mean(obs_space_metrics['ck_score']),
                                 orth_reg=torch.mean(obs_space_metrics['orth_reg']))

        # Include prediction metrics if available
        if pred_next_obs_state is not None:
            assert pred_next_obs_state.shape == next_obs_state.shape
            pred_loss, pred_metrics = forecasting_loss_and_metrics(state_gt=next_obs_state,
                                                                   state_pred=pred_next_obs_state,
                                                                   prefix='obs_')
            pred_metrics['obs_pred_loss'] = pred_loss
            cond_num_transfer_op = torch.linalg.cond(self.transfer_op)
            pred_metrics['cond_num_transfer_op'] = cond_num_transfer_op
            obs_space_metrics.update(pred_metrics)
            if log.level == logging.DEBUG and cond_num_transfer_op > 100:
                log.warning(f"Condition number of transfer operator: {cond_num_transfer_op:.2f}.")

        return loss, obs_space_metrics

    def compute_loss(self, spectral_score, ck_score, orth_reg):
        if self.ck_w == 0.0 or np.isclose(self.ck_w, 0.0):
            loss = - (spectral_score - self.orthonormal_w * orth_reg)
        else:
            loss = - (ck_score - self.orthonormal_w * orth_reg)

        assert not torch.isnan(loss), f"Loss is NaN."
        return loss

    @torch.no_grad()
    def eval_metrics(self,
                     obs_state: torch.Tensor,
                     next_obs_state: torch.Tensor,
                     obs_state_prime: Optional[torch.Tensor] = None,
                     next_obs_state_prime: Optional[torch.Tensor] = None,
                     pred_next_obs_state: Optional[torch.Tensor] = None,
                     state: Optional[torch.Tensor] = None,
                     next_state: Optional[torch.Tensor] = None,
                     pred_next_state: Optional[torch.Tensor] = None,
                     **kwargs
                     ) -> (dict[str, Figure], dict[str, torch.Tensor]):

        obs_state_traj = torch.cat([torch.unsqueeze(obs_state, dim=1), next_obs_state],
                                   dim=1).detach().cpu().numpy()
        pred_obs_state_traj = torch.cat([torch.unsqueeze(obs_state, dim=1), pred_next_obs_state],
                                        dim=1).detach().cpu().numpy()

        state_traj = torch.cat([torch.unsqueeze(state, dim=1), next_state],
                               dim=1).detach().cpu().numpy()
        if pred_next_state is not None:
            pred_state_traj = torch.cat([torch.unsqueeze(state, dim=1), pred_next_state],
                                        dim=1).detach().cpu().numpy()
        else:
            pred_state_traj = None

        fig = plot_two_panel_trajectories(state_trajs=state_traj,
                                          pred_state_trajs=pred_state_traj,
                                          obs_state_trajs=obs_state_traj,
                                          pred_obs_state_trajs=pred_obs_state_traj,
                                          dt=self.dt,
                                          n_trajs_to_show=5)
        figs = dict(prediction=fig)
        metrics = None
        return figs, metrics

    @torch.no_grad()
    def approximate_transfer_operator(self, train_data_loader: DataLoader):
        """ Compute the empirical transfer operator using the standard full-rank DMD algorithm.

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

        pred = self(**train_data)

        obs_state = pred['obs_state']
        next_obs_state = torch.squeeze(pred['next_obs_state'])

        # Generate the data matrices of x(w_t) and x(w_t+1)
        X = obs_state.T.conj()  # (obs_state_dim, n_samples)
        X_prime = next_obs_state.T.conj()  # (obs_state_dim, n_samples)

        empirical_transfer_op = self.dmd_algorithm(X, X_prime)
        # a = empirical_transfer_op.detach().cpu().numpy()
        self.transfer_op = empirical_transfer_op
        # dmd_algorithm = DMD(reduced_rank=False,
        #                     rank=self.obs_state_dim,
        #                     tikhonov_reg=0)
        # assert X.ndim == 2 and X_prime.ndim == 2
        # assert X.shape == X_prime.shape and X.shape[-1] == self.obs_state_dim
        # dmd_algorithm.fit(X, X_prime)
        # Compute the one-step empirical transfer operator.
        # U = dmd_algorithm.U
        # C_XY = dmd_algorithm.cov_XY
        # empirical_transfer_op = np.linalg.multi_dot([U.T, C_XY, U])

    @staticmethod
    def _full_rank_lstsq(X: torch.Tensor, X_prime: torch.Tensor, **kwargs) -> torch.Tensor:
        """ Compute the least squares solution of the linear system `X' = A·X`. Assuming full rank X and A.
        Args:
            X: (state_dim, n_samples) Data matrix of the initial states.
            X_prime: (state_dim, n_samples) Data matrix of the next states.
        Returns:
            A: (state_dim, state_dim) Least squares solution of the linear system `X' = A·X`.
        """
        assert X.ndim == 2 and X.shape == X_prime.shape and X.shape[0] <= X.shape[1], \
            f"X: {X.shape}, X_prime: {X_prime.shape}. Expected data matrices (state_dim, n_samples)"

        # a = X[:, :10].detach().cpu().numpy().T
        # b = X_prime[:, :10].detach().cpu().numpy().T

        # Torch convention uses X':(n_samples, state_dim) and X:(n_samples, state_dim) to solve the least squares
        # problem for `X' = X·A`, instead of our convention `X' = A·X`. So we have to do the appropriate transpose.
        result = torch.linalg.lstsq(X.T, X_prime.T, rcond=None)
        A = result.solution.T
        return A

    def get_metric_labels(self) -> list[str]:
        return ['pred_loss', 'S_score', 'ck_score', 'P_score', 'reg_orthonormal']

    def get_hparams(self):
        return {'encoder_backbone': self.projection_backbone.get_hparams(),
                'obs_fn':           self.obs_state_fn.get_hparams(),
                'aux_obs_fn':       self.obs_state_fn_prime.get_hparams(),
                }


if __name__ == "__main__":
    torch.set_printoptions(precision=3)
    seed_everything(42)
    path_to_data = Path('data')
    assert path_to_data.exists(), f"Invalid Dataset path {path_to_data.absolute()}"

    log.setLevel(logging.DEBUG)
    # Find all dynamic systems recordings
    path_to_data /= 'linear_system'
    path_to_dyn_sys_data = set([a.parent for a in list(path_to_data.rglob('*train.pkl'))])
    # Select a dynamical system
    mock_path = path_to_dyn_sys_data.pop()

    pred_horizon = 100
    batch_size = 10
    device = torch.device('cuda:0')
    data_module = DynamicsDataModule(data_path=mock_path,
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
    num_encoder_layers = 4

    state_type = data_module.state_field_type
    obs_state_dimension = state_type.size * 1
    num_encoder_hidden_neurons = obs_state_dimension * 2
    max_ck_window_length = 6

    dp_net = DPNet(state_dim=data_module.state_field_type.size,
                   obs_state_dim=obs_state_dimension,
                   num_encoder_layers=num_encoder_layers,
                   num_encoder_hidden_neurons=num_encoder_hidden_neurons,
                   max_ck_window_length=max_ck_window_length,
                   activation=torch.nn.Tanh,
                   bias=False,
                   batch_norm=False)

    dp_net.to(device)

    dp_net.approximate_transfer_operator(data_module.predict_dataloader())

    start_time = time.time()
    profiler = cProfile.Profile()
    profiler.enable()
    for i, batch in tqdm(enumerate(data_module.train_dataloader())):
        for k, v in batch.items():
            batch[k] = v.to(device)

        state, next_state = batch['state'], batch['next_state']
        n_steps = batch['next_state'].shape[1]

        # Test pre-processing function
        batched_state_traj = dp_net.pre_process_state(**batch)['state_trajectory']
        assert batched_state_traj.shape == (
            batch_size * (pred_horizon + 1), state.shape[-1]), f"state_traj: {batched_state_traj.shape}"

        state_traj_non_flat = torch.reshape(batched_state_traj, (batch_size, pred_horizon + 1, state.shape[-1]), )
        rec_state = state_traj_non_flat[:, 0]
        rec_next_state = state_traj_non_flat[:, 1:]
        assert rec_state.shape == state.shape, f"rec_state: {rec_state.shape}"
        assert torch.allclose(rec_state, state), f"rec_state: {rec_state - state}"

        assert rec_next_state.shape == next_state.shape, f"rec_next_state: {rec_next_state.shape}"
        assert torch.allclose(rec_next_state, next_state), f"rec_next_state: {rec_next_state - next_state}"

        # Test forward pass
        out = dp_net(**batch, n_steps=n_steps)

        # Test loss and metrics
        loss, metrics = dp_net.compute_loss_and_metrics(**out)
        figs, _ = dp_net.eval_metrics(**batch, **out)
        figs['prediction'].show()
        print(metrics.get('pred_loss', None))

        if i > 1:
            break
    profiler.disable()

    # print(f"Computing forward pass and loss/metrics for {id} batches took {time.time() - start_time:.2f}[s]"
    #       f"({(time.time() - start_time) / i:.2f} seconds per batch for {pred_horizon} steps in pred horizon)")

    # Create a pstats object
    stats = pstats.Stats(profiler)

    # Sort stats by the cumulative time spent in the function
    stats.sort_stats('cumulative')

    # Print only the info for the functions defined in your script
    # Assuming your script's name is 'your_script.py'
    stats.print_stats('koopman_robotics')
