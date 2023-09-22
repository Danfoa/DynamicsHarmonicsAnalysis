import logging
from typing import Optional, Tuple, Union
import torch
from plotly.graph_objs import Figure
from torch import Tensor

from nn.LinearDynamics import LinearDynamics
from nn.latent_markov_dynamics import LatentMarkovDynamics
from nn.markov_dynamics import MarkovDynamics
from nn.mlp import MLP
from utils.losses_and_metrics import forecasting_loss_and_metrics, obs_state_space_metrics
from utils.mysc import batched_to_flat_trajectory, traj_from_states
from utils.plotting import plot_system_2D, plot_system_3D, plot_two_panel_trajectories

log = logging.getLogger(__name__)


class DAE(LatentMarkovDynamics):
    _default_obs_fn_params = dict(
        num_layers=4,
        num_hidden_units=128,
        activation=torch.nn.ReLU,
        batch_norm=True,
        bias=False,
        init_mode='fan_in',
        )

    def __init__(
            self,
            state_dim: int,
            obs_state_dim: int,
            dt: Union[float, int] = 1,
            obs_pred_w: float = 0.1,
            orth_w: float = 0.1,
            corr_w: float = 0.0,
            obs_fn_params: Optional[dict] = None,
            **markov_dyn_params
            ):
        self.state_dim, self.obs_state_dim = state_dim, obs_state_dim
        self.dt = dt
        self.orth_w = orth_w
        self.obs_pred_w = obs_pred_w
        self.corr_w = corr_w

        _obs_fn_params = self._default_obs_fn_params.copy()
        if obs_fn_params is not None:
            _obs_fn_params.update(obs_fn_params)

        # Build the observation function and its inverse
        obs_fn = self.build_obs_fn(**_obs_fn_params)
        inv_obs_fn = self.build_inv_obs_fn(**_obs_fn_params)
        # Define the linear dynamics of the observable state space
        obs_state_dym = self.build_obs_dyn_module()
        # Variable holding the transfer operator used to evolve the observable state in time.
        # Initialize the base class
        super(DAE, self).__init__(obs_fn=obs_fn,
                                  inv_obs_fn=inv_obs_fn,
                                  obs_state_dynamics=obs_state_dym,
                                  state_dim=state_dim,
                                  obs_state_dim=obs_state_dim,
                                  dt=dt,
                                  **markov_dyn_params)

    def forecast(self, state: Tensor, n_steps: int = 1, **kwargs) -> [dict[str, Tensor]]:
        """Forward pass of the dynamics model, producing a prediction of the next `n_steps` states.

        This function uses the empirical transfer operator to compute forcast the observable state.
        Args:
            state: (batch_dim, obs_state_dim) Initial observable state of the system.
            n_steps: Number of steps to predict.
            **kwargs:
        Returns:
            pred_next_obs_state: (batch_dim, n_steps, obs_state_dim) Predicted observable state.
        """
        assert state.shape[-1] == self.state_dim, f"Invalid state: {state.shape}. Expected (batch, {self.state_dim})"
        time_horizon = n_steps + 1

        obs_state = self.obs_fn(state)

        pred_obs_state_traj = self.obs_state_dynamics.forcast(state=obs_state, n_steps=n_steps)

        pred_state_traj = self.inv_obs_fn(pred_obs_state_traj)

        assert pred_state_traj.shape == (self._batch_size, time_horizon, self.state_dim), \
            f"{pred_state_traj.shape}!=({self._batch_size}, {time_horizon}, {self.state_dim})"
        assert pred_obs_state_traj.shape == (self._batch_size, time_horizon, self.obs_state_dim), \
            f"{pred_obs_state_traj.shape}!=({self._batch_size}, {time_horizon}, {self.obs_state_dim})"
        return pred_state_traj, pred_obs_state_traj

    def post_process_obs_state(self, obs_state_traj: Tensor, pred_state_one_step: Tensor) -> dict[str, Tensor]:
        """ Post-process the predicted observable state trajectory given by the observable state dynamics.

        Args:
            obs_state_traj: (batch, time, obs_state_dim) Trajectory of the predicted (time -1) observable states
             predicted by the transfer operator.
            pred_state_one_step: (batch, time, obs_state_dim) Trajectory of the predicted one-step ahead (time)
             observable states predicted by the transfer operator.
            **kwargs:
        Returns:
            Dictionary containing
                - pred_obs_state_traj: (batch * time, obs_state_dim) Trajectory
                - pred_obs_state_one_step: (batch, time, obs_state_dim) Trajectory
        """
        batched_pred_obs_state_traj = batched_to_flat_trajectory(obs_state_traj)
        return dict(pred_obs_state_traj=batched_pred_obs_state_traj,
                    pred_obs_state_one_step=pred_state_one_step)

    def compute_loss_and_metrics(self,
                                 obs_state_traj: Tensor,
                                 pred_obs_state_traj: Optional[Tensor] = None,
                                 pred_obs_state_one_step: Optional[Tensor] = None,
                                 state: Optional[Tensor] = None,
                                 next_state: Optional[Tensor] = None,
                                 pred_state_traj: Optional[Tensor] = None,
                                 ) -> (Tensor, dict[str, Tensor]):

        state_traj = traj_from_states(state, next_state)
        time_horizon = state_traj.shape[1]

        obs_space_metrics = obs_state_space_metrics(obs_state_traj=obs_state_traj,
                                                    max_ck_window_length=time_horizon - 1)

        state_pred_loss, state_pred_metrics = forecasting_loss_and_metrics(gt=state_traj,
                                                                           pred=pred_state_traj)
        obs_pred_loss, obs_pred_metrics = forecasting_loss_and_metrics(gt=obs_state_traj,
                                                                       pred=pred_obs_state_traj,
                                                                       prefix='obs_')

        loss = self.compute_loss(state_pred_loss=state_pred_loss,
                                 obs_pred_loss=obs_pred_loss,
                                 orth_reg=obs_space_metrics["orth_reg"],
                                 spectral_score=obs_space_metrics["spectral_score"],
                                 corr_score=obs_space_metrics["corr_score"])

        metrics = state_pred_metrics
        metrics.update(obs_pred_metrics)
        metrics.update(obs_space_metrics)

        return loss, metrics

    def compute_loss(self,
                     state_pred_loss: Tensor, obs_pred_loss: Tensor,
                     orth_reg: Tensor, spectral_score: Tensor, corr_score: Tensor):
        # transfer_op_inv_score = spectral_score if self.use_spectral_score else corr_score
        transfer_op_inv_score = corr_score
        transfer_op_inv_score = torch.mean(transfer_op_inv_score)  # / self.obs_state_dim

        state_loss = torch.mean(state_pred_loss)
        obs_loss = self.obs_pred_w * torch.mean(obs_pred_loss)
        orth_regularization = (self.orth_w * self.obs_state_dim) * torch.mean(orth_reg)
        corr_score = self.corr_w * transfer_op_inv_score
        return state_loss + obs_loss + orth_regularization + corr_score

    @torch.no_grad()
    def eval_metrics(self,
                     state: Tensor,
                     next_state: Tensor,
                     obs_state_traj: Tensor,
                     obs_state_traj_aux: Optional[Tensor] = None,
                     pred_state_traj: Optional[Tensor] = None,
                     pred_obs_state_traj: Optional[Tensor] = None,
                     pred_obs_state_one_step: Optional[Tensor] = None,
                     ) -> (dict[str, Figure], dict[str, Tensor]):

        state_traj = traj_from_states(state, next_state)

        # Detach all arguments and ensure they are in CPU
        state_traj = state_traj.detach().cpu().numpy()
        obs_state_traj = obs_state_traj.detach().cpu().numpy()
        if obs_state_traj_aux is not None:
            obs_state_traj_aux = obs_state_traj_aux.detach().cpu().numpy()
        if pred_state_traj is not None:
            pred_state_traj = pred_state_traj.detach().cpu().numpy()
        if pred_obs_state_traj is not None:
            pred_obs_state_traj = pred_obs_state_traj.detach().cpu().numpy()

        fig = plot_two_panel_trajectories(state_trajs=state_traj,
                                          pred_state_trajs=pred_state_traj,
                                          obs_state_trajs=obs_state_traj,
                                          pred_obs_state_trajs=pred_obs_state_traj,
                                          dt=self.dt,
                                          n_trajs_to_show=5)
        if self.obs_state_dim == 3:
            fig_3ds = plot_system_3D(trajectories=state_traj, secondary_trajectories=pred_state_traj,
                                     title='state_traj', num_trajs_to_show=20)
            fig_3do = plot_system_3D(trajectories=obs_state_traj, secondary_trajectories=pred_obs_state_traj,
                                     title='obs_state', num_trajs_to_show=20)
            figs = dict(prediction=fig, state=fig_3ds, obs_state=fig_3do)
        elif self.obs_state_dim == 2:
            fig_2ds = plot_system_2D(trajs=state_traj, secondary_trajs=pred_state_traj, alpha=0.2,
                                     num_trajs_to_show=10)
            fig_2do = plot_system_2D(trajs=obs_state_traj, secondary_trajs=pred_obs_state_traj, alpha=0.2,
                                     num_trajs_to_show=10)
            figs = dict(prediction=fig, state=fig_2ds, obs_state=fig_2do)
        else:
            figs = dict(prediction=fig)

        metrics = None
        return figs, metrics

    def build_obs_fn(self, num_layers, **kwargs):
        return MLP(in_dim=self.state_dim, out_dim=self.obs_state_dim, num_layers=num_layers, **kwargs)

    def build_inv_obs_fn(self, num_layers, **kwargs):
        return MLP(in_dim=self.obs_state_dim, out_dim=self.state_dim, num_layers=num_layers, **kwargs)

    def build_obs_dyn_module(self) -> MarkovDynamics:
        return LinearDynamics(state_dim=self.obs_state_dim, dt=self.dt, trainable=True)


