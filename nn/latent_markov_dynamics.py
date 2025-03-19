import logging
from typing import Optional, Union

import torch.nn
from escnn.group import Representation
from plotly.graph_objs import Figure
from torch import Tensor

from nn.markov_dynamics import MarkovDynamics
from utils.losses_and_metrics import forecasting_loss_and_metrics
from utils.mysc import batched_to_flat_trajectory, flat_to_batched_trajectory, traj_from_states

log = logging.getLogger(__name__)


class LatentMarkovDynamics(MarkovDynamics):
    def __init__(
        self,
        obs_fn: torch.nn.Module,
        inv_obs_fn: torch.nn.Module,
        obs_state_dynamics: MarkovDynamics,
        state_dim: Optional[int] = None,
        state_rep: Optional[Representation] = None,
        obs_state_dim: Optional[int] = None,
        obs_state_rep: Optional[Representation] = None,
        dt: Optional[Union[float, int]] = 1,
        state_change_of_basis: Optional[Tensor] = None,
        state_inv_change_of_basis: Optional[Tensor] = None,
    ):
        assert obs_state_dim is not None or obs_state_rep is not None, (
            "Either obs_state_dim or obs_state_rep must be provided"
        )

        super().__init__(
            state_dim=state_dim,
            state_rep=state_rep,
            dt=dt,
            state_change_of_basis=state_change_of_basis,
            state_inv_change_of_basis=state_inv_change_of_basis,
        )
        self.obs_state_dim = obs_state_dim if obs_state_dim is not None else obs_state_rep.size
        self.obs_state_rep = obs_state_rep
        self.obs_fn = obs_fn
        self.inv_obs_fn = inv_obs_fn
        self.obs_space_dynamics = obs_state_dynamics

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
        assert "obs_state_traj" in pre_obs_fn_output, f"Missing 'obs_state_traj' in {pre_obs_fn_output}"
        obs_state_traj = pre_obs_fn_output.pop("obs_state_traj")

        # Evolution of observable states ===============================================
        # Evolve the observable state with the current observable dynamics model.
        obs_dyn_output = self.obs_space_dynamics(
            state=obs_state_traj[:, 0, :], next_state=obs_state_traj[:, 1:, :], **pre_obs_fn_output
        )
        obs_dyn_output = {k.replace("state", "obs_state"): v for k, v in obs_dyn_output.items()}
        pred_obs_state_traj = obs_dyn_output.pop("pred_obs_state_traj")

        # Observation function inversion ===============================================
        # This post-processing of observables ensures the input to the inverse function is of the correct shape.
        # Predicted trajectory of observable state in observable space
        post_pred_obs_dyn_output = self.post_process_obs_state(pred_obs_state_traj)
        # Predicted trajectory of the system's state in the original state space
        pred_state_traj = self.inv_obs_fn(post_pred_obs_dyn_output.pop("obs_state_traj"))
        # Ground-truth trajectory of observable state in observable space
        post_obs_dyn_output = self.post_process_obs_state(obs_state_traj)
        # Reconstruction of the system's state in the original state space
        rec_state_traj = self.inv_obs_fn(post_obs_dyn_output.pop("obs_state_traj"))

        # Apply the required post-processing of the state.
        pred_state_traj = self.post_process_state(pred_state_traj)
        rec_state_traj = self.post_process_state(rec_state_traj)

        # Sanity checks of shapes.
        self.check_state_traj_shape(
            pred_state_traj=pred_state_traj,
            rec_state_traj=rec_state_traj,
            time_horizon=time_horizon,
            state_dim=self.state_dim,
        )
        self.check_state_traj_shape(
            obs_state_traj=obs_state_traj,
            pred_obs_state_traj=pred_obs_state_traj,
            time_horizon=time_horizon,
            state_dim=self.obs_state_dim,
        )
        return dict(
            obs_state_traj=obs_state_traj,
            pred_obs_state_traj=pred_obs_state_traj,
            pred_state_traj=pred_state_traj,
            rec_state_traj=rec_state_traj,
            **pre_obs_fn_output,
            **obs_dyn_output,
        )

    def pre_process_obs_state(self, obs_state_traj: Tensor) -> dict[str, Tensor]:
        """Apply transformations to the observable state trajectory.

        Args:
            obs_state_traj: (batch * time, obs_state_dim) or (batch, time, obs_state_dim)

        Returns:
            Directory containing
                - obs_state_traj: (batch, time, obs_state_dim) tensor.
                - other: Other observations required for training of the model.

        """
        if len(obs_state_traj.shape) == 2:
            obs_state_traj = flat_to_batched_trajectory(
                obs_state_traj, batch_size=self._batch_size, state_dim=self.obs_state_dim
            )
        elif len(obs_state_traj.shape) == 3:
            pass
        else:
            raise RuntimeError(
                f"Invalid observable state trajectory shape {obs_state_traj.shape}. Expected "
                f"(batch, time, {self.obs_state_dim}) or (batch * time, {self.obs_state_dim})"
            )

        return dict(obs_state_traj=obs_state_traj)

    def post_process_obs_state(self, obs_state_traj: Tensor, **kwargs) -> dict[str, Tensor]:
        """Post-process the predicted observable state trajectory given by the observable state dynamics.

        Args:
            obs_state_traj: (batch, time, obs_state_dim) Trajectory of the predicted (time -1) observable states
             predicted by the transfer operator.
            **kwargs:
        Returns:
            Dictionary contraining
                - pred_obs_state_traj: (batch, time, obs_state_dim) Trajectory

        """
        flat_obs_state_traj = batched_to_flat_trajectory(obs_state_traj)
        return dict(obs_state_traj=flat_obs_state_traj)

    def compute_loss_and_metrics(
        self,
        state: Tensor,
        next_state: Tensor,
        pred_state_traj: Tensor,
        obs_state_traj: Tensor,
        pred_obs_state_traj: Tensor,
        rec_state_traj: Tensor,
    ) -> (Tensor, dict[str, Tensor]):
        """Compute the loss and metrics for the predicted state trajectory. You should probably override this."""
        state_traj = traj_from_states(state, next_state)

        obs_pred_loss, obs_pred_metrics = forecasting_loss_and_metrics(
            gt=obs_state_traj, pred=pred_obs_state_traj, prefix="obs_pred"
        )

        state_pred_loss, state_pred_metrics = forecasting_loss_and_metrics(
            gt=state_traj, pred=pred_state_traj, prefix="state_pred"
        )

        state_rec_loss, state_rec_metrics = forecasting_loss_and_metrics(
            gt=state_traj, pred=rec_state_traj, prefix="state_rec"
        )

        metrics = dict(
            obs_pred_loss=obs_pred_loss,
            state_rec_loss=state_rec_loss,
            state_pred_loss=state_pred_loss,
            **obs_pred_metrics,
            **state_pred_metrics,
            **state_rec_metrics,
        )

        return state_pred_loss, metrics

    @torch.no_grad()
    def eval_metrics(
        self,
        state: Tensor,
        next_state: Tensor,
        obs_state_traj: Tensor,
        obs_state_traj_aux: Optional[Tensor] = None,
        pred_state_traj: Optional[Tensor] = None,
        rec_state_traj: Optional[Tensor] = None,
        pred_obs_state_one_step: Optional[Tensor] = None,
        pred_obs_state_traj: Optional[Tensor] = None,
    ) -> (dict[str, Figure], dict[str, Tensor]):
        state_traj = traj_from_states(state, next_state)

        if obs_state_traj_aux is None and pred_obs_state_one_step is not None:
            obs_state_traj_aux = pred_obs_state_one_step

        # Detach all arguments and ensure they are in CPU
        state_traj = state_traj.detach().cpu().numpy()
        obs_state_traj = obs_state_traj.detach().cpu().numpy()
        if obs_state_traj_aux is not None:
            obs_state_traj_aux = obs_state_traj_aux.detach().cpu().numpy()
        if pred_state_traj is not None:
            pred_state_traj = pred_state_traj.detach().cpu().numpy()
        if pred_obs_state_traj is not None:
            pred_obs_state_traj = pred_obs_state_traj.detach().cpu().numpy()

        from utils.plotting import plot_two_panel_trajectories
        fig = plot_two_panel_trajectories(
            state_trajs=state_traj,
            pred_state_trajs=pred_state_traj,
            obs_state_trajs=obs_state_traj,
            pred_obs_state_trajs=pred_obs_state_traj,
            dt=self.dt,
            n_trajs_to_show=5,
        )
        figs = dict(prediction=fig)
        if self.obs_state_dim == 3:
            from utils.plotting import plot_system_3D
            fig_3do = plot_system_3D(
                trajectories=obs_state_traj,
                secondary_trajectories=pred_obs_state_traj,
                title="obs_state",
                num_trajs_to_show=20,
            )
            if obs_state_traj_aux is not None:
                fig_3do = plot_system_3D(
                    trajectories=obs_state_traj_aux,
                    legendgroup="aux",
                    traj_colorscale="solar",
                    num_trajs_to_show=20,
                    fig=fig_3do,
                )
            figs["obs_state"] = fig_3do
        if self.state_dim == 3:
            fig_3ds = plot_system_3D(
                trajectories=state_traj,
                secondary_trajectories=pred_state_traj,
                title="state_traj",
                num_trajs_to_show=20,
            )
            figs["state"] = fig_3ds

        if self.obs_state_dim == 2:
            from utils.plotting import plot_system_2D
            fig_2do = plot_system_2D(
                trajs=obs_state_traj, secondary_trajs=pred_obs_state_traj, alpha=0.2, num_trajs_to_show=10
            )
            if obs_state_traj_aux is not None:
                fig_2do = plot_system_2D(trajs=obs_state_traj_aux, legendgroup="aux", num_trajs_to_show=10, fig=fig_2do)
            figs["obs_state"] = fig_2do
        if self.state_dim == 2:
            fig_2ds = plot_system_2D(trajs=state_traj, secondary_trajs=pred_state_traj, alpha=0.2, num_trajs_to_show=10)
            figs["state"] = fig_2ds

        metrics = None
        return figs, metrics

    def get_hparams(self):
        hparams = {}
        if hasattr(self.obs_fn, "get_hparams"):
            obs_fn_params = self.obs_fn.get_hparams()
            hparams["obs_fn"] = obs_fn_params
        if hasattr(self.inv_obs_fn, "get_hparams"):
            inv_obs_fn_params = self.inv_obs_fn.get_hparams()
            hparams["inv_obs_fn"] = inv_obs_fn_params
        if hasattr(self.obs_space_dynamics, "get_hparams"):
            obs_state_dyn_params = self.obs_space_dynamics.get_hparams()
            hparams["obs_state_dynamics"] = obs_state_dyn_params
        hparams["state_dim"] = self.state_dim
        hparams["obs_state_dim"] = self.obs_state_dim
        hparams["dt"] = self.dt
        return hparams

    def __repr__(self):
        try:
            str = super().__repr__()
        except:
            str = ""
        return (
            f"{str} \n {self.__class__.__name__}(state_dim={self.state_dim} "
            f"obs_state_dim={self.obs_state_dim}, dt={self.dt:.1e})"
        )
