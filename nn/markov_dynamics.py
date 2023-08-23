from typing import Iterable

import torch
from torch.nn import Module


class MarkovDynamicsModule(Module):

    def __init__(self, state_dim: int, dt: float, **kwargs):
        super().__init__()
        self.state_dim = state_dim
        self.dt = dt

    def forward(self, initial_state: torch.Tensor, n_steps: int = 1, **kwargs) -> [dict[str, torch.Tensor]]:
        """ Forward pass of the dynamics model, producing a prediction of the next `n_steps` states.
        Args:
            initial_state: Initial state if the system
            n_steps: Number of steps to predict
            **kwargs: Auxiliary arguments

        Returns:
            predictions (dict): A dictionary containing the predicted states under the key 'state' and
            potentially other auxiliary measurements.
        """
        # Apply any required pre-processing to the state
        initial_state_pre = self.pre_process_state(initial_state)
        # Evolve dynamics
        predictions = self.forcast(initial_state=initial_state_pre, n_steps=n_steps)
        # Post-process predictions
        state_post = self.post_process_state(predictions['state'])
        predictions['state'] = state_post
        return predictions

    def forcast(self, initial_state: torch.Tensor, n_steps: int = 1, **kwargs) -> [dict[str, torch.Tensor]]:
        """ Forcasting of dynamics by `n_steps` from initial state `initial_state`.

        Args:
            initial_state: Initial state if the system
            n_steps: Number of steps to predict
            **kwargs: Auxiliary arguments

        Returns:
            predictions (dict): A dictionary containing the predicted states under the key 'state' and potentially
            other auxiliary measurements.
        """
        raise NotImplementedError("")

    def pre_process_state(self, state: torch.Tensor, **kwargs) -> torch.Tensor:
        return state

    def post_process_state(self, state: torch.Tensor, **kwargs) -> torch.Tensor:
        return state

    def compute_loss_metrics(self, pred, gt):
        """
        :param x: Batched sequence of consecutive states [x0, x1,..., xT], T=num_steps
        :param z: Batches embeddings/observations of `z = ø(x)` [z0, z1,..., zT]
        :param x_pred: Batches prediction of sequence of consecutive states `xt_pred = ø^-1(K^t•ø(x0))`:
                       [ø^-1(ø(x0)), ø^-1(Kø(x0)), ... ø^-1(K^T•ø(x0))]
        :param z_pred: Batches embeddings/observations of `x_pred` [z0, Kz0,...,K^T•z0]
        :return:
        """
        z_pred = pred['z_pred']
        x_pred = pred['x_pred']
        z = pred['z']
        x_unscaled = gt
        # The output of the dynamics model is already unscaled so we have to unstandarize the dataset samples to compare
        # the results in the appropiate scale.
        # TODO: Make a more elegant solution here
        mean = self._input_mean
        std = self._input_std
        x = (x_unscaled * std) + mean
        # n = torch.norm(x[..., [0, 1]], dim=-1) == 1.0
        nx = x.shape[-1]

        x_err = torch.abs(torch.sub(x, x_pred))  # ∀ t: |x_t - ø^-1(K^t•ø(x_0))|
        z_err = torch.abs(torch.sub(z, z_pred))  # ∀ t: |z_t - K^t•ø(x_0)|

        norm_x_err = torch.norm(x_err, dim=-1, p=2)
        norm_z_err = torch.norm(z_err, dim=-1, p=2)
        # Reconstruction loss of the system state x_0.
        reconstruction_loss = torch.mean(norm_x_err[:, 0])

        # Prediction loss. From the system state compute prediction accuracy from multi-step predictions.
        avg_state_pred_err = torch.mean(norm_x_err[:, 1:], dim=1)

        metrics = {}
        if self.robot is not None:
            nq, nv = self.robot.nq, self.robot.nv
            metrics.update(q_err_rec=x_err[:, 0, :nq].mean(), q_err_pred=x_err[:, 1:, :nq].mean(),
                           dq_err_rec=x_err[:, 0, nq: nq + nv].mean(), dq_err_pred=x_err[:, 1:, nq: nq + nv].mean(),
                           u_err_rec=x_err[:, 0, nq + nv: nx].mean(), u_err_pred=x_err[:, 1:, nq + nv: nx].mean())

        # Linear dynamics of the observable/embedding.
        avg_obs_pred_err = torch.mean(norm_z_err[:, 1:], dim=1)

        # L-inf norm of reconstruction and single step prediction.
        x0, x1 = x[:, 0], x[:, 1]
        x0_rec, x1_pred = x_pred[:, 0], x_pred[:, 1]
        # Linf = ||x_0 - ø^-1(ø(x_0))||_inf + ||x_1 - ø^-1(K•ø(x_0))||_inf
        linf_loss = torch.norm(torch.sub(x0, x0_rec), p=float('inf'), dim=1) + \
                    torch.norm(torch.sub(x1, x1_pred), p=float('inf'), dim=1)

        metrics.update(rec_loss=reconstruction_loss, pred_loss=avg_state_pred_err.mean(),
                       lin_loss=avg_obs_pred_err.mean(), linf_loss=linf_loss.mean())

        loss = self.pred_w * (reconstruction_loss + metrics["pred_loss"]) + metrics["lin_loss"]
        return loss, metrics

    def get_metric_labels(self) -> Iterable[str]:
        return ["rec_loss", "pred_loss", "lin_loss", "linf_loss", "rec_loss", "pred_loss", "lin_loss", "linf_loss"]
