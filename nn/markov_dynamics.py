from typing import Iterable
import torch


class MarkovDynamicsModule(torch.nn.Module):

    def __init__(self, state_dim: int, dt: float, **kwargs):
        super().__init__()
        self.state_dim = state_dim
        self.dt = dt

    def forward(self, state: torch.Tensor, n_steps: int = 1, **kwargs) -> [dict[str, torch.Tensor]]:
        """ Forward pass of the dynamics model, producing a prediction of the next `n_steps` states.
        Args:
            state: Initial state of the system
            n_steps: Number of steps to predict
            **kwargs: Auxiliary arguments

        Returns:
            predictions (dict): A dictionary containing the predicted states under the key 'state' and
            potentially other auxiliary measurements.
        """
        # Apply any required pre-processing to the state
        input = self.pre_process_state(state, **kwargs)
        # Evolve dynamics
        predictions = self.forcast(**input, n_steps=n_steps, )
        # Post-process predictions
        predictions = self.post_process_pred(predictions)
        return predictions

    def forcast(self, state: torch.Tensor, n_steps: int = 1, **kwargs) -> [dict[str, torch.Tensor]]:
        """ Forcasting of dynamics by `n_steps` from initial state `initial_state`.

        Args:
            state: Initial state if the system
            n_steps: Number of steps to predict
            **kwargs: Auxiliary arguments

        Returns:
            predictions (dict): A dictionary containing the predicted states under the key 'next_state' and potentially
            other auxiliary measurements.
        """
        raise NotImplementedError("")

    def pre_process_state(self, state: torch.Tensor, **kwargs) -> dict[str, torch.Tensor]:
        """ Preprocess the state of the system before forcasting dynamics
        Args:
            state (torch.Tensor): [batch, state_dim,] Initial state of the system from which to forcast dynamics
            **kwargs: Additional arguments
        Returns:
            Preprocessed dictionary containing the preprocess state under the `state` key and potentially additional
            measurements.
        """
        return dict(state=state)

    def post_process_pred(self, predictions: dict) -> torch.Tensor:
        return predictions

    def get_metric_labels(self) -> Iterable[str]:
        return ["rec_loss", "pred_loss", "lin_loss", "linf_loss", "rec_loss", "pred_loss", "lin_loss", "linf_loss"]

    @staticmethod
    def compute_loss_and_metrics(predictions: dict[str, torch.Tensor],
                                 ground_truth: dict[str, torch.Tensor]) -> (torch.Tensor, dict[str, torch.Tensor]):
        state_pred = predictions['next_state']
        state_gt = ground_truth['next_state']
        # Compute state squared error over time and the infinite norm of the state dimension over time.
        l2_loss = torch.norm(state_gt - state_pred, p=2, dim=-1)
        linf_loss = torch.norm(state_gt - state_pred, p=torch.inf, dim=-1)
        time_steps = state_gt.shape[1]
        metrics = {}
        if time_steps > 4:
            # Calculate the average prediction error in [25%,50%,75%] of the prediction horizon.
            for quartile in [.25, .5, .75]:
                stop_idx = int(quartile * time_steps)
                l2_loss_quat = l2_loss[:, :stop_idx].mean(dim=-1)
                metrics[f'l2_loss_{int(quartile * 100):d}%'] = l2_loss_quat.mean()
        # pred_horizon = time_steps * self.dt
        metrics.update(linf_loss=linf_loss.mean())
        return l2_loss.mean(), metrics

    def get_metric_labels(self) -> Iterable[str]:
        return ['l2_loss_25%', 'l2_loss_50%', 'l2_loss_75%']
