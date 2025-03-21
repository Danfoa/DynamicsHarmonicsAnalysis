import cProfile
import logging
import math
import pstats
import time
from pathlib import Path
from typing import Optional, Tuple, Union

import escnn
import numpy as np
import torch
from escnn.group import Representation
from escnn.nn import FieldType, GeometricTensor
from lightning import seed_everything
from morpho_symm.nn.EMLP import EMLP
from morpho_symm.utils.rep_theory_utils import isotypic_basis
from torch import Tensor
from tqdm import tqdm

from data.DynamicsDataModule import DynamicsDataModule
from nn.DeepProjections import DPNet
from nn.EquivLinearDynamics import EquivLinearDynamics
from nn.markov_dynamics import MarkovDynamics
from nn.ObservableNet import ObservableNet
from utils.linear_algebra import full_rank_lstsq_symmetric
from utils.losses_and_metrics import (
    iso_metrics_2_obs_space_metrics,
    obs_state_space_metrics,
)

log = logging.getLogger(__name__)


class EquivDPNet(DPNet):
    _default_obs_fn_params = dict(
        num_layers=4,
        num_hidden_units=128,  # Approximate number of neurons in hidden layers. Actual number depends on group order.
        activation="p_elu",
        batch_norm=True,
        bias=False,
        # backbone_layers=-2  # num_layers - 2
    )

    def __init__(
        self,
        state_rep: Representation,
        obs_state_dim: int,
        dt: Union[float, int] = 1,
        obs_fn_params: Optional[dict] = None,
        group_avg_trick: bool = True,
        **dpnet_kwargs,
    ):
        self.symm_group = state_rep.group
        self.gspace = escnn.gspaces.no_base_space(self.symm_group)
        self.group_avg_trick = group_avg_trick
        _obs_fn_params = self._default_obs_fn_params.copy()
        if obs_fn_params is not None:
            _obs_fn_params.update(obs_fn_params)

        # Number of regular fields in obs state and hidden layers of observable network
        multiplicity = math.ceil(obs_state_dim / state_rep.size)
        if multiplicity < 1:
            raise ValueError(f"State-dim:{state_rep.size}, |G|={self.symm_group.order()}, obs_dim:{obs_state_dim}")

        # Find the Isotypic basis of the state space and define the observation space representation as
        # `num_spect_field` copies of state representation (in isotypic basis).
        self.state_iso_reps, self.state_iso_dims = isotypic_basis(
            representation=state_rep, multiplicity=1, prefix="State"
        )
        # Store the change of basis from original input basis to the isotypic basis of the space.
        if np.allclose(Q_iso2state, np.eye(state_rep.size)):
            Q_iso2state, Q_state2iso = None, None
        else:
            Q_iso2state = torch.Tensor(Q_iso2state)
            Q_state2iso = torch.Tensor(np.linalg.inv(Q_iso2state))

        # Define the observation space representation in the isotypic basis.
        self.obs_iso_reps, self.obs_iso_dims, _ = isotypic_basis(
            representation=state_rep, multiplicity=multiplicity, prefix="Obs"
        )
        # Each Field for ESCNN will be an Isotypic Subspace.
        self.state_type = FieldType(self.gspace, [state_rep])
        # Field type on isotypic basis.
        self.state_type_iso = FieldType(self.gspace, [rep_iso for rep_iso in self.state_iso_reps.values()])
        self.obs_state_type = FieldType(self.gspace, [rep_iso for rep_iso in self.obs_iso_reps.values()])

        # Define a dict containing the transfer operator of each Isotypic subspace.
        self.iso_transfer_op = {irrep_id: None for irrep_id in self.obs_iso_reps.keys()}
        self.iso_inverse_projector = {irrep_id: None for irrep_id in self.obs_iso_reps.keys()}
        self.iso_inverse_projector_bias = {irrep_id: None for irrep_id in self.obs_iso_reps.keys()}

        super(EquivDPNet, self).__init__(
            state_dim=state_rep.size,
            obs_state_dim=obs_state_dim,
            dt=dt,
            obs_fn_params=_obs_fn_params,
            obs_state_rep=self.obs_state_type.representation,
            state_change_of_basis=Q_state2iso,
            state_inv_change_of_basis=Q_iso2state,
            **dpnet_kwargs,
        )

    def pre_process_state(self, state: Tensor, next_state: Optional[Tensor] = None) -> GeometricTensor:
        # Change basis to Isotypic basis.
        state_trajectory_iso_basis = super().pre_process_state(state=state, next_state=next_state)
        # Convert to Geometric Tensor
        return self.state_type_iso(state_trajectory_iso_basis)

    def pre_process_obs_state(
        self, obs_state_traj: GeometricTensor, obs_state_traj_prime: GeometricTensor
    ) -> dict[str, Tensor]:
        return super().pre_process_obs_state(obs_state_traj.tensor, obs_state_traj_prime.tensor)

    def post_process_state(self, state_traj: Tensor) -> Tensor:
        state_traj_input_basis = super().post_process_state(state_traj=state_traj)
        return state_traj_input_basis

    def get_obs_space_metrics(self, obs_state_traj: Tensor, obs_state_traj_prime: Optional[Tensor] = None) -> dict:
        # For each Isotypic Subspace, compute empirical Covariance and Cross-Covariance operators.
        # With these, compute spectral, projection scores and orthonormality and Chapman-Kolmogorov regularization.
        iso_spaces_metrics = {irrep_id: {} for irrep_id in self.obs_iso_reps.keys()}
        for irrep_id, iso_rep in self.obs_iso_reps.items():
            rep = iso_rep if irrep_id != self.symm_group.trivial_representation else None  # Check for Trivial IsoSpace
            # Get the projection of the observable state in the isotypic subspace
            # Iso subspace trajectory
            obs_state_traj_iso = obs_state_traj[..., self.obs_iso_dims[irrep_id]]
            obs_state_traj_prime_iso = obs_state_traj_prime[..., self.obs_iso_dims[irrep_id]]

            # Compute Covariance and Cross-Covariance operators for this Isotypic subspace.
            # Spectral and Projection scores, and CK loss terms.
            iso_metrics = obs_state_space_metrics(
                obs_state_traj=obs_state_traj_iso,
                obs_state_traj_aux=obs_state_traj_prime_iso,
                representation=rep if self.group_avg_trick else None,
                max_ck_window_length=self.max_ck_window_length,
            )

            iso_spaces_metrics[irrep_id] = iso_metrics

        # Now use the metrics of each Isotypic observable subspace to compute the loss and metrics of the entire
        # observable space.
        obs_space_metrics = iso_metrics_2_obs_space_metrics(
            iso_spaces_metrics=iso_spaces_metrics, obs_iso_reps=self.obs_iso_reps
        )

        return obs_space_metrics

    def empirical_lin_inverse_projector(self, state: Tensor, obs_state: Tensor):
        """Compute the empirical inverse projector from the observable state to the pre-processed state.

        Args:
            state: (batch, state_dim) Tensor containing the pre-processed state.
            obs_state: (batch, obs_state_dim) Tensor containing the observable state.

        Returns:
            A: (state_dim, obs_state_dim) Tensor containing the empirical inverse projector.
            B: (state_dim, 1) Tensor containing the empirical bias term.
            rec_error: Scalar tensor containing the reconstruction error or "residual".

        """
        # Inverse projector is computed from the observable state to the pre-processed state
        pre_state = self.pre_process_state(state).tensor

        iso_rec_error = []
        # For each Isotypic Subspace, compute the empirical inverse operator with the present observable state space.
        for irrep_id, iso_obs_rep in self.obs_iso_reps.items():
            obs_rep = iso_obs_rep if irrep_id != self.symm_group.identity else None  # Check for Trivial
            state_rep = (
                self.state_iso_reps[irrep_id] if irrep_id != self.symm_group.identity else None
            )  # Check for Trivial

            # Get the projection of the observable state in the isotypic subspace
            state_iso = pre_state[..., self.state_iso_dims[irrep_id]]
            obs_state_iso = obs_state[..., self.obs_iso_dims[irrep_id]]
            # Generate the data matrices of x(w_t) and x(w_t+1)
            X_iso = obs_state_iso.T  # (batch_dim, obs_state_dim)
            Y_iso = state_iso.T  # (state_dim, n_samples)
            A_iso, B_iso = full_rank_lstsq_symmetric(
                X=X_iso,
                Y=Y_iso,
                rep_X=obs_rep if self.group_avg_trick else None,
                rep_Y=state_rep if self.group_avg_trick else None,
                bias=True,
            )
            if B_iso is not None:
                iso_rec_error.append(torch.nn.functional.mse_loss(Y_iso, (A_iso @ X_iso) + B_iso))
            else:
                iso_rec_error.append(torch.nn.functional.mse_loss(Y_iso, A_iso @ X_iso))

            assert A_iso.shape == (state_iso.shape[-1], obs_state_iso.shape[-1]), f"A_iso: {A_iso.shape}"
            self.iso_inverse_projector[irrep_id] = A_iso
            self.iso_inverse_projector_bias[irrep_id] = B_iso

        A = torch.block_diag(*[self.iso_inverse_projector[irrep_id] for irrep_id in self.obs_iso_reps.keys()])
        B = torch.cat([self.iso_inverse_projector_bias[irrep_id] for irrep_id in self.obs_iso_reps.keys()])
        rec_error = torch.sum(torch.Tensor(iso_rec_error)).detach()

        metrics = dict(
            inverse_projector_rank=torch.linalg.matrix_rank(A.detach()).to(torch.float),
            inverse_projector_cond_num=torch.linalg.cond(A.detach()).to(torch.float),
            inverse_projector_error=rec_error,
            inverse_projector_error_dist=torch.Tensor(iso_rec_error).detach(),
        )

        return A, B, metrics

    def build_obs_fn(self, num_layers, **kwargs):
        num_hidden_units = kwargs.get("num_hidden_units")
        activation_type = kwargs.pop("activation")
        act = EMLP.get_activation(
            activation=activation_type, in_type=self.state_type_iso, desired_hidden_units=num_hidden_units
        )

        encoder = EMLP(
            in_type=self.state_type_iso, out_type=act.out_type, num_layers=num_layers, activation=act, **kwargs
        )
        return ObservableNet(
            encoder=encoder, obs_type=self.obs_state_type, explicit_transfer_op=self.explicit_transfer_op
        )

    def build_inv_obs_fn(self, num_layers, linear_decoder: bool, **kwargs):
        if linear_decoder:
            return super().build_inv_obs_fn(num_layers=num_layers, linear_decoder=linear_decoder, **kwargs)
        else:
            return EMLP(in_type=self.obs_state_type, out_type=self.state_type_iso, num_layers=num_layers, **kwargs)

    def build_obs_dyn_module(self) -> MarkovDynamics:
        return EquivLinearDynamics(
            state_type=self.obs_state_type,
            dt=self.dt,
            trainable=False,
            bias=self.enforce_constant_fn,
            group_avg_trick=self.group_avg_trick,
        )

    def __repr__(self):
        str = super().__repr__()
        str.replace("DPNet", "EquivDPNet")
        str += (
            f"\tState Space fields={self.state_type.representations} "
            f"\n\t\tirreps={self.state_type.representation.irreps}"
            f"\n\tObservation Space fields={self.obs_state_type.representations} "
            f"\n\t\tirreps={self.obs_state_type.representation.irreps}"
        )
        return str

    def get_hparams(self):
        hparams = super().get_hparams()
        hparams.update(group=self.symm_group.name, num_iso_spaces=len(self.obs_iso_reps))
        return hparams

    def evaluate_output_shape(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        batch_dim, state_dim = input_shape
        return batch_dim, self.obs_state_type.size


if __name__ == "__main__":
    torch.set_printoptions(precision=3)
    seed_everything(42)
    path_to_data = Path("data")
    assert path_to_data.exists(), f"Invalid Dataset path {path_to_data.absolute()}"

    log.setLevel(logging.DEBUG)
    # Find all dynamic systems recordings
    path_to_data /= "linear_system"
    path_to_dyn_sys_data = set([a.parent for a in list(path_to_data.rglob("*train.pkl"))])
    # Select a dynamical system
    mock_path = path_to_dyn_sys_data.pop()

    pred_horizon = 50
    batch_size = 1024
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
    num_encoder_layers = 4

    state_type = data_module.state_type
    obs_state_dimension = state_type.size * 1
    num_encoder_hidden_neurons = obs_state_dimension * 2
    max_ck_window_length = pred_horizon

    dp_net = EquivDPNet(
        state_type=data_module.state_type,
        obs_state_dim=obs_state_dimension,
        num_layers=num_encoder_layers,
        num_hidden_units=num_encoder_hidden_neurons,
        max_ck_window_length=max_ck_window_length,
        activation=escnn.nn.ReLU,
        bias=False,
        batch_norm=False,
    )

    dp_net.to(device)

    dp_net.approximate_transfer_operator(data_module.predict_dataloader())

    start_time = time.time()
    profiler = cProfile.Profile()
    profiler.enable()
    for i, batch in tqdm(enumerate(data_module.train_dataloader())):
        for k, v in batch.items():
            batch[k] = v.to(device)
        batch_size = batch["state"].shape[0]
        state, next_state = batch["state"], batch["next_state"]
        n_steps = batch["next_state"].shape[1]

        # Test pre-processing function
        batched_state_traj = dp_net.pre_process_state(**batch)["state_trajectory"]

        state_traj_non_flat = torch.reshape(
            batched_state_traj.tensor,
            (batch_size, pred_horizon + 1, state.shape[-1]),
        )
        rec_state = state_traj_non_flat[:, 0]
        rec_next_state = state_traj_non_flat[:, 1:]
        assert rec_state.shape == state.shape, f"rec_state: {rec_state.shape}"
        assert torch.allclose(rec_state, state), f"rec_state: {rec_state - state}"

        assert rec_next_state.shape == next_state.shape, f"rec_next_state: {rec_next_state.shape}"
        assert torch.allclose(rec_next_state, next_state), f"rec_next_state: {rec_next_state - next_state}"

        # Test forward pass
        out = dp_net(**batch, n_steps=n_steps)

        # Test loss and metrics
        loss, metrics = dp_net.compute_loss_and_metrics(**batch, **out)
        figs, val_metrics = dp_net.eval_metrics(**batch, **out)
        figs["prediction"].show()
        print(metrics.get("pred_loss", None))
        if i > 1:
            break
    profiler.disable()

    # print(f"Computing forward pass and loss/metrics for {id} batches took {time.time() - start_time:.2f}[s]"
    #       f"({(time.time() - start_time) / i:.2f} seconds per batch for {pred_horizon} steps in pred horizon)")

    # Create a pstats object
    stats = pstats.Stats(profiler)

    # Sort stats by the cumulative time spent in the function
    stats.sort_stats("cumulative")

    # Print only the info for the functions defined in your script
    # Assuming your script's name is 'your_script.py'
    stats.print_stats("koopman_robotics")
