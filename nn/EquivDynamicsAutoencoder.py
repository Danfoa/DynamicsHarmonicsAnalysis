import logging
import math
from typing import Optional, Union

import escnn
import numpy as np
import torch
from escnn.group import Representation
from escnn.nn import FieldType, GeometricTensor
from torch import Tensor

from nn.DynamicsAutoEncoder import DAE
from nn.EquivLinearDynamics import EquivLinearDynamics
from nn.markov_dynamics import MarkovDynamics
from morpho_symm.nn.EMLP import EMLP
from utils.losses_and_metrics import obs_state_space_metrics, iso_metrics_2_obs_space_metrics
from utils.mysc import batched_to_flat_trajectory
from morpho_symm.utils.rep_theory_utils import isotypic_basis

log = logging.getLogger(__name__)


class EquivDAE(DAE):
    _default_obs_fn_params = dict(
        num_layers=4,
        num_hidden_units=128,  # Approximate number of neurons in hidden layers. Actual number depends on group order.
        activation="p_elu",
        batch_norm=True,
        bias=False,
        )

    def __init__(self,
                 state_rep: Representation,
                 obs_state_dim: int,
                 dt: Union[float, int] = 1,
                 obs_fn_params: Optional[dict] = None,
                 group_avg_trick: bool = True,
                 state_dependent_obs_dyn: bool = False,
                 **dae_kwargs):

        self.symm_group = state_rep.group
        self.gspace = escnn.gspaces.no_base_space(self.symm_group)
        self.group_avg_trick = group_avg_trick
        self.state_dependent_obs_dyn = state_dependent_obs_dyn

        if state_dependent_obs_dyn:
            raise NotImplementedError("Some more work is required.")

        _obs_fn_params = self._default_obs_fn_params.copy()
        if obs_fn_params is not None:
            _obs_fn_params.update(obs_fn_params)

        # Number of regular fields in obs state and hidden layers of observable network
        multiplicity = math.ceil(obs_state_dim / state_rep.size)
        if multiplicity < 1:
            raise ValueError(f"State-dim:{state_rep.size}, |G|={self.symm_group.order()}, "
                             f"obs_dim:{obs_state_dim}")

        # Define the observation space representation in the isotypic basis.
        self.obs_iso_reps, self.obs_iso_dims = isotypic_basis(representation=state_rep,
                                                              multiplicity=multiplicity,
                                                              prefix='Obs')
        # Each Field for ESCNN will be an Isotypic Subspace.
        self.state_type = FieldType(self.gspace, [state_rep])
        # Field type on isotypic basis.
        self.obs_state_type = FieldType(self.gspace, [rep_iso for rep_iso in self.obs_iso_reps.values()])

        # Define a dict containing the transfer operator of each Isotypic subspace.
        # self.iso_transfer_op = {irrep_id: None for irrep_id in self.obs_iso_reps.keys()}
        # self.iso_inverse_projector = {irrep_id: None for irrep_id in self.obs_iso_reps.keys()}

        super(EquivDAE, self).__init__(state_dim=self.state_type.size,
                                       obs_state_dim=obs_state_dim,
                                       dt=dt,
                                       obs_fn_params=_obs_fn_params,
                                       obs_state_rep=self.obs_state_type.representation,
                                       **dae_kwargs)

    def pre_process_state(self, state: Tensor, next_state: Optional[Tensor] = None) -> GeometricTensor:
        state_trajectory = super().pre_process_state(state=state, next_state=next_state)
        # Convert to Geometric Tensor
        return self.state_type(state_trajectory)

    def pre_process_obs_state(self, obs_state_traj: GeometricTensor) -> dict[str, Tensor]:
        return super().pre_process_obs_state(obs_state_traj.tensor)

    def post_process_obs_state(self, obs_state_traj: Tensor, **kwargs) -> dict[str, GeometricTensor]:
        """ Post-process the predicted observable state trajectory given by the observable state dynamics.

        Args:
            obs_state_traj: (batch, time, obs_state_dim) Trajectory of the predicted (time -1) observable states
             predicted by the transfer operator.
            **kwargs:
        Returns:
            Dictionary contraining
                - pred_obs_state_traj: (batch * time, obs_state_dim) Geometric Tensor Trajectory
        """
        flat_obs_state_traj = batched_to_flat_trajectory(obs_state_traj)
        return dict(obs_state_traj=self.obs_state_type(flat_obs_state_traj))

    def post_process_state(self, state_traj: GeometricTensor) -> Tensor:
        state_traj = super().post_process_state(state_traj=state_traj.tensor)
        return state_traj

    def build_obs_fn(self, num_layers: int, **kwargs):
        return EMLP(in_type=self.state_type,
                    out_type=self.obs_state_type,
                    num_layers=num_layers,
                    **kwargs)

    def build_inv_obs_fn(self, num_layers: int, **kwargs):
        return EMLP(in_type=self.obs_state_type,
                    out_type=self.state_type,
                    num_layers=num_layers,
                    **kwargs)

    def build_obs_dyn_module(self) -> MarkovDynamics:
        return EquivLinearDynamics(state_type=self.obs_state_type,
                                   dt=self.dt,
                                   trainable=True,
                                   bias=self.enforce_constant_fn)

    def get_obs_space_metrics(self,
                              obs_state_traj: Tensor,
                              obs_state_traj_prime: Optional[Tensor] = None) -> dict:
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
            time_horizon = obs_state_traj_iso.shape[1]
            iso_metrics = obs_state_space_metrics(obs_state_traj=obs_state_traj_iso,
                                                  obs_state_traj_aux=obs_state_traj_prime_iso,
                                                  representation=rep if self.group_avg_trick else None,
                                                  max_ck_window_length=time_horizon - 1)

            iso_spaces_metrics[irrep_id] = iso_metrics

        # Now use the metrics of each Isotypic observable subspace to compute the loss and metrics of the entire
        # observable space.
        obs_space_metrics = iso_metrics_2_obs_space_metrics(iso_spaces_metrics=iso_spaces_metrics,
                                                            obs_iso_reps=self.obs_iso_reps)

        return obs_space_metrics


if __name__ == "__main__":
    G = escnn.group.DihedralGroup(3)
    rep_state = G.regular_representation
    obs_state_dimension = rep_state.size * 3
    gspace = escnn.gspaces.no_base_space(G)

    state_type = FieldType(gspace, [rep_state])

    dt = 0.1
    num_encoder_layers = 4
    num_encoder_hidden_neurons = obs_state_dimension * 2
    activation = torch.nn.Tanh
    equivariant = True

    model = EquivDAE(state_type=state_type,
                     obs_state_dimension=obs_state_dimension,
                     num_encoder_layers=num_encoder_layers,
                     num_encoder_hidden_neurons=num_encoder_hidden_neurons,
                     activation=escnn.nn.ReLU,
                     eigval_network=False,
                     equivariant=equivariant,
                     dt=dt)
    model.eval()
    g = G.sample()

    s = model.state_type(torch.randn(256, rep_state.size))
    g_s = model.state_type.transform_fibers(s.tensor, g)
    g_s = model.state_type(g_s)

    n_steps = 2
    s_pred = model(s.tensor, n_steps=n_steps)
    g_s_pred = model(g_s.tensor, n_steps=n_steps)

    for measurement in ['next_state']:
        for i in range(n_steps):
            s_next = s_pred[measurement][i]
            g_s_next = g_s_pred[measurement][i]

            for s, g_s in zip(s_next, g_s_next):
                rep_g = torch.Tensor(model.state_type.representation(g), device=s.device)
                g_s_next_true = rep_g @ s
                g_s_next_pred = g_s
                absolute_error = torch.norm(g_s_next_true - g_s_next_pred) / torch.norm(g_s_next_true)
                assert torch.allclose(absolute_error, torch.zeros(1), rtol=1e-5,
                                      atol=1e-5), f"t={i} - {absolute_error * 100:.2f}% error"

    print(f"Done here it is your Equivariant Dynamics Autoencoder :)")
