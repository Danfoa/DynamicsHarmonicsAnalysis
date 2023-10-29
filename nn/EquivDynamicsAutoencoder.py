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
from utils.losses_and_metrics import obs_state_space_metrics
from utils.mysc import batched_to_flat_trajectory
from utils.representation_theory import isotypic_basis

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

        # Find the Isotypic basis of the state space and define the observation space representation as
        # `num_spect_field` copies of state representation (in isotypic basis).
        self.state_iso_reps, self.state_iso_dims, Q_iso2state = isotypic_basis(representation=state_rep,
                                                                               multiplicity=1,
                                                                               prefix='State')
        # Store the change of basis from original input basis to the isotypic basis of the space.
        if np.allclose(Q_iso2state, np.eye(state_rep.size)):
            Q_iso2state, Q_state2iso = None, None
        else:
            Q_iso2state = torch.Tensor(Q_iso2state)
            Q_state2iso = torch.Tensor(np.linalg.inv(Q_iso2state))

        # Define the observation space representation in the isotypic basis.
        self.obs_iso_reps, self.obs_iso_dims, _ = isotypic_basis(representation=state_rep,
                                                                 multiplicity=multiplicity,
                                                                 prefix='Obs')
        # Each Field for ESCNN will be an Isotypic Subspace.
        self.state_type = FieldType(self.gspace, [state_rep])
        # Field type on isotypic basis.
        self.state_type_iso = FieldType(self.gspace, [rep_iso for rep_iso in self.state_iso_reps.values()])
        self.obs_state_type = FieldType(self.gspace, [rep_iso for rep_iso in self.obs_iso_reps.values()])

        # Define a dict containing the transfer operator of each Isotypic subspace.
        self.iso_transfer_op = {irrep_id: None for irrep_id in self.obs_iso_reps.keys()}
        self.iso_inverse_projector = {irrep_id: None for irrep_id in self.obs_iso_reps.keys()}

        super(EquivDAE, self).__init__(state_dim=self.state_type.size,
                                       obs_state_dim=obs_state_dim,
                                       dt=dt,
                                       obs_fn_params=_obs_fn_params,
                                       obs_state_rep=self.obs_state_type.representation,
                                       state_change_of_basis=Q_state2iso,
                                       state_inv_change_of_basis=Q_iso2state,
                                       **dae_kwargs)

    def pre_process_state(self, state: Tensor, next_state: Optional[Tensor] = None) -> GeometricTensor:
        # Change basis to Isotypic basis.
        state_trajectory_iso_basis = super().pre_process_state(state=state, next_state=next_state)
        # Convert to Geometric Tensor
        return self.state_type_iso(state_trajectory_iso_basis)

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
        state_traj_input_basis = super().post_process_state(state_traj=state_traj.tensor)
        return state_traj_input_basis

    def build_obs_fn(self, num_layers: int, **kwargs):
        return EMLP(in_type=self.state_type_iso,
                    out_type=self.obs_state_type,
                    num_layers=num_layers,
                    **kwargs)

    def build_inv_obs_fn(self, num_layers: int, **kwargs):
        return EMLP(in_type=self.obs_state_type,
                    out_type=self.state_type_iso,
                    num_layers=num_layers,
                    **kwargs)

    def build_obs_dyn_module(self) -> MarkovDynamics:
        return EquivLinearDynamics(state_type=self.obs_state_type,
                                   dt=self.dt,
                                   trainable=True,
                                   group_avg_trick=self.group_avg_trick,
                                   bias=self.enforce_constant_fn)

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
            iso_metrics = obs_state_space_metrics(obs_state_traj=obs_state_traj_iso,
                                                  obs_state_traj_aux=obs_state_traj_prime_iso,
                                                  representation=rep if self.group_avg_trick else None,
                                                  max_ck_window_length=self.max_ck_window_length)

            iso_spaces_metrics[irrep_id] = iso_metrics

        # Now use the metrics of each Isotypic observable subspace to compute the loss and metrics of the entire
        # observable space.
        obs_space_metrics = self.iso_metrics_2_obs_space_metrics(iso_spaces_metrics=iso_spaces_metrics)

        return obs_space_metrics

    def iso_metrics_2_obs_space_metrics(self, iso_spaces_metrics: dict) -> dict:
        """ Compute the observable space metrics from the isotypic subspace metrics.

        This function exploits the fact that the Hilbert-Schmidt (HS) norm of an operator (or the Frobenious norm
        of a matrix) that is block-diagonal is defined as the square root of the sum of the squared norms of the blocks:
         ||A||_HS = sqrt(||A_o||_HS^2 + ... + ||A_i||_HS^2)  | A := block_diag(A_o, ..., A_i).
        Thus, we have that the projection score defined as:
         P_score = ||CovX^-1/2 CovXY CovY^-1/2||_HS, can be decomposed into the projection score of the Iso spaces
                 = sqrt(sum_iso(||Cov_iso(X)^-1/2 Cov_iso(XY) Cov_iso(Y)^-1/2||_HS))
        Likewise for the orthogonal and the Chapman-Kolmogorov (Markovian) regularization terms.
        Args:
            iso_spaces_metrics:
        Returns:
            Dictionary containing:
            - spectral_score: (time_horizon - 1) Tensor containing the average spectral score between time steps
            separated
             apart by a shift of `dt` [steps/time]. That is:
                spectral_score[dt - 1] = avg(||Cov(x_i, x'_i+dt)||_HS^2/(||Cov(x_i, x_i)||_2*||Cov(x'_i+dt,
                x'_i+dt)||_2))
                 | ∀ i in [0, time_horizon - dt], dt in [1, min(time_horizon - i, window_size)]
            - corr_score: (time_horizon - 1) Tensor containing the correlation scores between time steps separated
             apart by a shift of `dt` [steps/time]. That is:
                corr_score[dt - 1] = avg(||Cov(x_i, x_i)^-1 Cov(x_i, x'_i+dt) Cov(x'_i+dt, x'_i+dt)^-1||_HS^2)
                 | ∀ i in [0, time_horizon - dt], dt in [1, min(time_horizon - i, window_size)]
            - orth_reg: (time_horizon) Tensor containing the orthonormality regularization term for each time step.
             That is: orth_reg[t] = || Cov(t,t) - I ||_2
            - ck_reg: (time_horizon - 1,) Average CK error per `dt` time steps. That is:
                ck_error[dt - 2] = avg(|| Cov(t, t+dt) - Cov(t, t+1) Cov(t+1, t+2) ... Cov(t+dt-1, t+dt) ||) |
                ∀ t in [0, time_horizon - 2], dt in [2, min(time_horizon - 2, ck_window_length)]
            TODO:
                - cov_cond_num: (float) Average condition number of the Covariance matrix of the entire observation
                space.
        """

        # Compute the entire obs space Orthonormal regularization terms for all time horizon.
        # orth_reg[t] = ||Cov(t, t) - I_obs ||_Fro = sqrt(sum_iso(||Cov_iso(t, t) - I_iso ||^2_Fro))
        iso_orth_reg = torch.vstack(
            [iso_spaces_metrics[irrep_id]['orth_reg'] for irrep_id in self.obs_iso_reps.keys()])
        obs_space_orth_reg = torch.sqrt(torch.sum(iso_orth_reg ** 2, dim=0))

        # Compute the entire obs space CK regularization terms.
        # ck_reg[t, t+d] = ||Cov(t, t+d) - (Cov(t, t+1)...Cov(t+d-1, t+d))||_Fro
        #                = sqrt(sum_iso(||Cov_iso(t, t+d) - (Cov_iso(t, t+1)...Cov_iso(t+d-1, t+d))||^2_Fro))
        iso_ck_reg = torch.stack(
            [iso_spaces_metrics[irrep_id]['ck_reg'] for irrep_id in self.obs_iso_reps.keys()], dim=0)
        obs_space_ck_reg = torch.sqrt(torch.sum(iso_ck_reg ** 2, dim=0))

        # Compute the Correlation/Projection score
        # P_score[t, t+d] = ||Cov(t)^-1/2 Cov[t, t+d] Cov(t+d)^-1/2||_HS^2
        #                 = sum_iso ||Cov_iso(t)^-1/2 Cov_iso(t, t+d) Cov_iso(t+d)^-1/2||_HS^2)
        #                 = sum_iso (P_score_iso[t, t+d])
        iso_corr_score = torch.stack(
            [iso_spaces_metrics[irrep_id]['corr_score'] for irrep_id in self.obs_iso_reps.keys()], dim=0)
        obs_space_corr_score = torch.sum(iso_corr_score, dim=0)

        # Compute the Spectral scores for the entire obs-space
        # S_score[t, t+d] = ||Cov(t, t+d)||_HS / (||Cov(t)||_HS ||Cov(t+d)||_HS)
        #                 <= ||Cov(t)^-1/2 Cov(t,t+d) Cov(t+d)^-1/2||_HS
        #                 <= sum_iso((||Cov_iso(X,Y)||_HS / (||Cov_iso(X)||_HS ||Cov_iso(Y)||_HS)^2))
        #                 <= sum_iso(S_score_iso[t, t+d])
        iso_S_score = torch.stack(
            [iso_spaces_metrics[irrep_id]['spectral_score'] for irrep_id in self.obs_iso_reps.keys()], dim=0)
        obs_space_S_score = torch.sum(iso_S_score, dim=0)

        return dict(orth_reg=obs_space_orth_reg,
                    ck_reg=obs_space_ck_reg,
                    spectral_score=obs_space_S_score,
                    corr_score=obs_space_corr_score,
                    # corr_score_t=torch.nanmean(corr_score_t, dim=0, keepdim=True),  # (batch, time)
                    # spectral_score_t=torch.nanmean(spectral_score_t, dim=0, keepdim=True)  # (batch, time)
                    )

    @staticmethod
    def compute_invariant_features(x: torch.Tensor, field_type: FieldType) -> torch.Tensor:
        n_inv_features = len(field_type.irreps)
        # TODO: Ensure isotypic basis i.e irreps of the same type are consecutive to each other.
        inv_features = []
        for field_start, field_end, rep in zip(field_type.fields_start, field_type.fields_end,
                                               field_type.representations):
            # Each field here represents a representation of an Isotypic Subspace. This rep is only composed of a single
            # irrep type.
            x_field = x[..., field_start:field_end]
            num_G_stable_spaces = len(rep.irreps)  # Number of G-invariant features = multiplicity of irrep
            # Again this assumes we are already in an Isotypic basis
            assert len(np.unique(rep.irreps, axis=0)) == 1, "This only works for now on the Isotypic Basis"
            # This basis is useful because we can apply the norm in a vectorized way
            # Reshape features to [batch, num_G_stable_spaces, num_features_per_G_stable_space]
            x_field_p = torch.reshape(x_field, (x_field.shape[0], num_G_stable_spaces, -1))
            # Compute G-invariant measures as the norm of the features in each G-stable space
            inv_field_features = torch.norm(x_field_p, dim=-1)
            # Append to the list of inv features
            inv_features.append(inv_field_features)
        # Concatenate all the invariant features
        inv_features = torch.cat(inv_features, dim=-1)
        assert inv_features.shape[-1] == n_inv_features, f"Expected {n_inv_features} got {inv_features.shape[-1]}"
        return inv_features
        # if self.eigval_net:
        #     # The eigenvalue network needs to be a G-invariant network. For obtaining G-invariant features for this
        #     # network
        #     # We can exploit the fact that in the isotypic basis every irreducible G-stable space can be used to
        #     # obtain a
        #     # G-invariant feature: The norm of the state in that irreducible G-stable space is G-invariant.
        #     num_G_stable_spaces = sum([len(rep.irreps) for rep in obs_space_iso_reps.values()])
        #     num_degrees_of_freedom = self.obs_state_dynamics.equiv_lin_map.basisexpansion.dimension()
        #     self.eigval_net = MLP(in_dim=num_G_stable_spaces,
        #                           out_dim=num_degrees_of_freedom,
        #                           num_hidden_units=num_encoder_hidden_neurons,
        #                           num_layers=3,
        #                           activation=torch.nn.ReLU,
        #                           bias=True)
        #     raise NotImplementedError("TODO: Need to implement this. "
        #                               "There is no easy way to get batched parametrization of equivariant maps")


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
