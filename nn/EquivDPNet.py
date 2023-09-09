import logging
import math
from typing import Optional, Tuple, Union

import escnn
import numpy as np
import torch
from escnn.group import Representation
from escnn.nn import FieldType, GeometricTensor
from escnn.nn.modules.basismanager import BlocksBasisExpansion
from torch.utils.data import DataLoader

from nn.DPNet import DPNet, DmdSolver
from nn.mlp import EMLP
from utils.losses_and_metrics import forecasting_loss_and_metrics, obs_state_space_metrics
from utils.representation_theory import isotypic_basis

log = logging.getLogger(__name__)

class EquivDPNet(DPNet):

    def __init__(self,
                 state_type: FieldType,
                 obs_state_dim: int,
                 num_encoder_layers: int = 4,
                 num_encoder_hidden_neurons: int = 128,
                 max_ck_window_length: int = 6,
                 ck_w: float = 0.0,
                 orthonormal_w: float = 0.1,
                 dmd_algorithm: Optional[DmdSolver] = None,
                 activation: escnn.nn.EquivariantModule = escnn.nn.ReLU,
                 batch_norm: bool = True,
                 bias: bool = True):

        # Default dmd algorithm naively exploiting symmetries
        dmd_algorithm = dmd_algorithm if dmd_algorithm is not None else self._full_rank_lstsq_symmetric
        super().__init__(state_dim=state_type.size,
                         obs_state_dim=obs_state_dim,
                         num_encoder_layers=num_encoder_layers,
                         num_encoder_hidden_neurons=num_encoder_hidden_neurons,
                         max_ck_window_length=max_ck_window_length,
                         ck_w=ck_w,
                         orthonormal_w=orthonormal_w,
                         dmd_algorithm=dmd_algorithm,
                         activation=activation,
                         batch_norm=batch_norm,
                         bias=bias)

        self.state_type = state_type
        self.in_type = state_type  # Keep the structure of ESCNN
        gspace = state_type.gspace
        self.symm_group = gspace.fibergroup

        # Number of regular fields in obs state and hidden layers of observable network
        multiplicity = math.ceil(obs_state_dim / state_type.size)
        if multiplicity < 1:
            raise ValueError(f"State-dim:{state_type.size}, |G|={self.symm_group.order()}, "
                             f"obs_dim:{obs_state_dim}")

        # Find the Isotypic basis of the state space and define the observation space representation as
        # `num_spect_field` copies of state representation (in isotypic basis).
        self.obs_space_iso_reps = isotypic_basis(representation=self.state_type.representation,
                                                 multiplicity=multiplicity,
                                                 prefix='ObsSpace')
        # Define the observation space representation in the isotypic basis.
        # Each Field for ESCNN will be an Isotypic Subspace.
        self.obs_state_type = FieldType(gspace, [rep_iso for rep_iso in self.obs_space_iso_reps.values()])
        # Auxiliary variables determining the start and end dimensions of Isotypic subspaces of observable space.
        self.iso_space_dims = {irrep_id: range(s, e) for s, e, irrep_id in zip(self.obs_state_type.fields_start,
                                                                               self.obs_state_type.fields_end,
                                                                               self.obs_space_iso_reps.keys())}

        self.out_type = self.obs_state_type  # Keep the structure of ESCNN

        # When approximating the transfer/Koopman operator from the symmetric observable space, we know the operator
        # belongs to the space of G-equivariant operators (Group Endomorphism of the observable space).
        # Using ESCNN we can compute the basis of the Endomorphism space, and use this basis to compute an empirical
        # G-equivariant approximation of the transfer operator. Since the observable space is defined in the Isotypic
        # basis, the operator is block-diagonal and the basis of the Endomorphism space is the block diagonal sum of
        # the
        # basis of the Endomorphism space of each Isotypic subspace.
        self.iso_space_basis = {}
        self.iso_space_basis_mask = {}  # Used to mask empirical covariance between orthogonal observables
        for irrep_id, rep in self.obs_space_iso_reps.items():
            iso_basis = BlocksBasisExpansion(in_reprs=[self.symm_group.irrep(*id) for id in rep.irreps],
                                             out_reprs=[self.symm_group.irrep(*id) for id in rep.irreps],
                                             basis_generator=gspace.build_fiber_intertwiner_basis,
                                             points=np.zeros((1, 1)))
            self.iso_space_basis[irrep_id] = iso_basis
            basis_coefficients = torch.rand((iso_basis.dimension(),)) + 2
            non_zero_elements = iso_basis(basis_coefficients)[:, :, 0]
            mask = torch.logical_not(torch.isclose(non_zero_elements, torch.zeros_like(non_zero_elements), atol=1e-6))
            self.iso_space_basis_mask[irrep_id] = mask

        # Define the observable network producing the observation state
        num_hidden_regular_fields = int(np.ceil(num_encoder_hidden_neurons / self.symm_group.order()))
        regular_rep = self.symm_group.regular_representation
        intermediate_type = FieldType(gspace, [regular_rep] * num_hidden_regular_fields)
        self.projection_backbone = EMLP(in_type=self.state_type,
                                        out_type=intermediate_type,
                                        num_hidden_units=num_encoder_hidden_neurons,
                                        num_layers=num_encoder_layers - 2,
                                        head_with_activation=True,
                                        batch_norm=batch_norm,
                                        activation=activation,
                                        with_bias=bias)

        self.obs_state_fn = EMLP(in_type=intermediate_type,
                                 out_type=self.obs_state_type,
                                 num_hidden_units=num_encoder_hidden_neurons,
                                 num_layers=2,
                                 batch_norm=batch_norm,
                                 activation=activation,
                                 with_bias=bias)

        self.obs_state_fn_prime = EMLP(in_type=intermediate_type,
                                       out_type=self.obs_state_type,
                                       num_hidden_units=num_encoder_hidden_neurons,
                                       num_layers=2,
                                       batch_norm=batch_norm,
                                       activation=activation,
                                       with_bias=bias)

        # Private variables
        self._batch_dim = None  # Used to convert back and forward between Tensor and GeometricTensor.

        self.iso_transfer_op = {irrep_id: None for irrep_id in self.obs_space_iso_reps.keys()}
        self.transfer_op = None

        num_params = sum([param.nelement() for param in self.parameters()])
        num_train_params = sum([param.nelement() for param in self.parameters() if param.requires_grad])
        log.info(f"Equiv-DPnet Num. Parameters: {num_params} ({num_train_params} trainable)\n"
                 f"\tObservation Space: \n\t\tdim={self.obs_state_type.size} "
                 f"\n\t\tfields={self.obs_state_type.representations} "
                 f"\n\t\tirreps={self.obs_state_type.representation.irreps}"
                 f"\n\tState Space: \n\t\tdim={self.state_type.size}"
                 f"\n\t\tfields={self.state_type.representations} "
                 f"\n\t\tirreps={self.state_type.representation.irreps}")

    # def project(self, state_trajectory: GeometricTensor, **kwargs) -> [dict[str, GeometricTensor]]:
    #
    #     obs_backbone = self.projection_backbone(state_trajectory)
    #     obs_state_traj = self.obs_state_fn(obs_backbone)
    #     obs_state_traj_prime = self.obs_state_fn_prime(obs_backbone)
    #
    #     return dict(obs_state_traj=obs_state_traj,
    #                 obs_state_traj_prime=obs_state_traj_prime)

    def pre_process_state(self,
                          state: torch.Tensor,
                          next_state=torch.Tensor, **kwargs) -> dict:

        preprocessed = super().pre_process_state(state, next_state, **kwargs)
        # Convert to Geometric Tensor
        preprocessed['state_trajectory'] = self.state_type(preprocessed['state_trajectory'])
        return preprocessed

    def post_process_pred(self,
                          obs_state_traj: GeometricTensor,
                          obs_state_traj_prime: GeometricTensor) -> dict[str, torch.Tensor]:
        return super().post_process_pred(obs_state_traj=obs_state_traj.tensor,
                                         obs_state_traj_prime=obs_state_traj_prime.tensor)

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
        pred_next_obs_state = []
        for irrep_id, iso_rep in self.obs_space_iso_reps.items():
            # Get the projection of the observable state in the isotypic subspace
            obs_state_iso = obs_state[..., self.iso_space_dims[irrep_id]]

            # Use the empirical transfer operator to compute the maximum likelihood prediction of the trajectory
            pred_iso_state_traj = [obs_state_iso]
            # The transfer operator of this Isotypic subspace
            iso_transfer_op = self.iso_transfer_op[irrep_id]
            for step in range(n_steps):
                # Compute the next state prediction s_t+1 = K @ s_t
                pred_iso_state_traj.append(pred_iso_state_traj[-1] @ iso_transfer_op)

            pred_next_state_iso = torch.stack(pred_iso_state_traj[1:], dim=1)
            pred_next_obs_state.append(pred_next_state_iso)

        # Concatenate the predictions of each isotypic subspace into the prediction of the entire observable space.
        pred_next_obs_state = torch.cat(pred_next_obs_state, dim=-1)
        return dict(pred_next_obs_state=pred_next_obs_state)

    def compute_loss_and_metrics(self,
                                 obs_state: torch.Tensor,
                                 next_obs_state: torch.Tensor,
                                 pred_next_obs_state: Optional[torch.Tensor] = None,
                                 **kwargs
                                 ) -> (torch.Tensor, dict[str, torch.Tensor]):
        # Decompose batched observations in their isotypic spaces.
        pred_horizon = next_obs_state.shape[1]

        # For each Isotypic Subspace, compute empirical Covariance and Cross-Covariance operators.
        # With these, compute spectral, projection scores and orthonormality and Chapman-Kolmogorov regularization.
        iso_spaces_metrics = {irrep_id: {} for irrep_id in self.obs_space_iso_reps.keys()}
        for irrep_id, iso_rep in self.obs_space_iso_reps.items():
            rep = iso_rep if irrep_id != self.symm_group.trivial_representation else None  # Check for Trivial IsoSpace
            # Get the projection of the observable state in the isotypic subspace
            obs_state_iso = obs_state[..., self.iso_space_dims[irrep_id]]
            next_obs_state_iso = next_obs_state[..., self.iso_space_dims[irrep_id]]

            # Compute Covariance and Cross-Covariance operators for this Isotypic subspace.
            # Spectral and Projection scores, and CK loss terms.
            iso_metrics = obs_state_space_metrics(obs_state=obs_state_iso,
                                                  next_obs_state=next_obs_state_iso,
                                                  representation=rep,
                                                  max_ck_window_length=self.max_ck_window_length,
                                                  ck_w=self.ck_w)

            iso_spaces_metrics[irrep_id] = iso_metrics

        # Now use the metrics of each Isotypic observable subspace to compute the loss and metrics of the entire
        # observable space.
        obs_space_metrics = self.obs_space_metrics(iso_spaces_metrics=iso_spaces_metrics,
                                                   time_horizon=pred_horizon + 1)

        loss = self.compute_loss(spectral_score=obs_space_metrics['spectral_score'],
                                 ck_score=obs_space_metrics['ck_score'],
                                 orth_reg=obs_space_metrics['orth_reg'])

        # Include prediction metrics if available
        if pred_next_obs_state is not None:
            assert pred_next_obs_state.shape == next_obs_state.shape
            pred_loss, pred_metrics = forecasting_loss_and_metrics(state_gt=next_obs_state,
                                                                   state_pred=pred_next_obs_state)
            pred_metrics['pred_loss'] = pred_loss
            obs_space_metrics.update(pred_metrics)

        return loss, obs_space_metrics

    def obs_space_metrics(self, iso_spaces_metrics: dict, time_horizon: int) -> dict:
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
            TODO:
        """

        # Compute the entire obs space Orthonormal regularization terms for all time horizon.
        # orth_reg[t] = ||Cov(t, t) - I_obs ||_Fro = sqrt(sum_iso(||Cov_iso(t, t) - I_iso ||^2_Fro))
        iso_orth_reg = torch.vstack(
            [iso_spaces_metrics[irrep_id]['orth_reg'] for irrep_id in self.obs_space_iso_reps.keys()])
        obs_space_orth_reg = torch.sqrt(torch.sum(iso_orth_reg ** 2, dim=0))

        # Compute the entire obs space CK regularization terms.
        # ck_reg[t, t+d] = ||Cov(t, t+d) - (Cov(t, t+1)...Cov(t+d-1, t+d))||_Fro
        #                = sqrt(sum_iso(||Cov_iso(t, t+d) - (Cov_iso(t, t+1)...Cov_iso(t+d-1, t+d))||^2_Fro))
        iso_ck_reg = torch.stack(
            [iso_spaces_metrics[irrep_id]['ck_reg'] for irrep_id in self.obs_space_iso_reps.keys()], dim=0)
        obs_space_ck_reg = torch.sqrt(torch.sum(iso_ck_reg ** 2, dim=0))

        # Compute the Correlation/Projection score
        # P_score[t, t+d] = ||Cov(t)^-1/2 Cov[t, t+d] Cov(t+d)^-1/2||_HS
        #                 = sqrt(sum_iso(||Cov_iso(t)^-1/2 Cov_iso(t, t+d) Cov_iso(t+d)^-1/2||_HS^2))
        #                 = sqrt(sum_iso(P_score_iso[t, t+d]^2))
        iso_P_score = torch.stack(
            [iso_spaces_metrics[irrep_id]['projection_scores'] for irrep_id in self.obs_space_iso_reps.keys()], dim=0)
        obs_space_P_score = torch.sqrt(torch.sum(iso_P_score ** 2, dim=0))

        # Compute the Spectral scores for the entire obs-space
        # S_score[t, t+d] = ||Cov(t, t+d)||_HS / (||Cov(t)||_HS ||Cov(t+d)||_HS)
        #                 <= ||Cov(t)^-1/2 Cov(t,t+d) Cov(t+d)^-1/2||_HS
        #                 <= sqrt(sum_iso((||Cov_iso(X,Y)||_HS / (||Cov_iso(X)||_HS ||Cov_iso(Y)||_HS)^2)))
        #                 <= sqrt(sum_iso(S_score_iso[t, t+d]^2)))
        iso_S_score = torch.stack(
            [iso_spaces_metrics[irrep_id]['spectral_scores'] for irrep_id in self.obs_space_iso_reps.keys()], dim=0)
        obs_space_S_score = torch.sqrt(torch.sum(iso_S_score ** 2, dim=0))

        # With the Spectral, Projection scores and the Orthogonal and CK regularization terms of the entire
        # observation space, we proceed to compute the regularized Spectral and CK scores:
        # CK_score_reg[t, t+d] = (Σ_i=0^d-1 S_score[t+i, t+i+1]) - ck_w * (ck_reg[t, t+d])
        min_steps = 2
        device, dtype = obs_space_orth_reg.device, obs_space_orth_reg.dtype
        # s_scores_reg = torch.fill(torch.zeros((time_horizon, time_horizon), dtype=dtype, device=device), torch.nan)
        ck_scores_reg = torch.fill(torch.zeros((time_horizon, time_horizon), dtype=dtype, device=device), torch.nan)
        for t in range(0, time_horizon - min_steps):  # t ∈ [# 0, time_horizon - 2]
            max_dt = min(time_horizon - t, self.max_ck_window_length + 1)
            for dt in range(min_steps, max_dt):
                avg_s_scores = torch.mean(obs_space_S_score[t, t + 1: t + dt])  # Σ_i=0^d-1 S_score[t+i, t+i+1]
                ck_scores_reg[t, t + dt] = avg_s_scores - self.ck_w * obs_space_ck_reg[t, t + dt]
                assert not torch.isnan(ck_scores_reg[t, t + dt]), f"Something is fishy here"

        non_nans = lambda x: torch.logical_not(torch.isnan(x))

        return dict(orth_reg=torch.mean(obs_space_orth_reg),
                    ck_score=torch.mean(ck_scores_reg[non_nans(ck_scores_reg)]),
                    ck_reg=torch.mean(obs_space_ck_reg[non_nans(obs_space_ck_reg)]),
                    spectral_score=torch.mean(obs_space_S_score[non_nans(obs_space_S_score)]),
                    projection_score=torch.mean(obs_space_P_score[non_nans(obs_space_P_score)])
                    )

    @torch.no_grad()
    def approximate_transfer_operator(self, train_data_loader: DataLoader):
        train_data = {}
        for batch in train_data_loader:
            for key, value in batch.items():
                if key not in train_data:
                    train_data[key] = torch.squeeze(value)
                else:
                    torch.cat([train_data[key], torch.squeeze(value)], dim=0)

        # Perform data augmentation of the entire dataset.
        # TODO: We should avoid this by simply solving the least square problem, considering equivariance.
        for key, value in train_data.items():
            orbit = [value]
            for g in self.symm_group.elements:
                gv = self.state_type.transform_fibers(value, g)
                orbit.append(gv)
            train_data[key] = torch.cat(orbit, dim=0)

        pred = self(**train_data, n_steps=0)
        obs_state = pred['obs_state']
        next_obs_state = torch.squeeze(pred['next_obs_state'])

        # For each Isotypic Subspace, compute the empirical transfer operator with the present observable state
        # space.
        for irrep_id, iso_rep in self.obs_space_iso_reps.items():
            rep = iso_rep if irrep_id != self.symm_group.trivial_representation else None  # Check for Trivial
            # IsoSpace
            # Get the projection of the observable state in the isotypic subspace
            obs_state_iso = obs_state[..., self.iso_space_dims[irrep_id]]
            next_obs_state_iso = next_obs_state[..., self.iso_space_dims[irrep_id]]

            # Generate the data matrices of x(w_t) and x(w_t+1)
            X = obs_state_iso  # (batch_dim, obs_state_dim)
            X_prime = next_obs_state_iso  # (batch_dim, obs_state_dim)

            # Compute the empirical transfer operator of this Isotypic subspace
            A_iso = self.dmd_algorithm(X, X_prime, representation=rep)
            self.iso_transfer_op[irrep_id] = A_iso

        op = torch.block_diag(*self.iso_transfer_op.values())
        self.transfer_op = op

    @staticmethod
    def _full_rank_lstsq_symmetric(
            X: torch.Tensor, X_prime: torch.Tensor, rep: Representation) -> torch.Tensor:
        """ Compute the least squares solution of the linear system X_prime = A·X.

        If the representation is provided the empirical transfer operator is improved using the group average trick to
        enforce equivariance rep(g) A = A rep(g) | forall g in G.

        Args:
            X: (state_dim, n_samples) Data matrix of the initial states.
            X_prime: (state_dim, n_samples) Data matrix of the next states.
            rep (Representation): Group Representation on the state space.
        Returns:
            A: (state_dim, state_dim) Least squares solution of the linear system `X' = A·X`.
        """
        A = DPNet._full_rank_lstsq(X, X_prime)
        # Do the group average trick to enforce equivariance.
        # This is equivalent to applying the group average trick on the singular vectors of the covariance matrices.
        A_G = []
        for g in rep.group.elements:
            A_g = rep(g) @ A @ rep(~g)
            A_G.append(A_g)
        A_G = torch.stack(A_G, dim=0)
        A_G = torch.mean(A_G, dim=0)
        return A_G

    def get_metric_labels(self) -> list[str]:
        return ['pred_loss', 'S_score', 'ck_score', 'P_score', 'reg_orthonormal']

    def get_hparams(self):
        hparams = super().get_hparams()
        hparams.update(group=self.symm_group.name, num_iso_spaces=len(self.obs_space_iso_reps))
        return hparams

    def evaluate_output_shape(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        batch_dim, state_dim = input_shape
        return batch_dim, self.obs_state_type.size
