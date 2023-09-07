import cProfile
import logging
import math
import pstats
import time
from functools import reduce
from pathlib import Path
from typing import Optional, Tuple, Union

import escnn
import numpy as np
import torch
from escnn.nn import FieldType, GeometricTensor
from escnn.nn.modules.basismanager import BlocksBasisExpansion
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.DynamicsDataModule import DynamicsDataModule
from utils.representation_theory import isotypic_basis
from nn.mlp import EMLP, MLP
from utils.losses_and_metrics import (chapman_kolmogorov_regularization, compute_chain_projection_scores,
                                      compute_chain_spectral_scores, empirical_cov_cross_cov,
                                      forecasting_loss_and_metrics, obs_state_space_metrics,
                                      regularization_orthonormality)
from utils.mysc import append_dictionaries

log = logging.getLogger(__name__)


class EquivDeepProjectionNet(escnn.nn.EquivariantModule):
    TIME_DIM = 1

    def __init__(self,
                 state_type: FieldType,
                 obs_state_dimension: int,
                 num_encoder_layers: int = 4,
                 num_encoder_hidden_neurons: int = 128,
                 max_ck_window_length: int = 6,
                 ck_w: float = 0.0,
                 orthonormal_w: float = 0.1,
                 activation: escnn.nn.EquivariantModule = escnn.nn.ReLU,
                 batch_norm: bool =True,
                 bias: bool = True):
        super().__init__()

        self.state_type = state_type
        self.in_type = state_type  # Keep the structure of ESCNN
        assert max_ck_window_length >= 2, "Minimum window_size of Chapman-Kolmogorov regularization is 2 steps"
        self.max_ck_window_length = max_ck_window_length
        gspace = state_type.gspace
        self.symm_group = gspace.fibergroup
        self.ck_w = ck_w
        self.orthonormal_w = orthonormal_w

        # Number of regular fields in obs state and hidden layers of observable network
        multiplicity = math.ceil(obs_state_dimension / state_type.size)
        if multiplicity < 1:
            raise ValueError(f"State-dim:{state_type.size}, |G|={self.symm_group.order()}, "
                             f"obs_dim:{obs_state_dimension}")

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

    def project(self, state_trajectory: GeometricTensor, predict=True, **kwargs) -> [dict[str, GeometricTensor]]:

        obs_backbone = self.projection_backbone(state_trajectory)
        obs_state_traj = self.obs_state_fn(obs_backbone)
        obs_state_traj_prime = self.obs_state_fn_prime(obs_backbone)

        return dict(obs_state_traj=obs_state_traj,
                    obs_state_traj_prime=obs_state_traj_prime)

    def pre_process_state(self,
                          state: torch.Tensor,
                          next_state=torch.Tensor, **kwargs) -> Union[GeometricTensor, dict[str, GeometricTensor]]:
        self._batch_dim = state.shape[0]

        if next_state.shape == state.shape:  # next_state : (batch_dim, state_dim)
            state_trajectory = torch.cat([state, next_state], dim=0)
        else:                                # next_state : (batch_dim, pred_horizon, state_dim)
            state_trajectory = torch.cat([torch.unsqueeze(state, dim=1), next_state], dim=1)
        # Combine initial state and next states into a state trajectory.
        state_trajectory = state_trajectory.reshape(-1, self.state_type.size)
        state_trajectory = self.state_type(state_trajectory)

        return dict(state_trajectory=state_trajectory, **kwargs)

    def post_process_pred(self,
                          obs_state_traj: GeometricTensor,
                          obs_state_traj_prime: GeometricTensor) -> dict[str, torch.Tensor]:

        obs_traj = torch.reshape(obs_state_traj.tensor, (self._batch_dim, -1, self.obs_state_type.size))
        obs_state = obs_traj[:, 0, ...]  # First "time step" is the initial state observation
        next_obs_state = obs_traj[:, 1:, ...]  # The Rest is the observations of the next states,

        obs_traj_prime = torch.reshape(obs_state_traj_prime.tensor, (self._batch_dim, -1, self.obs_state_type.size))
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
        # Apply any required pre-processing to the initial state and state trajectory
        input = self.pre_process_state(state, **kwargs)
        # Compute the observations of the trajectory of motion
        projections = self.project(**input)
        # Post-process predictions
        output = self.post_process_pred(**projections)

        if n_steps > 0 and self.transfer_op is not None:
            obs_state = output['obs_state']
            pred_next_obs_state = self.forcast(obs_state, n_steps=n_steps)
            output.update(pred_next_obs_state)

        return output

    def compute_loss_and_metrics(self,
                                 obs_state: torch.Tensor,
                                 next_obs_state: torch.Tensor,
                                 pred_next_obs_state: Optional[torch.Tensor] = None,
                                 **kwargs
                                 ) -> (torch.Tensor, dict[str, torch.Tensor]):

        # Decompose batched observations in their isotypic spaces.
        pred_horizon = next_obs_state.shape[1]

        # log.debug(f"Computing Loss and Metrics for {projections['obs_state'].shape[0]} samples. Device {device}")

        # Computing the cov and cross-covariance operator with the entire obs state dimension and the average trick
        # results in large equivariance error. Might be faster to compute on GPU but not accurate.
        # CovXX_iso, CovXY_iso = self.empirical_cov_cross_cov(obs_state, next_obs_state,
        #                                                     rep=self.obs_state_type.fiber_representation)

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
        spectral_score = obs_space_metrics['spectral_score']
        ck_score = obs_space_metrics['ck_score']
        orthonormal_reg = obs_space_metrics['orth_reg']

        if self.ck_w == 0.0 or np.isclose(self.ck_w, 0.0):
            loss = - (spectral_score - self.orthonormal_w * orthonormal_reg)
        else:
            loss = - (ck_score - self.orthonormal_w * orthonormal_reg)

        assert not torch.isnan(loss), f"Loss is NaN. Metrics: {obs_space_metrics}"

        # Include prediction metrics if available
        if pred_next_obs_state is not None:
            assert pred_next_obs_state.shape == next_obs_state.shape
            pred_loss, pred_metrics = forecasting_loss_and_metrics(state_gt=next_obs_state,
                                                                   state_pred=pred_next_obs_state)
            pred_metrics['pred_loss'] = pred_loss
            obs_space_metrics.update(pred_metrics)
        # log.debug(f"Computing Loss and Metrics took {time.time() - start_time:.2f}[s]")
        return loss, obs_space_metrics

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

    def approximate_transfer_operator(self, data_loader: DataLoader):
        with torch.no_grad():
            train_data = {}
            for batch in data_loader:
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

                X = obs_state_iso  # .detach().cpu().numpy()
                Y = next_obs_state_iso  # .detach().cpu().numpy()
                # Compute the empirical transfer operator for this Isotypic subspace.
                result = torch.linalg.lstsq(X, Y, rcond=None)  # min||A X - B ||_F   |  B = A @ X

                op = result.solution
                self.iso_transfer_op[irrep_id] = op  # y =  x @ op   op:(|y| x |x|)
                # residuals = result.residuals
                # if log.level == logging.DEBUG:
                #     pred = torch.einsum('yx,bx->by', op, obs_state_iso)
                #     emp_residuals = torch.mean(torch.square(pred - next_obs_state_iso), dim=0)
                #     pred2 = torch.einsum('yx,bx->by', op.T, obs_state_iso)
                #     emp_residuals2 = torch.mean(torch.square(pred - next_obs_state_iso), dim=0)
                #     residuals = torch.mean(torch.square(emp_residuals - next_obs_state_iso))

        op = torch.block_diag(*self.iso_transfer_op.values())
        self.transfer_op = op

    def get_metric_labels(self) -> list[str]:
        return ['pred_loss', 'S_score', 'ck_score', 'P_score', 'reg_orthonormal']

    def get_hparams(self):
        return {'encoder':             self.projection_backbone.get_hparams(),
                'num_isotypic_spaces': len(self.obs_space_iso_reps),
                }

    def evaluate_output_shape(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        batch_dim, state_dim = input_shape
        return batch_dim, self.obs_state_type.size


class DeepProjectionNet(torch.nn.Module):

    def __init__(self,
                 state_dim: int,
                 obs_state_dimension: int,
                 num_encoder_layers: int = 4,
                 num_encoder_hidden_neurons: int = 128,
                 max_ck_window_length: int = 6,
                 ck_w: float = 0.1,
                 orthonormal_w: float = 0.1,
                 approx_surjective_obs=False,
                 activation=torch.nn.ReLU,
                 **kwargs):
        super().__init__()

        self.state_dim = state_dim
        self.obs_state_dim = obs_state_dimension
        assert max_ck_window_length >= 2, "Minimum window_size of Chapman-Kolmogorov regularization is 2 steps"
        self.max_ck_window_length = max_ck_window_length
        self.approx_surjective_obs = approx_surjective_obs
        self.ck_w = ck_w
        self.orthonormal_w = orthonormal_w

        # Define the observable network producing the observation state
        self.projection = MLP(in_dim=self.state_dim,
                              out_dim=self.obs_state_dim,
                              num_hidden_units=num_encoder_hidden_neurons,
                              num_layers=num_encoder_layers,
                              activation=activation,
                              with_bias=True)

        if self.approx_surjective_obs:
            self.projection_inv = MLP(in_dim=self.obs_state_dim,
                                      out_dim=self.state_dim,
                                      num_hidden_units=num_encoder_hidden_neurons,
                                      num_layers=num_encoder_layers,
                                      activation=activation,
                                      with_bias=True)

        # Private variables
        self._batch_dim = None  # Used to convert back and forward between Tensor and GeometricTensor.

        self.transfer_op = None

        num_params = sum([param.nelement() for param in self.parameters()])
        num_train_params = sum([param.nelement() for param in self.parameters() if param.requires_grad])
        log.info(f"DPnet Num. Parameters: {num_params} ({num_train_params} trainable)\n")

    def project(self, state_trajectory: torch.Tensor, **kwargs) -> [dict[str, torch.Tensor]]:

        obs_trajectory = self.projection(state_trajectory)

        if self.approx_surjective_obs:
            state_traj = self.projection_inv(obs_trajectory)
            return dict(obs_state_trajectory=obs_trajectory, state_traj=state_traj)

        return dict(obs_state_trajectory=obs_trajectory)

    def pre_process_state(self,
                          state: torch.Tensor,
                          next_state=torch.Tensor, **kwargs) -> Union[torch.Tensor, dict[str, torch.Tensor]]:

        flat_next_state = torch.reshape(next_state, (-1, next_state.shape[-1]))
        state_trajectory = torch.cat([state, flat_next_state], dim=0)

        return dict(state_trajectory=state_trajectory, **kwargs)

    def post_process_pred(self,
                          obs_state_trajectory: torch.Tensor,
                          state_trajectory: Optional[torch.Tensor] = None) -> dict[str, torch.Tensor]:

        state_traj = torch.reshape(obs_state_trajectory, (self._batch_dim, -1, self.obs_state_dim))
        obs_state = state_traj[:, 0, ...]  # First "time step" is the initial state observation
        next_obs_state = state_traj[:, 1:, ...]  # The Rest is the observations of the next states,

        if state_trajectory is not None:
            state_traj = torch.reshape(state_trajectory.tensor, (self._batch_dim, -1, self.state_dim))
            state = state_traj[:self._batch_dim, ...]  # First "batch" is the initial state observation
            next_state = state_traj[self._batch_dim:, ...]  # Rest is the observations of the next states,
            # Return using the convention of the MarkovDynamicsModule
            return dict(state=state, next_state=next_state, obs_state=obs_state, next_obs_state=next_obs_state)

        return dict(obs_state=obs_state, next_obs_state=next_obs_state)

    def forward(self,
                state: torch.Tensor, n_steps: int = 1, **kwargs) -> [dict[str, torch.Tensor]]:
        """ Forward pass of the dynamics model, producing a prediction of the next `n_steps` states.
        Args:
            state: Initial state of the system.
            n_steps: Number of steps to predict.
            **kwargs: Auxiliary arguments

        Returns:
            predictions (dict): A dictionary containing the predicted states under the key 'state' and
            potentially other auxiliary measurements.
        """
        self._batch_dim = state.shape[0]
        # Apply any required pre-processing to the initial state and state trajectory
        input = self.pre_process_state(state, **kwargs)
        # Compute the observations of the trajectory of motion
        predictions = self.project(**input)
        # Post-process predictions
        predictions = self.post_process_pred(**predictions)
        return predictions

    def compute_loss_and_metrics(self,
                                 projections: dict[str, torch.Tensor],
                                 ground_truth: dict[str, torch.Tensor],
                                 predict: bool = False) -> (torch.Tensor, dict[str, torch.Tensor]):

        # Decompose batched observations in their isotypic spaces.
        obs_state = projections['obs_state']
        next_obs_state = projections['next_obs_state']

        pred_horizon = next_obs_state.shape[1]
        # Compute Covariance and Cross-Covariance operators for the observation state space.
        # Spectral and Projection scores, and CK loss terms.
        obs_space_metrics = obs_state_space_metrics(obs_state=obs_state,
                                                    next_obs_state=next_obs_state,
                                                    representation=None,
                                                    max_ck_window_length=self.max_ck_window_length,
                                                    ck_w=self.ck_w, )

        if predict and self.transfer_op is not None:
            with torch.no_grad():  # Fast forcasting
                # Use the empirical transfer operator to compute the maximum likelihood prediction of the trajectory
                pred_state_traj = [obs_state]
                for step in range(pred_horizon):
                    # Compute the next state prediction s_t+1 = K @ s_t
                    pred_state_traj.append(pred_state_traj[-1] @ self.transfer_op)
                    # iso_transfer_op @ state_traj[-1])
                pred_next_state_iso = torch.stack(pred_state_traj[1:], dim=1)
                gt_next_state_iso = next_obs_state

                # Compute the forcasting prediction error.
                pred_loss, pred_metrics = forecasting_loss_and_metrics(state_gt=gt_next_state_iso,
                                                                       state_pred=pred_next_state_iso)
                obs_space_metrics.update(pred_metrics)
                obs_space_metrics['pred_loss'] = pred_loss

        non_nans = lambda x: torch.logical_not(torch.isnan(x))
        # Summarize the metrics into a scalar value
        for metric, values in obs_space_metrics.items():
            obs_space_metrics[metric] = torch.mean(values[non_nans(values)])

        # Compute loss
        spectral_score = obs_space_metrics['spectral_scores']
        ck_score = obs_space_metrics['ck_scores']
        orthonormal_reg = obs_space_metrics['orth_reg']

        if self.ck_w == 0.0 or np.isclose(self.ck_w, 0.0):
            loss = - (spectral_score - self.orthonormal_w * orthonormal_reg)
        else:
            loss = - (ck_score - self.orthonormal_w * orthonormal_reg)

        assert not torch.isnan(loss), f"Loss is NaN"
        # log.debug(f"Computing Loss and Metrics took {time.time() - start_time:.2f}[s]")
        return loss, obs_space_metrics

    def approximate_transfer_operator(self, data_loader: DataLoader):
        with torch.no_grad():
            train_data = {}
            for batch in data_loader:
                for key, value in batch.items():
                    if key not in train_data:
                        train_data[key] = torch.squeeze(value)
                    else:
                        torch.cat([train_data[key], torch.squeeze(value)], dim=0)

            pred = self(**train_data)
            obs_state = pred['obs_state']
            next_obs_state = torch.squeeze(pred['next_obs_state'])

            X = obs_state  # .detach().cpu().numpy()
            Y = next_obs_state  # .detach().cpu().numpy()
            # Compute the empirical transfer operator for this Isotypic subspace.
            result = torch.linalg.lstsq(X, Y, rcond=None)  # min||A X - B ||_F   |  B = A @ X

            op = result.solution.T
            self.transfer_op = op  # y = op @ x   op:(|y| x |x|)

    def get_metric_labels(self) -> list[str]:
        return ['pred_loss', 'S_score', 'ck_score', 'P_score', 'reg_orthonormal']

    def get_hparams(self):
        return {'encoder': self.projection.get_hparams(),
                }


if __name__ == "__main__":
    torch.set_printoptions(precision=3)
    path_to_data = Path('data')
    assert path_to_data.exists(), f"Invalid Dataset path {path_to_data.absolute()}"

    log.setLevel(logging.DEBUG)
    # Find all dynamic systems recordings
    path_to_data /= 'linear_system'
    path_to_dyn_sys_data = set([a.parent for a in list(path_to_data.rglob('*train.pkl'))])
    # Select a dynamical system
    mock_path = path_to_dyn_sys_data.pop()

    pred_horizon = 10
    batch_size = 2
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
    activation = torch.nn.Tanh
    equivariant = True
    #
    G = data_module.symm_group
    state_type = data_module.state_field_type
    obs_state_dimension = state_type.size * 2
    num_encoder_hidden_neurons = obs_state_dimension * 2
    max_ck_window_length = 6
    gspace = escnn.gspaces.no_base_space(G)

    # #
    dp_net = EquivDeepProjectionNet(state_type=data_module.state_field_type,
                                    obs_state_dimension=obs_state_dimension,
                                    num_encoder_layers=num_encoder_layers,
                                    num_encoder_hidden_neurons=num_encoder_hidden_neurons,
                                    max_ck_window_length=max_ck_window_length,
                                    activation=escnn.nn.IdentityModule,
                                    batch_norm=True,
                                    bias=True
                                    )

    # dp_net = DeepProjectionNet(state_dim=data_module.state_field_type.size,
    #                            obs_state_dimension=obs_state_dimension,
    #                            num_encoder_layers=num_encoder_layers,
    #                            num_encoder_hidden_neurons=num_encoder_hidden_neurons,
    #                            max_ck_window_length=max_ck_window_length)

    dp_net.to(device)

    # dp_net.approximate_transfer_operator(data_module.predict_dataloader())

    start_time = time.time()
    profiler = cProfile.Profile()
    profiler.enable()
    for i, batch in tqdm(enumerate(data_module.train_dataloader())):
        for k, v in batch.items():
            batch[k] = v.to(device)

        state, next_state = batch['state'], batch['next_state']
        n_steps = batch['next_state'].shape[1]

        # Test pre-processing function
        batched_state_traj = dp_net.pre_process_state(**batch)['state_trajectory'].tensor
        assert batched_state_traj.shape == (batch_size * (pred_horizon + 1), state.shape[-1]), f"state_traj: {batched_state_traj.shape}"

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
        dp_net.compute_loss_and_metrics(**out)
        if i > 10:
            break
    profiler.disable()

    print(f"Computing forward pass and loss/metrics for {i} batches took {time.time() - start_time:.2f}[s]"
          f"({(time.time() - start_time) / i:.2f} seconds per batch for {pred_horizon} steps in pred horizon)")

    print(f"\nDone here it is your Equivariant Dynamics Autoencoder :)")

    # Create a pstats object
    stats = pstats.Stats(profiler)

    # Sort stats by the cumulative time spent in the function
    stats.sort_stats('cumulative')

    # Print only the info for the functions defined in your script
    # Assuming your script's name is 'your_script.py'
    stats.print_stats('koopman_robotics')
