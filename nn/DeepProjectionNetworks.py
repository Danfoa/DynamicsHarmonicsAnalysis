import logging
from functools import reduce
from pathlib import Path
from typing import Optional, Tuple, Union

import escnn
import numpy as np
import torch
from escnn.nn import FieldType, GeometricTensor
from escnn.nn.modules.basismanager import BlocksBasisExpansion

from data.DynamicsDataModule import DynamicsDataModule
from nn.EquivDynamicsAutoencoder import isotypic_basis
from nn.mlp import EMLP
from utils.losses_and_metrics import (chapman_kolmogorov_regularization, compute_projection_score,
                                      compute_spectral_score, empirical_cov_cross_cov,
                                      regularization_orthonormality)

log = logging.getLogger(__name__)


class EquivDeepProjectionNet(escnn.nn.EquivariantModule):
    TIME_DIM = 1

    def __init__(self,
                 state_type: FieldType,
                 obs_state_dimension: int,
                 num_encoder_layers: int = 4,
                 num_encoder_hidden_neurons: int = 128,
                 max_ck_window_length: int = 6,
                 approx_surjective_obs=False,
                 activation: escnn.nn.EquivariantModule = escnn.nn.ReLU,
                 **kwargs):
        super().__init__()

        self.state_type = state_type
        self.in_type = state_type  # Keep the structure of ESCNN
        assert max_ck_window_length >= 2, "Minimum window_size of Chapman-Kolmogorov regularization is 2 steps"
        self.max_ck_window_length = max_ck_window_length
        gspace = state_type.gspace
        self.symm_group = gspace.fibergroup
        self.approx_surjective_obs = approx_surjective_obs

        # Number of regular fields in obs state and hidden layers of observable network
        num_regular_field = obs_state_dimension // state_type.size
        if num_regular_field < 1:
            raise ValueError(f"State-dim:{state_type.size}, |G|={self.symm_group.order()}, "
                             f"obs_dim:{obs_state_dimension}")

        # Compute the observation space Isotypic Rep from the regular representation
        self.obs_space_iso_reps = isotypic_basis(self.symm_group, num_regular_field, prefix='ObsSpace')
        # Define the observation space in the ISOTYPIC BASIS!
        self.obs_state_type = FieldType(gspace, [rep_iso for rep_iso in self.obs_space_iso_reps.values()])
        # Get hold of useful variables of Isotypic Decomposition
        self.iso_space_dims = {irrep_id: range(s, e) for s, e, irrep_id in zip(self.obs_state_type.fields_start,
                                                                               self.obs_state_type.fields_end,
                                                                               self.obs_space_iso_reps.keys())}

        self.out_type = self.obs_state_type  # Keep the structure of ESCNN
        # Compute the basis of Endomorphisms of the observation space. We can compute this by computing the
        # basis of Endomorphisms of each isotypic space.
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
        self.projection = EMLP(in_type=self.state_type,
                               out_type=self.obs_state_type,
                               num_hidden_units=num_encoder_hidden_neurons,
                               num_layers=num_encoder_layers,
                               activation=activation,
                               with_bias=True)

        if self.approx_surjective_obs:
            self.projection_inv = EMLP(in_type=self.obs_state_type,
                                       out_type=self.state_type,
                                       num_hidden_units=num_encoder_hidden_neurons,
                                       num_layers=num_encoder_layers,
                                       activation=activation,
                                       with_bias=True)

        # Private variables
        self._batch_dim = None  # Used to convert back and forward between Tensor and GeometricTensor.

    def project(self, state_trajectory: GeometricTensor, **kwargs) -> [dict[str, GeometricTensor]]:

        obs_trajectory = self.projection(state_trajectory)

        if self.approx_surjective_obs:
            state_traj = self.projection_inv(obs_trajectory)
            return dict(obs_state_trajectory=obs_trajectory, state_traj=state_traj)

        return dict(obs_state_trajectory=obs_trajectory)

    def pre_process_state(self, state: torch.Tensor,
                          next_state=torch.Tensor, **kwargs) -> Union[GeometricTensor, dict[str, GeometricTensor]]:
        self._batch_dim = state.shape[0]
        geom_next_state = torch.reshape(next_state, (-1, next_state.shape[-1]))

        state_trajectory = torch.cat([state, geom_next_state], dim=0)
        state_trajectory = self.state_type(state_trajectory)

        return dict(state_trajectory=state_trajectory)

    def post_process_pred(self,
                          obs_state_trajectory: GeometricTensor,
                          state_trajectory: Optional[GeometricTensor] = None) -> dict[str, torch.Tensor]:

        state_traj = torch.reshape(obs_state_trajectory.tensor, (self._batch_dim, -1, self.obs_state_type.size))
        obs_state = state_traj[:, 0, ...]        # First "time step" is the initial state observation
        next_obs_state = state_traj[:, 1:, ...]  # The Rest is the observations of the next states,

        if state_trajectory is not None:
            state_traj = torch.reshape(state_trajectory.tensor, (self._batch_dim, -1, self.state_type.size))
            state = state_traj[:self._batch_dim, ...]       # First "batch" is the initial state observation
            next_state = state_traj[self._batch_dim:, ...]  # Rest is the observations of the next states,
            # Return using the convention of the MarkovDynamicsModule
            return dict(state=state, next_state=next_state, obs_state=obs_state, next_obs_state=next_obs_state)
        return dict(obs_state=obs_state, next_obs_state=next_obs_state)

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
        # Apply any required pre-processing to the initial state and state trajectory
        input = self.pre_process_state(state, **kwargs)
        # Compute the observations of the trajectory of motion
        predictions = self.project(**input, n_steps=n_steps, )
        # Post-process predictions
        predictions = self.post_process_pred(**predictions)
        return predictions

    def loss_and_metrics(self,
                         predictions: dict[str, torch.Tensor],
                         ground_truth: dict[str, torch.Tensor]) -> (torch.Tensor, dict[str, torch.Tensor]):

        # Decompose batched observations in their isotypic spaces.
        obs_state = predictions['obs_state']
        next_obs_state = predictions['next_obs_state']

        iso_metrics = {irrep_id: {} for irrep_id in self.obs_space_iso_reps.keys()}

        # For each Isotypic Subspace, compute empirical Covariance and Cross-Covariance operators.
        # With these, compute spectral, projection scores and orthonormality and Chapman-Kolmogorov regularization.
        for irrep_id, iso_rep in self.obs_space_iso_reps.items():
            # Get the projection of the observable state in the isotypic subspace
            obs_state_iso = obs_state[..., self.iso_space_dims[irrep_id]]
            next_obs_state_iso = next_obs_state[..., self.iso_space_dims[irrep_id]]
            # Compute the empirical covariance and cross-covariance operators, ensuring that operators are equivariant.
            # Cov_t[i] = E[(X_i - E[X_i])(X_i - E[X_i])^T] | t in [0, pred_horizon + 1]
            # Cov_t0_ti[i] := Cov(X_0, X_t+1) = E[(X_0 - E[X_0])(X_t+1 - E[X_t+1])^T] | t in [0, pred_horizon]
            Cov_t, Cov_t0_ti = empirical_cov_cross_cov(state_0=obs_state_iso, next_states=next_obs_state_iso,
                                                       representation=iso_rep, cross_cov_only=False,
                                                       check_equivariance=log.level == logging.DEBUG)
            # Compute the Projection, Spectral and Orthonormality regularization terms for ALL time steps in horizon.
            # (t0, t) | t in [t0+1, pred_horizon]
            reg_orthonormal = regularization_orthonormality(Cov_t)
            spectral_score = compute_spectral_score(cov_x=Cov_t[:-1], cov_y=Cov_t[1:], cov_xy=Cov_t0_ti)
            projection_score = compute_projection_score(cov_x=Cov_t[:-1], cov_y=Cov_t[1:], cov_xy=Cov_t0_ti)
            ck_scores = chapman_kolmogorov_regularization(state_0=obs_state_iso, next_states=next_obs_state_iso,
                                                          ck_window_length=self.max_ck_window_length,
                                                          representation=iso_rep, cov_t0_ti=Cov_t0_ti,
                                                          verbose=log.level == logging.DEBUG)
            # Additional metrics
            covZZ_eigvals = torch.linalg.eigvalsh(Cov_t + (1e-6 * torch.eye(self.obs_state_type.size, device=covZZ.device)))


            iso_metrics['reg_orthonormal'] = reg_orthonormal
            iso_metrics['spectral_score'] = spectral_score
            iso_metrics['proj_score'] = projection_score
            iso_metrics['ck_scores'] = ck_scores

            print("HI")

        # Computing the cov and cross-covariance operator with the entire obs state dimension and the average trick
        # results in large equivariance error. Might be faster to compute on GPU but not accurate.
        # CovXX_iso, CovXY_iso = self.empirical_cov_cross_cov(obs_state, next_obs_state,
        #                                                     rep=self.obs_state_type.fiber_representation)

        print("Yes motherfucker !")
        # Compute the statistical metrics of the first

    def compute_statistical_metrics(self, X: torch.Tensor, iso_irrep_id: tuple):
        """
        For a multi-dimensional random variable Z this functions makes the following process: Center the samples,
        compute the covariance matrix `covZZ`, compute the Frobenius norm of the covariance matrix `covZZ_Fnorm`,
        and a metric indicating the independence/orthogonality of the dimensions of Z (i.e., the norm of the difference
        between the covariance matrix and the identity matrix, divided by dimension of the obs, so a value of 1 will
        indicate complete independece/orthogonality of dimensions).
        TODO: If samples are drawn from a symmetric random process we have that
          Expected value of variables E(Z) is invariant to symmetry transformations. Also, The covariance matrix is
          a linear operator that commutes with the group actions (same as Koopman operator) thus, in a
          "symmetry enabled" basis (exposing isotypic components) the covariance matrix is expected to
          be block-diagonal. We can exploit that known structure to reduce empirical estimation errors a sort of
          structural regularizaiton, one would say. Fun fact. E(Z) is 0 for all dimensions not invariant to all g in G.
        TODO: There is difference in numerical error between numpy and torch processing
          complex numbers. Despite having both double precision. Documentation shows a warning sign indicating for
          complex operations, saying we should use Cuda 11.6 (not sure how impactfull this is). We should follow.
        :param Z: Batched samples of the random variable Z, shape (M, dim(z)) | M = batch_size
        :return:
            - Z_centered: (torch.Tensor) Centered samples of Z (i.e., Z_uncentered - mean(Z_uncentered))
            - covZZ: (torch.Tensor) Covariance matrix of Z
            - covZZ_Fnorm: (float) Frobenius norm of the covariance matrix of Z
            - Z_dependence: (float) Metric measuring orthogonality/independence of dimensions of Z.
                0 means complete independece of dimensions.
        """
        assert Z.state_dim() == 2, "We expect a batch of samples of shape (M, dim(z)) | M = batch_size"
        # Z_mean = torch.mean(Z_uncentered, dim=0, keepdim=True)
        Z_centered = Z    # Avoid centering feature maps for now - Z_mean
        # Compute covariance per sample in batch
        covZZ = torch.matmul((Z_centered.conj()[:, :, None]), Z_centered[:, None, :])
        # Average over the M samples in batch. i.e. (M, dim(z), dim(z)) -> (dim(z), dim(z))
        covZZ = torch.mean(covZZ, dim=0)
        # assert torch.allclose(covZZ, covZZ.conj().T), "Covariance matrix should be hermitian (symmetric)!"
        # Compute eigvalues of the squared hermitian Cov(Z,Z) matrix. (How likely is to obtain a defective matrix here?)
        covZZ_eigvals = torch.linalg.eigvalsh(covZZ + (1e-6 * torch.eye(covZZ.shape[0], device=covZZ.device)))
        # Compute Frobenius norm of the covariance matrix
        covZZ_Fnorm = torch.norm(covZZ_eigvals)

        # Get the operator norm. i.e., the largest eigenvalue of the covariance matrix
        covZZ_OPnorm = covZZ_eigvals[-1]
        # Compute regularization term encouraging for orthogonality/independence of dimensions of obs vector
        # Z_dependence = (||Cov(Z,Z) - I||_F)^2
        Z_dependence = torch.sum(torch.square(covZZ_eigvals - 1))
        # z_dependence2 = torch.linalg.matrix_norm(covZZ - torch.eye(covZZ.shape[0], device=covZZ.device), ord='fro')**2
        return Z_centered, covZZ, covZZ_eigvals, covZZ_Fnorm, covZZ_OPnorm, Z_dependence


    def get_hparams(self):
        return {'encoder':      self.projection.get_hparams(),
                'num_isotypic_spaces': len(self.obs_space_iso_reps),
                }

    def evaluate_output_shape(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        batch_dim, state_dim = input_shape
        return batch_dim, self.obs_state_type.size


if __name__ == "__main__":
    path_to_data = Path('data')
    assert path_to_data.exists(), f"Invalid Dataset path {path_to_data.absolute()}"

    # Find all dynamic systems recordings
    path_to_data /= 'linear_system'
    path_to_dyn_sys_data = set([a.parent for a in list(path_to_data.rglob('*train.pkl'))])
    # Select a dynamical system
    mock_path = path_to_dyn_sys_data.pop()

    data_module = DynamicsDataModule(data_path=mock_path,
                                     pred_horizon=10,
                                     eval_pred_horizon=.5,
                                     frames_per_step=1,
                                     num_workers=2,
                                     batch_size=1000,
                                     augment=True
                                     )
    data_module.prepare_data()

    obs_state_dimension = 12
    dt = 0.1
    num_encoder_layers = 4
    num_encoder_hidden_neurons = obs_state_dimension * 2
    activation = torch.nn.Tanh
    equivariant = True
    #
    G = escnn.group.DihedralGroup(3)
    rep_state = G.regular_representation
    obs_state_dimension = rep_state.size * 3
    gspace = escnn.gspaces.no_base_space(G)

    state_type = FieldType(gspace, [rep_state])
    #
    dp_net = EquivDeepProjectionNet(state_type=data_module.state_field_type,
                                    obs_state_dimension=obs_state_dimension,
                                    num_encoder_layers=num_encoder_layers,
                                    num_encoder_hidden_neurons=num_encoder_hidden_neurons,
                                    activation=escnn.nn.ReLU,
                                    eigval_network=False,
                                    equivariant=equivariant, )
    dp_net.eval()

    for batch in data_module.train_dataloader():
        obs = dp_net(**batch)
        dp_net.loss_and_metrics(predictions=obs, ground_truth=batch)

    print(f"Done here it is your Equivariant Dynamics Autoencoder :)")
