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
from escnn.nn.modules.basismanager import BlocksBasisExpansion
from lightning import seed_everything
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.DynamicsDataModule import DynamicsDataModule
from nn.DPNet import DPNet, DmdSolver
from nn.mlp import EMLP
from utils.losses_and_metrics import forecasting_loss_and_metrics, obs_state_space_metrics
from utils.representation_theory import isotypic_basis

log = logging.getLogger(__name__)


class EquivDPNet(DPNet):

    def __init__(self,
                 state_type: FieldType,
                 obs_state_dim: int,
                 num_layers: int = 4,
                 num_hidden_units: int = 128,
                 max_ck_window_length: int = 6,
                 ck_w: float = 0.0,
                 orth_w: float = 0.1,
                 dmd_algorithm: Optional[DmdSolver] = None,
                 activation: escnn.nn.EquivariantModule = escnn.nn.ReLU,
                 batch_norm: bool = True,
                 bias: bool = True,
                 dt: Union[float, int] = 1):

        # Default dmd algorithm naively exploiting symmetries
        dmd_algorithm = dmd_algorithm if dmd_algorithm is not None else self._full_rank_lstsq_symmetric
        self.state_type = self.in_type = state_type
        self.gspace = state_type.gspace
        self.symm_group = self.gspace.fibergroup

        # Number of regular fields in obs state and hidden layers of observable network
        multiplicity = math.ceil(obs_state_dim / state_type.size)
        if multiplicity < 1:
            raise ValueError(f"State-dim:{state_type.size}, |G|={self.symm_group.order()}, "
                             f"obs_dim:{obs_state_dim}")

        # Find the Isotypic basis of the state space and define the observation space representation as
        # `num_spect_field` copies of state representation (in isotypic basis).
        self.state_iso_reps = isotypic_basis(representation=self.state_type.representation,
                                             multiplicity=1,
                                             prefix='Space')
        self.obs_iso_reps = isotypic_basis(representation=self.state_type.representation,
                                           multiplicity=multiplicity,
                                           prefix='ObsSpace')
        # Define the observation space representation in the isotypic basis.
        # Each Field for ESCNN will be an Isotypic Subspace.
        self.obs_state_type = FieldType(self.gspace, [rep_iso for rep_iso in self.obs_iso_reps.values()])
        self.out_type = self.obs_state_type

        # Auxiliary variables determining the start and end dimensions of Isotypic subspaces of observable space.
        self.obs_iso_dims = {irrep_id: range(s, e) for s, e, irrep_id in zip(self.obs_state_type.fields_start,
                                                                             self.obs_state_type.fields_end,
                                                                             self.obs_iso_reps.keys())}
        self.state_iso_dims = {irrep_id: range(s, e) for s, e, irrep_id in zip(self.state_type.fields_start,
                                                                               self.state_type.fields_end,
                                                                               self.state_iso_reps.keys())}

        # Define the representation and field type of the hidden layers of the encoder/observable network.
        # TODO: Shift to an Isotypic basis for the hidden layers with Irrep-Norm-Relu activations.
        num_hidden_regular_fields = int(np.ceil(num_hidden_units / self.symm_group.order()))
        regular_rep = self.symm_group.regular_representation
        self.intermediate_type = FieldType(self.gspace, [regular_rep] * num_hidden_regular_fields)

        # Define a dict containing the transfer operator of each Isotypic subspace.
        self.iso_transfer_op = {irrep_id: None for irrep_id in self.obs_iso_reps.keys()}

        super(EquivDPNet, self).__init__(state_dim=state_type.size,
                                         obs_state_dim=obs_state_dim,
                                         dt=dt,
                                         num_layers=num_layers,
                                         num_hidden_units=num_hidden_units,
                                         max_ck_window_length=max_ck_window_length,
                                         ck_w=ck_w,
                                         orth_w=orth_w,
                                         dmd_algorithm=dmd_algorithm,
                                         activation=activation,
                                         batch_norm=batch_norm,
                                         bias=bias)

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
    def forecast(self,
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
        for irrep_id, iso_rep in self.obs_iso_reps.items():
            # Get the projection of the observable state in the isotypic subspace
            obs_state_iso = obs_state[..., self.obs_iso_dims[irrep_id]]

            # Use the empirical transfer operator to compute the maximum likelihood prediction of the trajectory
            pred_iso_state_traj = [obs_state_iso]
            # The transfer operator of this Isotypic subspace
            iso_transfer_op = self.iso_transfer_op[irrep_id]
            for step in range(n_steps):
                # Compute the next state prediction s_t+1 = K @ s_t
                next_iso_state = (iso_transfer_op @ pred_iso_state_traj[-1].T).T
                pred_iso_state_traj.append(next_iso_state)

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
        iso_spaces_metrics = {irrep_id: {} for irrep_id in self.obs_iso_reps.keys()}
        for irrep_id, iso_rep in self.obs_iso_reps.items():
            rep = iso_rep if irrep_id != self.symm_group.trivial_representation else None  # Check for Trivial IsoSpace
            # Get the projection of the observable state in the isotypic subspace
            obs_state_iso = obs_state[..., self.obs_iso_dims[irrep_id]]
            next_obs_state_iso = next_obs_state[..., self.obs_iso_dims[irrep_id]]

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
                                 ck_reg=obs_space_metrics['ck_score'],
                                 orth_reg=obs_space_metrics['orth_reg'])

        # Include prediction metrics if available
        if pred_next_obs_state is not None:
            assert pred_next_obs_state.shape == next_obs_state.shape
            pred_loss, pred_metrics = forecasting_loss_and_metrics(state_gt=next_obs_state,
                                                                   state_pred=pred_next_obs_state)
            pred_metrics['obs_pred_loss'] = pred_loss
            pred_metrics['obs_pred_loss_t'] = pred_metrics.pop('pred_loss_t')  # Change metric name.
            cond_num_transfer_op = torch.linalg.cond(self.transfer_op)
            pred_metrics['cond_num_transfer_op'] = cond_num_transfer_op
            iso_cond_num = torch.stack([torch.linalg.cond(A) for A in self.iso_transfer_op.values()], dim=0)
            pred_metrics['cond_num_transfer_op_dist'] = iso_cond_num
            obs_space_metrics.update(pred_metrics)
            if log.level == logging.DEBUG and cond_num_transfer_op > 100:
                log.warning(f"Condition number of transfer operator: {cond_num_transfer_op:.2f}.")

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
            [iso_spaces_metrics[irrep_id]['orth_reg'] for irrep_id in self.obs_iso_reps.keys()])
        obs_space_orth_reg = torch.sqrt(torch.sum(iso_orth_reg ** 2, dim=0))

        # Compute the entire obs space CK regularization terms.
        # ck_reg[t, t+d] = ||Cov(t, t+d) - (Cov(t, t+1)...Cov(t+d-1, t+d))||_Fro
        #                = sqrt(sum_iso(||Cov_iso(t, t+d) - (Cov_iso(t, t+1)...Cov_iso(t+d-1, t+d))||^2_Fro))
        iso_ck_reg = torch.stack(
            [iso_spaces_metrics[irrep_id]['ck_reg'] for irrep_id in self.obs_iso_reps.keys()], dim=0)
        obs_space_ck_reg = torch.sqrt(torch.sum(iso_ck_reg ** 2, dim=0))

        # Compute the Correlation/Projection score
        # P_score[t, t+d] = ||Cov(t)^-1/2 Cov[t, t+d] Cov(t+d)^-1/2||_HS
        #                 = sqrt(sum_iso(||Cov_iso(t)^-1/2 Cov_iso(t, t+d) Cov_iso(t+d)^-1/2||_HS^2))
        #                 = sqrt(sum_iso(P_score_iso[t, t+d]^2))
        iso_P_score = torch.stack(
            [iso_spaces_metrics[irrep_id]['projection_score'] for irrep_id in self.obs_iso_reps.keys()], dim=0)
        obs_space_P_score = torch.sqrt(torch.sum(iso_P_score ** 2, dim=0))

        # Compute the Spectral scores for the entire obs-space
        # S_score[t, t+d] = ||Cov(t, t+d)||_HS / (||Cov(t)||_HS ||Cov(t+d)||_HS)
        #                 <= ||Cov(t)^-1/2 Cov(t,t+d) Cov(t+d)^-1/2||_HS
        #                 <= sqrt(sum_iso((||Cov_iso(X,Y)||_HS / (||Cov_iso(X)||_HS ||Cov_iso(Y)||_HS)^2)))
        #                 <= sqrt(sum_iso(S_score_iso[t, t+d]^2)))
        iso_S_score = torch.stack(
            [iso_spaces_metrics[irrep_id]['spectral_score'] for irrep_id in self.obs_iso_reps.keys()], dim=0)
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

        # Obtain the matrix (T, max_ck_window_length) of spectral scores for all possible dt transitions.
        spectral_score_t = torch.fill(torch.zeros((time_horizon - 1, self.max_ck_window_length - 1),
                                                  dtype=dtype, device=device), torch.nan)
        projection_score_t = torch.fill(torch.zeros_like(spectral_score_t), torch.nan)
        for n_steps in range(1, self.max_ck_window_length):
            # Spectral scores for transitions of `n_steps`
            n_steps_s_scores = torch.diagonal(obs_space_S_score, offset=n_steps)
            n_steps_p_scores = torch.diagonal(obs_space_P_score, offset=n_steps)
            spectral_score_t[:len(n_steps_s_scores), n_steps - 1] = n_steps_s_scores
            projection_score_t[:len(n_steps_p_scores), n_steps - 1] = n_steps_p_scores

        non_nans = lambda x: torch.logical_not(torch.isnan(x))
        return dict(orth_reg=torch.mean(obs_space_orth_reg),
                    ck_score=torch.mean(ck_scores_reg[non_nans(ck_scores_reg)]),
                    ck_reg=torch.mean(obs_space_ck_reg[non_nans(obs_space_ck_reg)]),
                    spectral_score=torch.mean(obs_space_S_score[non_nans(obs_space_S_score)]),
                    projection_score=torch.mean(obs_space_P_score[non_nans(obs_space_P_score)]),
                    projection_score_t=torch.nanmean(projection_score_t, dim=0, keepdim=True),  # (batch, time)
                    spectral_score_t=torch.nanmean(spectral_score_t, dim=0, keepdim=True)  # (batch, time)
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
        state = batch['state']
        obs_state = pred['obs_state']
        next_obs_state = torch.squeeze(pred['next_obs_state'])

        # For each Isotypic Subspace, compute the empirical transfer operator with the present observable state space.
        for irrep_id, iso_rep in self.obs_iso_reps.items():
            obs_rep = iso_rep if irrep_id != self.symm_group.identity else None  # Check for Trivial
            state_rep = self.state_iso_reps[irrep_id] if irrep_id != self.symm_group.identity else None  # Check for Trivial

            # IsoSpace
            # Get the projection of the observable state in the isotypic subspace
            state_iso = state[..., self.state_iso_dims[irrep_id]]
            obs_state_iso = obs_state[..., self.obs_iso_dims[irrep_id]]
            next_obs_state_iso = next_obs_state[..., self.obs_iso_dims[irrep_id]]

            # Generate the data matrices of x(w_t) and x(w_t+1)
            X = obs_state_iso  # (batch_dim, obs_state_dim)
            X_prime = next_obs_state_iso  # (batch_dim, obs_state_dim)

            # Compute the empirical transfer operator of this Observable Isotypic subspace
            A_iso = self.dmd_algorithm(X, X_prime, rep_X=obs_rep, rep_Y=obs_rep)
            self.iso_transfer_op[irrep_id] = A_iso

            # Approximate a linear decoder from "main" observable Iso space to its associated Iso state subspace.
            X = obs_state_iso.T.conj()  # (obs_state_dim, n_samples)
            Y = state_iso.T.conj()  # (state_dim, n_samples)
            self.iso_inverse_projector[irrep_id] = self._full_rank_lstsq_symmetric(X, Y, rep_X=obs_rep, rep_Y=state_rep)

        op = torch.block_diag(*self.iso_transfer_op.values())
        self.transfer_op = op
        self.inverse_projector = torch.block_diag(*self.iso_inverse_projector.values())

    @staticmethod
    def _full_rank_lstsq_symmetric(
            X: torch.Tensor, Y: torch.Tensor, rep_X: Representation, rep_Y: Representation) -> torch.Tensor:
        """ Compute the least squares solution of the linear system Y = A·X.

        If the representation is provided the empirical transfer operator is improved using the group average trick to
        enforce equivariance considering that:
                            rep_Y(g) y = A rep_X(g) x
                        rep_Y(g) (A x) = A rep_X(g) x
                            rep_Y(g) A = A rep_X(g)
                rep_Y(g) A rep_X(g)^-1 = A                | forall g in G.

        Args:
            X: (|x|, n_samples) Data matrix of the initial states.
            Y: (|y|, n_samples) Data matrix of the next states.
        Returns:
            A: (|y|, |x|) Least squares solution of the linear system `Y = A·X`.
        """
        assert rep_Y.group == rep_X.group, "Representations must belong to the same group."

        A = DPNet._full_rank_lstsq(X, Y)
        # Do the group average trick to enforce equivariance.
        # This is equivalent to applying the group average trick on the singular vectors of the covariance matrices.
        A_G = []
        group = rep_X.group
        for g in group.elements:
            if g == group.identity:
                A_g = A
            else:
                rep_X_g = torch.from_numpy(rep_X(g)).to(dtype=X.dtype, device=X.device)
                rep_Y_g_inv = torch.from_numpy(rep_Y(~g)).to(dtype=X.dtype, device=X.device)
                A_g = rep_X_g @ A @ rep_Y_g_inv
            A_G.append(A_g)
        A_G = torch.stack(A_G, dim=0)
        A_G = torch.mean(A_G, dim=0)
        return A_G

    def _compute_endomorphism_basis(self):
        # When approximating the transfer/Koopman operator from the symmetric observable space, we know the operator
        # belongs to the space of G-equivariant operators (Group Endomorphism of the observable space).
        # Using ESCNN we can compute the basis of the Endomorphism space, and use this basis to compute an empirical
        # G-equivariant approximation of the transfer operator. Since the observable space is defined in the Isotypic
        # basis, the operator is block-diagonal and the basis of the Endomorphism space is the block diagonal sum of
        # the
        # basis of the Endomorphism space of each Isotypic subspace.
        self.iso_space_basis = {}
        self.iso_space_basis_mask = {}  # Used to mask empirical covariance between orthogonal observables
        for irrep_id, rep in self.obs_iso_reps.items():
            iso_basis = BlocksBasisExpansion(in_reprs=[self.symm_group.irrep(*id) for id in rep.irreps],
                                             out_reprs=[self.symm_group.irrep(*id) for id in rep.irreps],
                                             basis_generator=self.gspace.build_fiber_intertwiner_basis,
                                             points=np.zeros((1, 1)))
            self.iso_space_basis[irrep_id] = iso_basis
            basis_coefficients = torch.rand((iso_basis.dimension(),)) + 2
            non_zero_elements = iso_basis(basis_coefficients)[:, :, 0]
            mask = torch.logical_not(torch.isclose(non_zero_elements, torch.zeros_like(non_zero_elements), atol=1e-6))
            self.iso_space_basis_mask[irrep_id] = mask

    def build_obs_fn(self, activation, batch_norm, bias, num_hidden_units, num_layers):
        return EMLP(in_type=self.state_type,
                    out_type=self.intermediate_type,
                    num_hidden_units=num_hidden_units,
                    num_layers=num_layers - 2,
                    head_with_activation=True,
                    batch_norm=batch_norm,
                    activation=activation,
                    with_bias=bias)

    def __repr__(self):
        str = super().__repr__()
        str += (f"\tState Space fields={self.state_type.representations} "
                f"\n\t\tirreps={self.state_type.representation.irreps}"
                f"\n\tObservation Space fields={self.obs_state_type.representations} "
                f"\n\t\tirreps={self.obs_state_type.representation.irreps}")
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
    path_to_data = Path('data')
    assert path_to_data.exists(), f"Invalid Dataset path {path_to_data.absolute()}"

    log.setLevel(logging.DEBUG)
    # Find all dynamic systems recordings
    path_to_data /= 'linear_system'
    path_to_dyn_sys_data = set([a.parent for a in list(path_to_data.rglob('*train.pkl'))])
    # Select a dynamical system
    mock_path = path_to_dyn_sys_data.pop()

    pred_horizon = 50
    batch_size = 1024
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

    state_type = data_module.state_field_type
    obs_state_dimension = state_type.size * 1
    num_encoder_hidden_neurons = obs_state_dimension * 2
    max_ck_window_length = pred_horizon

    dp_net = EquivDPNet(state_type=data_module.state_field_type,
                        obs_state_dim=obs_state_dimension,
                        num_layers=num_encoder_layers,
                        num_hidden_units=num_encoder_hidden_neurons,
                        max_ck_window_length=max_ck_window_length,
                        activation=escnn.nn.ReLU,
                        bias=False,
                        batch_norm=False)

    dp_net.to(device)

    dp_net.approximate_transfer_operator(data_module.predict_dataloader())

    start_time = time.time()
    profiler = cProfile.Profile()
    profiler.enable()
    for i, batch in tqdm(enumerate(data_module.train_dataloader())):
        for k, v in batch.items():
            batch[k] = v.to(device)
        batch_size = batch['state'].shape[0]
        state, next_state = batch['state'], batch['next_state']
        n_steps = batch['next_state'].shape[1]

        # Test pre-processing function
        batched_state_traj = dp_net.pre_process_state(**batch)['state_trajectory']

        state_traj_non_flat = torch.reshape(batched_state_traj.tensor,
                                            (batch_size, pred_horizon + 1, state.shape[-1]), )
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
        figs['prediction'].show()
        print(metrics.get('pred_loss', None))
        if i > 1:
            break
    profiler.disable()

    # print(f"Computing forward pass and loss/metrics for {id} batches took {time.time() - start_time:.2f}[s]"
    #       f"({(time.time() - start_time) / i:.2f} seconds per batch for {pred_horizon} steps in pred horizon)")

    # Create a pstats object
    stats = pstats.Stats(profiler)

    # Sort stats by the cumulative time spent in the function
    stats.sort_stats('cumulative')

    # Print only the info for the functions defined in your script
    # Assuming your script's name is 'your_script.py'
    stats.print_stats('koopman_robotics')
