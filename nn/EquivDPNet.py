import cProfile
import copy
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
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.DynamicsDataModule import DynamicsDataModule
from nn.DPNet import DPNet, DmdSolver
from nn.TwinMLP import TwinMLP
from nn.emlp import EMLP
from utils.losses_and_metrics import forecasting_loss_and_metrics, obs_state_space_metrics
from utils.mysc import traj_from_states
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
                 group_avg_trick: bool = True,
                 dmd_algorithm: Optional[DmdSolver] = None,
                 activation: escnn.nn.EquivariantModule = escnn.nn.ReLU,
                 batch_norm: bool = True,
                 bias: bool = True,
                 dt: Union[float, int] = 1):

        # Default dmd algorithm naively exploiting symmetries
        dmd_algorithm = dmd_algorithm if dmd_algorithm is not None else self._full_rank_lstsq_symmetric
        self.gspace = state_type.gspace
        self.symm_group = self.gspace.fibergroup
        self.group_avg_trick = group_avg_trick
        # Number of regular fields in obs state and hidden layers of observable network
        multiplicity = math.ceil(obs_state_dim / state_type.size)
        if multiplicity < 1:
            raise ValueError(f"State-dim:{state_type.size}, |G|={self.symm_group.order()}, "
                             f"obs_dim:{obs_state_dim}")

        # Find the Isotypic basis of the state space and define the observation space representation as
        # `num_spect_field` copies of state representation (in isotypic basis).
        self.state_iso_reps, Q_iso2state = isotypic_basis(representation=state_type.representation,
                                                          multiplicity=1,
                                                          prefix='State')
        # Store the change of basis from original input basis to the isotypic basis of the space.
        self.Q_iso2state = torch.Tensor(Q_iso2state)
        self.Q_state2iso = torch.Tensor(np.linalg.inv(Q_iso2state))

        self.obs_iso_reps, _ = isotypic_basis(representation=state_type.representation,
                                              multiplicity=multiplicity,
                                              prefix='Obs')
        # Define the observation space representation in the isotypic basis.
        # Each Field for ESCNN will be an Isotypic Subspace.
        self.state_type = FieldType(self.gspace, [rep_iso for rep_iso in self.state_iso_reps.values()])
        self.in_type = self.state_type
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
        # regular_rep = self.symm_group.regular_representation
        # self.intermediate_type = FieldType(self.gspace, [regular_rep] * num_hidden_regular_fields)

        # Define a dict containing the transfer operator of each Isotypic subspace.
        self.iso_transfer_op = {irrep_id: None for irrep_id in self.obs_iso_reps.keys()}
        self.iso_inverse_projector = {irrep_id: None for irrep_id in self.obs_iso_reps.keys()}

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
                                         bias=bias, )

    def pre_process_state(self, state: Tensor, next_state=Tensor, **kwargs) -> GeometricTensor:
        self.Q_state2iso = self.Q_state2iso.to(state.device, dtype=state.dtype)
        flat_state_trajectory = super().pre_process_state(state, next_state, **kwargs)
        # Change basis to Isotypic basis.
        state_trajectory_iso_basis = torch.einsum('is,bs->bi', self.Q_state2iso, flat_state_trajectory)
        # Convert to Geometric Tensor
        return self.state_type(state_trajectory_iso_basis)

    def post_process_projections(self,
                                 obs_state_traj: GeometricTensor,
                                 obs_state_traj_prime: GeometricTensor) -> dict[str, Tensor]:
        return super().post_process_projections(obs_state_traj=obs_state_traj.tensor,
                                                obs_state_traj_prime=obs_state_traj_prime.tensor)

    def post_process_state(self, state_traj: Tensor) -> Tensor:
        self.Q_iso2state = self.Q_iso2state.to(state_traj.device, dtype=state_traj.dtype)
        # Convert to original basis
        state_traj_input_basis = torch.einsum('is,bts->bti', self.Q_iso2state, state_traj)
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
            iso_metrics = obs_state_space_metrics(obs_state_traj=obs_state_traj_iso,
                                                  obs_state_traj_prime=obs_state_traj_prime_iso,
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

    @torch.no_grad()
    def approximate_transfer_operator(self, train_data_loader: DataLoader):
        train_data = {}
        for batch in train_data_loader:
            for key, value in batch.items():
                if key not in train_data:
                    train_data[key] = torch.squeeze(value)
                else:
                    torch.cat([train_data[key], torch.squeeze(value)], dim=0)

        self.eval()
        pred = self(**train_data)
        self.train()

        # Change state space basis to Isotypic basis.
        state_iso_basis = torch.einsum('is,bs->bi', self.Q_state2iso, train_data["state"])
        obs_state_traj = pred["obs_state_traj"]

        assert obs_state_traj.shape[1] == 2, f"Expected single step datapoints, got {obs_state_traj.shape[1]} steps."
        obs_state, next_obs_state = obs_state_traj[:, 0, :], obs_state_traj[:, 1, :]

        # For each Isotypic Subspace, compute the empirical transfer operator with the present observable state space.
        for irrep_id, iso_rep in self.obs_iso_reps.items():
            obs_rep = iso_rep if irrep_id != self.symm_group.identity else None  # Check for Trivial
            state_rep = self.state_iso_reps[
                irrep_id] if irrep_id != self.symm_group.identity else None  # Check for Trivial

            # IsoSpace
            # Get the projection of the observable state in the isotypic subspace
            state_iso = state_iso_basis[..., self.state_iso_dims[irrep_id]]
            obs_state_iso = obs_state[..., self.obs_iso_dims[irrep_id]]
            next_obs_state_iso = next_obs_state[..., self.obs_iso_dims[irrep_id]]

            # Generate the data matrices of x(w_t) and x(w_t+1)
            X = obs_state_iso.T  # (batch_dim, obs_state_dim)
            X_prime = next_obs_state_iso.T  # (batch_dim, obs_state_dim)

            # Compute the empirical transfer operator of this Observable Isotypic subspace
            A_iso = self.dmd_algorithm(X, X_prime,
                                       rep_X=obs_rep if self.group_avg_trick else None,
                                       rep_Y=obs_rep if self.group_avg_trick else None)
            self.iso_transfer_op[irrep_id] = A_iso

            # Approximate a linear decoder from "main" observable Iso space to its associated Iso state subspace.
            Y = state_iso.T  # (state_dim, n_samples)
            self.iso_inverse_projector[irrep_id] = self._full_rank_lstsq_symmetric(
                X, Y,
                rep_X=obs_rep if self.group_avg_trick else None,
                rep_Y=state_rep if self.group_avg_trick else None)

        self.transfer_op = torch.block_diag(*self.iso_transfer_op.values())
        self.inverse_projector = torch.block_diag(*self.iso_inverse_projector.values())

        obs_one_step_error = torch.mean(torch.abs(next_obs_state.T - self.transfer_op @ obs_state.T), dim=-1)
        iso_one_step_error = [obs_one_step_error[self.obs_iso_dims[irrep_id]] for irrep_id in self.obs_iso_reps.keys()]
        rec_error = torch.mean(torch.abs(state_iso_basis.T - self.inverse_projector @ obs_state.T), dim=-1)
        iso_rec_error = [rec_error[self.state_iso_dims[irrep_id]] for irrep_id in self.state_iso_reps.keys()]

        return dict(solution_op_rank=torch.linalg.matrix_rank(self.transfer_op).to(dtype=torch.float),
                    solution_op_cond_num=torch.linalg.cond(self.transfer_op).to(dtype=torch.float),
                    solution_op_error=obs_one_step_error.to(dtype=torch.float).mean(),
                    solution_op_error_dist=torch.cat(iso_one_step_error),
                    inverse_projector_rank=torch.linalg.matrix_rank(self.inverse_projector),
                    inverse_projector_cond_num=torch.linalg.cond(self.inverse_projector),
                    inverse_projector_error=rec_error,
                    inverse_projector_error_dist=torch.cat(iso_rec_error),
                    rank_obs_state=torch.linalg.matrix_rank(obs_state))


    @staticmethod
    def _full_rank_lstsq_symmetric(
            X: Tensor, Y: Tensor, rep_X: Representation, rep_Y: Representation) -> Tensor:
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

        A = DPNet._full_rank_lstsq(X, Y)
        if rep_X is None or rep_Y is None:
            return A
        assert rep_Y.group == rep_X.group, "Representations must belong to the same group."

        # Do the group average trick to enforce equivariance.
        # This is equivalent to applying the group average trick on the singular vectors of the covariance matrices.
        A_G = []
        group = rep_X.group
        elements = group.elements if not group.continuous else group.grid(type='rand', N=group._maximum_frequency)
        for g in elements:
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

    def build_obs_fn(self, num_layers, backbone_layers=None, **kwargs):
        equivariant = True
        backbone_params = None
        # if num_layers > 3:
        #     num_backbone_layers = max(2, num_layers - 2)
        #     backbone_params = dict(in_type=self.state_type, out_type=self.intermediate_type,
        #                            num_layers=num_backbone_layers, head_with_activation=True, **copy.copy(kwargs))
        #     kwargs['with_bias'] = False
        #     kwargs['batch_norm'] = False
        #     obs_fn_params = dict(in_type=self.intermediate_type, out_type=self.obs_state_type,
        #                          num_layers=num_layers - num_backbone_layers, head_with_activation=False, **kwargs)
        # else:
        obs_fn_params = dict(in_type=self.state_type, out_type=self.obs_state_type, num_layers=num_layers,
                             head_with_activation=False, **kwargs)

        return TwinMLP(net_kwargs=obs_fn_params, backbone_kwargs=backbone_params, equivariant=equivariant)

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
