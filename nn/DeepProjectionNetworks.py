import cProfile
import logging
import pstats
import time
from pathlib import Path
from typing import Optional, Tuple, Union

import escnn
import numpy as np
import torch
from escnn.nn import FieldType, GeometricTensor
from escnn.nn.modules.basismanager import BlocksBasisExpansion
from tqdm import tqdm

from data.DynamicsDataModule import DynamicsDataModule
from nn.EquivDynamicsAutoencoder import isotypic_basis
from nn.mlp import EMLP
from utils.losses_and_metrics import (chapman_kolmogorov_regularization, compute_chain_projection_scores,
                                      compute_chain_spectral_scores, empirical_cov_cross_cov,
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
                 ck_w: float = 0.1,
                 orthonormal_w: float = 0.1,
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
        self.ck_w = ck_w
        self.orthonormal_w = orthonormal_w
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

        num_params = sum([param.nelement() for param in self.parameters()])
        num_train_params = sum([param.nelement() for param in self.parameters() if param.requires_grad])
        log.info(f"Equiv-DPnet Num. Parameters: {num_params} ({num_train_params} trainable)\n")

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
        obs_state = state_traj[:, 0, ...]  # First "time step" is the initial state observation
        next_obs_state = state_traj[:, 1:, ...]  # The Rest is the observations of the next states,

        if state_trajectory is not None:
            state_traj = torch.reshape(state_trajectory.tensor, (self._batch_dim, -1, self.state_type.size))
            state = state_traj[:self._batch_dim, ...]  # First "batch" is the initial state observation
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

    def compute_loss_and_metrics(self,
                                 predictions: dict[str, torch.Tensor],
                                 ground_truth: dict[str, torch.Tensor]) -> (torch.Tensor, dict[str, torch.Tensor]):


        # Decompose batched observations in their isotypic spaces.
        obs_state = predictions['obs_state']
        next_obs_state = predictions['next_obs_state']
        time_horizon = obs_state.shape[1]

        device = obs_state.device
        start_time = time.time()
        log.debug(f"Computing Loss and Metrics for {predictions['obs_state'].shape[0]} samples. Device {device}")

        # Computing the cov and cross-covariance operator with the entire obs state dimension and the average trick
        # results in large equivariance error. Might be faster to compute on GPU but not accurate.
        # CovXX_iso, CovXY_iso = self.empirical_cov_cross_cov(obs_state, next_obs_state,
        #                                                     rep=self.obs_state_type.fiber_representation)

        # For each Isotypic Subspace, compute empirical Covariance and Cross-Covariance operators.
        # With these, compute spectral, projection scores and orthonormality and Chapman-Kolmogorov regularization.
        iso_losses = []
        iso_metrics = {irrep_id: {} for irrep_id in self.obs_space_iso_reps.keys()}
        for irrep_id, iso_rep in self.obs_space_iso_reps.items():
            rep = iso_rep if irrep_id != self.symm_group.trivial_representation else None  # Check for Trivial IsoSpace
            # Get the projection of the observable state in the isotypic subspace
            obs_state_iso = obs_state[..., self.iso_space_dims[irrep_id]]
            next_obs_state_iso = next_obs_state[..., self.iso_space_dims[irrep_id]]
            pred_horizon = next_obs_state_iso.shape[1]
            iso_state_dim = obs_state_iso.shape[-1]

            # Compute the empirical covariance and cross-covariance operators, ensuring that operators are equivariant.
            # Cov_t_tdt[i, j] := Cov(X_i, X_j)  | t in [0, pred_horizon], i,j in [0, pred_horizon], j >= i
            Cov_t_dt = empirical_cov_cross_cov(state_0=obs_state_iso, next_states=next_obs_state_iso,
                                               representation=rep, cov_window_size=self.max_ck_window_length,
                                               debug=False)  # log.level == logging.DEBUG)
            # Orthonormality regularization terms for ALL time steps in horizon
            # reg_orthonormal[t] = || Cov_t[i] - I || | t in [0, pred_horizon]
            Cov_t = Cov_t_dt[range(pred_horizon + 1), range(pred_horizon + 1)]
            reg_orthonormal = regularization_orthonormality(Cov_t)

            # Compute the Projection, Spectral and Orthonormality regularization terms for ALL time steps in horizon.
            # spectral_scores[t, t+d] := ||Cov(X_t, X_t+d)||^2_HS / (||CovX_t|| ||CovX_t+d||)   |
            #   t in [0, pred_horizon], d in [0, pred_horizon - t]
            spectral_scores = compute_chain_spectral_scores(Cov=Cov_t_dt,
                                                            window_size=self.max_ck_window_length,
                                                            debug=False)  # log.level == logging.DEBUG)
            projection_scores = compute_chain_projection_scores(Cov=Cov_t_dt,
                                                                window_size=self.max_ck_window_length,
                                                                debug=False)  # log.level == logging.DEBUG)
            # Apply the Orthonormal regularization term to the spectral scores
            spectral_scores_reg = torch.clone(spectral_scores)
            time_horizon = pred_horizon + 1
            for t in range(0, time_horizon):
                for dt in range(1, time_horizon - t):
                    spectral_scores_reg[t, t + dt] -= self.orthonormal_w * (reg_orthonormal[t] + reg_orthonormal[t + dt])
                    # projection_scores[t, t + dt] -= self.orthonormal_w * (reg_orthonormal[t] + reg_orthonormal[t + dt])

            # Compute the Chapman-Kolmogorov regularization scores for all possible step transitions. In return, we get:
            # ck_scores[i,j] = || Cov(X_i, X_j) - ( Cov(X_i, X_i+1), ... Cov(X_j-1, X_j) ) ||_2  | j >= i + 2
            ck_regularization = chapman_kolmogorov_regularization(Cov=Cov_t_dt,
                                                                  ck_window_length=self.max_ck_window_length,
                                                                  debug=False)  # log.level == logging.DEBUG)

            # Minimum number of steps to compute the CK regularization term
            # s_ck_scores[t, t+d] = mean(Σ_i=t^t+d s_scores[t, t+i]) - ck_scores[t, t+d]
            #                                               | t in [0, time_horizon - 2], d in [2, min(pred_horizon-t, ck_window)]
            dtype = obs_state_iso.dtype
            device = obs_state_iso.device
            min_steps = 2
            s_ck_scores = torch.fill(torch.zeros((time_horizon, time_horizon), dtype=dtype, device=device), torch.nan)
            iso_loss = torch.fill(torch.zeros_like(s_ck_scores, dtype=dtype, device=device), torch.nan)
            for t in range(0, time_horizon - 2):  # ts ∈ [0, time_horizon - 2]
                max_dt = min(time_horizon - t, self.max_ck_window_length)
                for dt in range(min_steps, max_dt):
                    td = t + dt
                    s_ck_scores[t, td] = torch.mean(spectral_scores_reg[t, t+1:td])
                    iso_loss[t, td] = -((s_ck_scores[t, td] - self.ck_w * ck_regularization[t, td]))

            non_nans = lambda x: torch.logical_not(torch.isnan(x))
            # Store the loss results.
            iso_losses.append(iso_loss[non_nans(iso_loss)])

            # Useful to debug the expected sparsity pattern of the matrix.
            # spectral_score_np = spectral_scores.detach().cpu().numpy()
            # spectral_score_reg_np = spectral_scores_reg.detach().cpu().numpy()
            # ck_reg_np = ck_regularization.detach().cpu().numpy()
            # s_ck_scores_np = s_ck_scores.detach().cpu().numpy()
            # loss_np = iso_loss.detach().cpu().numpy()

            iso_metrics[irrep_id]['reg_orthonormal'] = reg_orthonormal[non_nans(reg_orthonormal)]
            iso_metrics[irrep_id]['reg_orthonormal_scaled'] = (reg_orthonormal[non_nans(reg_orthonormal)]
                                                               / iso_state_dim**2)
            iso_metrics[irrep_id]['S_score'] = spectral_scores[non_nans(spectral_scores)]
            iso_metrics[irrep_id]['S_score_reg'] = spectral_scores_reg[non_nans(spectral_scores_reg)]
            iso_metrics[irrep_id]['P_score'] = projection_scores[non_nans(projection_scores)]
            iso_metrics[irrep_id]['CK_score'] = ck_regularization[non_nans(ck_regularization)]
            iso_metrics[irrep_id]['CK_score_scaled'] = ck_regularization[non_nans(ck_regularization)] / iso_state_dim**2
            # iso_metrics[irrep_id]['loss'] = iso_loss[non_nans(iso_loss)]
            # iso_metrics[irrep_id]['loss_scaled'] = iso_loss[non_nans(iso_loss)] / iso_state_dim**2
            log.debug(f"Isotypic Space {irrep_id} of dim {iso_state_dim} took {time.time() - start_time:.2f}[s]")

        metrics = dict(
            reg_orthonormal=torch.cat([iso_metrics[iso_id]['reg_orthonormal'] for iso_id in iso_metrics.keys()]),
            # reg_orthonormal_s=torch.cat([iso_metrics[iso_id]['reg_orthonormal_scaled'] for iso_id in iso_metrics.keys()]),
            S_score=torch.cat([iso_metrics[iso_id]['S_score'] for iso_id in iso_metrics.keys()]),
            S_score_reg=torch.cat([iso_metrics[iso_id]['S_score_reg'] for iso_id in iso_metrics.keys()]),
            P_score=torch.cat([iso_metrics[iso_id]['P_score'] for iso_id in iso_metrics.keys()]),
            ck_score=torch.cat([iso_metrics[iso_id]['CK_score'] for iso_id in iso_metrics.keys()]),
            # ck_score_s=torch.cat([iso_metrics[iso_id]['CK_score_scaled'] for iso_id in iso_metrics.keys()]),
            # loss_s=torch.cat([iso_metrics[iso_id]['loss_scaled'] for iso_id in iso_metrics.keys()]),
            # loss=torch.cat([iso_metrics[iso_id]['loss'] for iso_id in iso_metrics.keys()]),
            )

        loss = torch.mean(torch.cat(iso_losses))

        assert not torch.isnan(loss), f"Loss is NaN. Metrics: {metrics}"
        log.debug(f"Computing Loss and Metrics took {time.time() - start_time:.2f}[s]")
        return loss, metrics

    def get_hparams(self):
        return {'encoder':             self.projection.get_hparams(),
                'num_isotypic_spaces': len(self.obs_space_iso_reps),
                }

    def evaluate_output_shape(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        batch_dim, state_dim = input_shape
        return batch_dim, self.obs_state_type.size


if __name__ == "__main__":
    torch.set_printoptions(precision=5)
    path_to_data = Path('data')
    assert path_to_data.exists(), f"Invalid Dataset path {path_to_data.absolute()}"

    log.setLevel(logging.DEBUG)
    # Find all dynamic systems recordings
    path_to_data /= 'linear_system'
    path_to_dyn_sys_data = set([a.parent for a in list(path_to_data.rglob('*train.pkl'))])
    # Select a dynamical system
    mock_path = path_to_dyn_sys_data.pop()

    pred_horizon = 25
    device = torch.device('cuda:0')
    data_module = DynamicsDataModule(data_path=mock_path,
                                     pred_horizon=pred_horizon,
                                     eval_pred_horizon=100,
                                     frames_per_step=1,
                                     num_workers=0,
                                     batch_size=1024,
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
    obs_state_dimension = state_type.size * 20
    num_encoder_hidden_neurons = obs_state_dimension * 2
    gspace = escnn.gspaces.no_base_space(G)

    #
    dp_net = EquivDeepProjectionNet(state_type=data_module.state_field_type,
                                    obs_state_dimension=obs_state_dimension,
                                    num_encoder_layers=num_encoder_layers,
                                    num_encoder_hidden_neurons=num_encoder_hidden_neurons,
                                    max_ck_window_length=6,
                                    activation=escnn.nn.ReLU,
                                    eigval_network=False,
                                    equivariant=equivariant, )
    dp_net.to(device)

    start_time = time.time()
    profiler = cProfile.Profile()
    profiler.enable()
    for i, batch in tqdm(enumerate(data_module.train_dataloader())):
        for k, v in batch.items():
            batch[k] = v.to(device)
        obs = dp_net(**batch)
        dp_net.compute_loss_and_metrics(predictions=obs, ground_truth=batch)
        if i > 100:
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

    # TODO: Have to parallelize compute_chain_spectral_scores.
    # Print only the top 50 functions sorted by time
    stats.print_stats(50)