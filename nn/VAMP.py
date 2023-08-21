from typing import Iterable

import torch

from data.dynamics_dataset import STATES, CTRLS
from nn.EquivariantLinearDynamics import EquivariantLinearDynamics
from src.RobotEquivariantNN.nn.EMLP import MLP


import logging

from utils.losses_and_metrics import observation_dynamics_error

log = logging.getLogger(__name__)

class VAMP(torch.nn.Module):

    def __init__(self, state_dim: int, obs_dim: int, dt: float, pred_horizon: int = 1, reg_lambda=1e-2, n_layers=3,
                 n_hidden_neurons=32, n_head_layers=None, robot=None, activation: torch.nn.Module = torch.nn.ReLU,
                 **kwargs):
        """
        :param pred_horizon: Number of time-steps to compute correlation score.
        :param activation:
        :return:
        """
        super().__init__()
        assert isinstance(pred_horizon, int) and pred_horizon >= 1, "Number of time-steps must be an integer >= 1"
        self.pred_horizon = pred_horizon
        assert isinstance(dt, float) and dt > 0, "Time-step must be a positive float"
        self.dt =dt
        assert reg_lambda >= 0, "Regularization coefficient must be non-negative"
        self.reg_lambda = reg_lambda
        n_head_layers = n_layers if n_head_layers is None else n_head_layers
        output_dim = obs_dim * 2  # Complex observations of dim (..., 2) (real and imaginary parts)

        # We have a core encoder/feature-extraction-module processing all time-steps of the trajectory
        # and then we have `m=look_ahead` heads that output the observations of each time-step separately
        # Ideally sharing the same backbone will make this module efficient at extracting the features required to
        # create the observations at each time-step.
        # Input to encoder is expected to be [batch_size, time_steps, state_dim]
        self.encoder = MLP(d_in=state_dim, d_out=output_dim, ch=n_hidden_neurons, n_layers=n_layers,
                           activation=[activation] * (n_layers - 1) + [torch.nn.Identity],)
                           # activation=[activation] * n_layers)

        # heads = []
        # for head_id in range(pred_horizon + 1):
        #     # Each head is constructed as squared perceptrons, one with non-linearity and the last without.
        #     # This is a convenient convention. Nothing special behind it.
        #     n_head_layers = 2
        #     heads.append(MLP(d_in=n_hidden_neurons, d_out=output_dim, ch=n_hidden_neurons, n_layers=n_head_layers,
        #                      with_bias=False, activation=[activation] * (n_head_layers - 1) + [torch.nn.Identity]))
        #
        # self.heads = torch.nn.ModuleList(heads)

        # Module for evolving observations in eigenspace basis.
        self.observation_dynamics = EquivariantLinearDynamics(state_dim=obs_dim, trainable=False)
        self._set_updated_eigenmatrix(False)
        # Everytime the observation function approximation is modified, we need to update Koopman approximation.
        # To detect changes in obs function, we place a backward hook on its parameters.
        self.register_full_backward_hook(hook=self._backward_hook)

    def forward(self, x):
        # Backbone operation
        encoder_out = self.encoder(x)

        # observations = []
        # for i, head in enumerate(self.heads):
        #     # Each head computes observation functions from each timestep [t+i*dt]
        #     head_out = head(encoder_out[:, i, :])
        #     z_t_idt = self.realobs2complex(head_out)
        #     observations.append(z_t_idt)
        # obs = torch.stack(observations, dim=1)  # [batch_size, pred_horizon+1, obs_dim]
        # psi = torch.reshape(encoder_out, shape=(encoder_out.shape[:-1] + (encoder_out.shape[-1] // 2, 2)))
        # z_t_idt = torch.view_as_complex(psi)
        # obs = z_t_idt
        return self.realobs2complex(encoder_out)

    def realobs2complex(self, obs):
        # obs is composed of:
        # (re(Ψi_1(x_t+i*dt), img(Ψi_1(x_t+i*dt),..., re(Ψi_N(x_t+i*dt)), img(Ψi_N(x_t+i*dt))),
        psi = torch.reshape(obs, shape=(obs.shape[:-1] + (obs.shape[-1] // 2, 2)))
        return torch.view_as_complex(psi)

    def forecast(self, x):
        # Forcasting uses only the first head as the obs function.
        # Compute observations
        z = self.forward(x)
        # z_real = self.heads[0](encoder_out)
        # z = self.realobs2complex(z_real)

        n_frames = z.shape[1]
        # Take initial observation z0 and use it to forcast using eigenfunction/value dynamics.
        z0 = z[:, 0, :]

        z_pred = torch.zeros_like(z)
        z_pred[:, 0, :] = z0
        for i in range(1, n_frames):
            z_pred[:, i, :] = torch.matmul(z_pred[:, i-1, :], self.K)

        # TODO: Once everything works use eigendecomp for forcasting
        # dts = torch.arange(0, n_frames, device=z.device) * self.dt
        # z_pred, z_eigen_pred = self.observation_dynamics.forcast(z0, dts, obs_in_eigenbasis=False)
        #
        # Change basis of observations to compute metrics in eigenbasis
        # z_eigen = self.observation_dynamics.obs2eigenbasis(z)
        # return obs and predictions in eigenbasis for now.
        # return {'z': z_eigen, 'z_pred': z_eigen_pred, 'z_obs': z, 'z_pred_obs': z_pred}
        return {'z': z, 'z_pred': z_pred}

    def approximate_koopman_op(self, bacthed_obs: list[torch.Tensor]):
        assert isinstance(bacthed_obs, list), "We expect a list of observations of [(batch_size, time, obs_dim), ...]"
        assert bacthed_obs[0].ndim == 3, "We expect observations of shape (batch_size, time, obs_dim)"
        n_frames = bacthed_obs[0].shape[1]
        assert n_frames >= 2, "We expect at least 2 observations to compute correlation score"

        # TODO: We will face a problem once the storage of all training data becomes too large. For iterative
        #   DMD we have https://oar.princeton.edu/bitstream/88435/pr1002j/1/RowleyPoFV26-N11-2014.pdf
        # Define the data matrices X':(D, N) and Y':(D, M) containing the observations at time t and t+1
        X, Y = [], []
        for t in range(n_frames - 1):
            X += [obs[:, t, :] for obs in bacthed_obs]
            Y += [obs[:, t+1, :] for obs in bacthed_obs]
        X = torch.cat(X, dim=0)
        Y = torch.cat(Y, dim=0)

        # DEBUG test Least Square by testing a random linear dynamical sytem.
        # K_p = torch.rand((X.shape[-1], Y.shape[-1]), device=X.device, dtype=torch.cfloat)
        # YY = torch.matmul(X, K_p)
        # sol = torch.linalg.lstsq(X, YY)
        # KK = sol.solution
        # Y_pred = torch.matmul(X, KK)
        # error = torch.mean(torch.abs(Y_pred - YY))

        # Torch convention uses X':(D, N) and Y':(D, M) to solve the least squares problem Y' = X'·K', where
        # K':(N, M) is our approximate Koopman operator, for the current function space.
        sol = torch.linalg.lstsq(X, Y)
        K = sol.solution
        self.K = K
        # Decompose Koopman operator into K = V·Λ·V^-1 where V are the eigenvectors and Λ are the eigenvalues.
        eigvals, eigvects = torch.linalg.eig(self.K)

        # K_p = eigvects @ torch.diag(eigvals) @ torch.linalg.inv(eigvects)
        # assert torch.allclose(K_p, self.K, atol=1e-3), "-"
        self.observation_dynamics.update_eigvals_and_eigvects(eigvals, eigvects)
        self._set_updated_eigenmatrix(True)

    def evaluate_observation_space(self, obs: torch.Tensor, state: torch.Tensor) -> dict:
        out = self.forecast(state)
        metrics = observation_dynamics_error(z=out['z'][:, 1:, :], z_pred=out['z_pred'][:, 1:, :])
        return metrics

    def compute_loss_metrics(self, obs, _):
        """
        This function takes a list of observations per time frame ([M, z_t+i*dt] | i ∈ [0,..., n_frames),
            M=batch_size/n_samples) and computes the generalized Rayleigh score between z_0 and each
            following time frame z_0+i*dt | i ∈ [1,...,n_frames).
            TODO: result is an avg of the correlation scores??
        :param obs:
        :return: loss: torch.Tensor containing the negative of the average of the regularized generalized Rayleigh
            scores between z_0 and each following time frame z_0+i*dt | i ∈ [1,...,n_frames).
            metrics: dict containing debug metrics
        """
        assert obs.ndim == 3, "We expect a list of observations of shape (batch_size, time, obs_dim)"
        n_frames = obs.shape[1]
        assert n_frames >= 2, "We expect at least 2 observations to compute correlation score"

        gen_rayleigh_quotients = []       # Rayleigh quotient for each time-step
        reg_gen_rayleigh_quotients = []   # Regularized Rayleigh quotient for each time-step
        obs_independence_scores = []              # Regularization terms indicating orthogonality of observations
        obs_covZZ_Fnorms, obs_covZZ_OPnorms, obs_covZZ_sval_min = [], [], []
        obs_covXY_sval_min, obs_covXY_OPnorms = [], []

        # Compute correlation score between z_t and z_t+idt   | i in [1, ..., n_frames]
        # We denote as X, and Y as the data matrices, containing observations of time-step 0 and consecutive time-steps
        X_uncentered = obs[:, 0, :]
        X, covXX, covXX_eigvals, covXX_Fnorm, covXX_OPnorm, covXX_reg = self.compute_statistical_metrics(X_uncentered)

        # obs_independence_scores.append(covXX_reg)
        # obs_covZZ_Fnorms.append(covXX_Fnorm)
        # obs_covZZ_OPnorms.append(covXX_OPnorm)
        # obs_covZZ_sval_min.append(covXX_eigvals[0])
        # if i == 1:
        #     X_uncentered = obs[:, 0, :]
        #     X, covXX, covXX_eigvals, covXX_Fnorm, covXX_OPnorm, covXX_reg = self.compute_statistical_metrics(
        #         X_uncentered)
        # else:
        #     X, covXX, covXX_eigvals, covXX_Fnorm, covXX_OPnorm, covXX_reg = Y, covYY, covYY_eigvals, covYY_Fnorm, covYY_OPnorm, covYY_reg

        for i in range(1, n_frames):
            Y_uncentered = obs[:, i, :]
            Y, covYY, covYY_eigvals, covYY_Fnorm, covYY_OPnorm, covYY_reg = self.compute_statistical_metrics(Y_uncentered)

            # Obtain empirical estimate of Cross-covariance matrix
            # CovXY = X^H·Y   | a^H : Hermitian (conjugate transpose) of a
            # Here X is expected to have shape (M, dim(z_t)) | M = batch_size ->  shape(CovXX) := (M, dim(z_t), dim(z_t)
            covXY = torch.matmul((X.conj()[:, :, None]), Y[:, None, :])
            covXY = torch.mean(covXY, dim=0)  # Average over all samples (in batch)
            # Cross-Covariance matrix is not necessarily Hermitian. A simple way to compute the Frobenious norm is by
            # (1) ||A||_F = sqrt(trace(A^H·A)) | C^H : Hermitian (conjugate transpose) of C
            # Considering the SVD of A=U·Σ·V^H, and of A^H·A=V·Σ^H·Σ·V^H = V·|Σ|^2·V^H, this is also equivalent to
            # (2) ||A||_F = sqrt(sum<σ_i,σ_i>^2) = sqrt(sum(|σ_i|^2)) | σ_i singular vals of A, and |c| the modulus of c
            # TODO: check whats the fastest way to compute the norm. For now we use svdvals to get min singular value
            covXY_svals = torch.linalg.svdvals(covXY)  # + (1e-6 * torch.eye(covXY.shape[0], device=covXY.device)))
            covXY_Fnorm = torch.norm(covXY_svals, p=2)
            covXY_max_singval, covXY_min_singval = covXY_svals[0], covXY_svals[-1]
            assert torch.isclose(covXY_Fnorm, torch.linalg.matrix_norm(covXY, ord='fro'))
            covXY_OPnorm = covXY_max_singval

            # Compute the generalized Rayleigh quotient
            # r = ||CovXY||_F^2 / ||CovXX||_op · ||CovYY||_op   # Add small reg term for numerical stability
            gen_rayleigh = covXY_Fnorm**2 / (covXX_OPnorm * covYY_OPnorm)
            # Add regularization term encouraging orthogonality of observation dimensions.
            reg_gen_rayleigh = gen_rayleigh - (self.reg_lambda * covYY_reg)

            # Save each time-frame scores for metrics and loss computation.
            gen_rayleigh_quotients.append(gen_rayleigh)
            reg_gen_rayleigh_quotients.append(reg_gen_rayleigh)
            obs_independence_scores.append(covXX_reg)
            obs_covZZ_Fnorms.append(covXX_Fnorm)
            obs_covZZ_OPnorms.append(covXX_OPnorm)
            obs_covZZ_sval_min.append(covXX_eigvals[0])
            obs_covXY_OPnorms.append(covXY_OPnorm)
            obs_covXY_sval_min.append(covXY_min_singval)

        # Store in metrics the individual generalized Rayleigh quotients, to see difference between time gaps.
        metrics = {}

        avg_rayleigh_score = torch.mean(torch.stack(gen_rayleigh_quotients))
        avg_reg_rayleigh_score = torch.mean(torch.stack(reg_gen_rayleigh_quotients))
        # Store relevant metrics.
        metrics['avg_obs_dependence'] = torch.mean(torch.stack(obs_independence_scores))
        metrics['avg_CovZZ_Fnorm'] = torch.mean(torch.stack(obs_covZZ_Fnorms))
        metrics['avg_CovZZ_OPnorm'] = torch.mean(torch.stack(obs_covZZ_OPnorms))
        metrics['avg_CovZZ_sval_min'] = torch.mean(torch.stack(obs_covZZ_sval_min))
        metrics['avg_gen_rayleigh'] = avg_rayleigh_score
        metrics['avg_reg_gen_rayleigh'] = avg_reg_rayleigh_score
        metrics['CovXY_OPnorm'] = torch.mean(torch.stack(obs_covXY_OPnorms))
        metrics['CovXY_sval_min'] = torch.mean(torch.stack(obs_covXY_sval_min))
        # We assume the optimizer is set (as default) to minimize the loss. Thus, we return the negative of the score.
        return -avg_reg_rayleigh_score, metrics

    @staticmethod
    def compute_statistical_metrics(Z):
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

    @property
    def updated_eigenmatrix(self):
        return self._is_koopman_operator_ready

    def _set_updated_eigenmatrix(self, val: bool):
        self._is_koopman_operator_ready = val

    @staticmethod
    def _backward_hook(module, grad_input, grad_output) -> None:
        """
        We use the backward hook to identify when the observation function (the NN generating the obs) gets
        parameter updates. This implies the space of observations changes and thus the Koopman Operator on this space
        needs to be recomputed.
        :param module:
        :return:
        """
        if isinstance(module, VAMP):
            module._set_updated_eigenmatrix(False)

    def get_metric_labels(self) -> Iterable[str]:
        return ['avg_gen_rayleigh', 'avg_obs_dependence', 'obs_dyn_error']

    def batch_unpack(self, batch):
        return self.state_ctrl_to_x(batch)

    def state_ctrl_to_x(self, batch):
        """
        Mapping from batch of ClosedLoopDynamics data points to NN model input-output data points
        """
        inputs = torch.concatenate([batch[STATES], batch[CTRLS]], dim=2)
        return inputs

    def batch_pack(self, x):
        return x

    def get_hparams(self):
        return {'encoder': self.encoder.get_hparams(),
                'pred_horizon': self.pred_horizon}
