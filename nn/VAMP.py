from typing import Iterable

import torch

from data.ClosedLoopDynamics import STATES, CTRLS
from src.RobotEquivariantNN.nn.EMLP import MLP


class VAMP(torch.nn.Module):

    def __init__(self, state_dim: int, obs_dim: int, pred_horizon: int = 1, reg_lambda=1e-2, n_layers=3,
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
        assert reg_lambda > 0, "Regularization coefficient must be positive"
        self.reg_lambda = reg_lambda
        n_head_layers = n_layers if n_head_layers is None else n_head_layers
        output_dim = obs_dim * 2  # Complex observations of dim (..., 2) (real and imaginary parts)

        # We have a core encoder/feature-extraction-module processing all time-steps of the trajectory
        # and then we have `m=look_ahead` heads that output the observations of each time-step separately
        # Ideally sharing the same backbone will make this module efficient at extracting the features required to
        # create the observations at each time-step.
        # Input to encoder is expected to be [batch_size, time_steps, state_dim]
        self.encoder = MLP(d_in=state_dim, d_out=output_dim, ch=n_hidden_neurons, n_layers=n_layers,
                           activation=[activation] * n_layers)

        heads = []
        for head_id in range(pred_horizon + 1):
            # Each head is constructed as squared perceptrons, one with non-linearity and the last without.
            # This is a convenient convention. Nothing special behind it.
            heads.append(MLP(d_in=output_dim, d_out=output_dim, ch=n_hidden_neurons, n_layers=n_head_layers,
                             activation=[activation] * (n_head_layers - 1) + [torch.nn.Identity]))

        self.heads = torch.nn.ModuleList(heads)

    def forward(self, x):
        # Backbone operation
        encoder_out = self.encoder(x)

        observations = []
        for i, head in enumerate(self.heads):
            # Each head computes observation functions from each timestep [t+i*dt]
            head_out = head(encoder_out[:, i, :])
            # output is composed of:
            # (re(Ψi_1(x_t+i*dt), img(Ψi_1(x_t+i*dt),..., re(Ψi_N(x_t+i*dt)), img(Ψi_N(x_t+i*dt))),
            # Convert it to a complex tensor
            # With reshape consecutive scalars are assumed to be real and imaginary parts of a complex number
            psi = torch.reshape(head_out, shape=(head_out.shape[:-1] + (head_out.shape[-1] // 2, 2)))
            z_t_idt = torch.view_as_complex(psi)
            observations.append(z_t_idt)

        return observations

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
        assert isinstance(obs, list), "We expect a list of observations of shape (batch_size, obs_dim)"
        n_frames = len(obs)
        assert n_frames >= 2, "We expect at least 2 observations to compute correlation score"

        gen_rayleigh_quotients = []       # Rayleigh quotient for each time-step
        reg_gen_rayleigh_quotients = []   # Regularized Rayleigh quotient for each time-step
        obs_independence_scores = []              # Regularization terms indicating orthogonality of observations
        obs_cov_FNorms = []

        # Compute correlation score between z_t and z_t+idt   | i in [1, ..., n_frames]
        X_uncentered = obs[0]
        X, covXX, covXX_Fnorm, covXX_reg = self.compute_statistical_metrics(X_uncentered)
        obs_independence_scores.append(covXX_reg)
        obs_cov_FNorms.append(covXX_Fnorm)

        for i in range(1, n_frames):
            Y_uncentered = obs[i]
            Y, covYY, covYY_Fnorm, covYY_reg = self.compute_statistical_metrics(Y_uncentered)

            # Obtain empirical estimates of covariance matrices
            # CovXX = X^H·X   | a^H : Hermitian (conjugate transpose) of a
            # Here X is expected to have shape (M, dim(z_t)) | M = batch_size ->  shape(CovXX) := (M, dim(z_t), dim(z_t)
            covXY = torch.matmul((X.conj()[:, :, None]), Y[:, None, :])
            covXY = torch.mean(covXY, dim=0)  # Average over all samples (in batch)
            covXY_Fnorm = torch.norm(covXY, p='fro')
            # Compute the generalized Rayleigh quotient
            # r = Σ eig(CovXY)^2 / sqrt(Σ eig(CovXX)·Σ eig(CovYY)) | eig(X) = {λ_i | X·v_i = λ_i·v_i} (with multiplicty)
            #   = ||CovXY||_F^2 / sqrt(||CovXX||_F · ||CovYY||_F)   # Add small reg term to avoid division by 0
            gen_rayleigh = covXY_Fnorm**2 / (torch.sqrt(covXX_Fnorm * covYY_Fnorm) + 1e-6)
            # Add regularization term encouraging orthogonality of observation dimensions.
            reg_gen_rayleigh = gen_rayleigh - (self.reg_lambda * (covXX_reg + covYY_reg))

            # Save each time-frame scores for metrics and loss computation .
            gen_rayleigh_quotients.append(gen_rayleigh)
            reg_gen_rayleigh_quotients.append(reg_gen_rayleigh)
            obs_independence_scores.append(covYY_reg)
            obs_cov_FNorms.append(covYY_Fnorm)

        # Store in metrics the individual generalized Rayleigh quotients, to see difference between time gaps.
        metrics = {}
        # for head, (independence_score, covFnorm) in enumerate(zip(obs_independence_scores, obs_cov_FNorms)):
        #     metrics[f'obs{head}/independence'] = independence_score
        #     metrics[f'obs{head}/covFnorm'] = covFnorm
        # for i, (gen_rayleigh, reg_gen_rayleigh) in enumerate(zip(gen_rayleigh_quotients, reg_gen_rayleigh_quotients)):
            # metrics[f'gen_rayleigh_{i+1}'] = gen_rayleigh
            # metrics[f'reg_gen_rayleigh_{i+1}'] = reg_gen_rayleigh
            # pass

        avg_rayleigh_score = torch.mean(torch.stack(gen_rayleigh_quotients))
        avg_reg_rayleigh_score = torch.mean(torch.stack(reg_gen_rayleigh_quotients))
        # Store relevant metrics.
        metrics['avg_obs_independence'] = torch.mean(torch.stack(obs_independence_scores))
        metrics['avg_obs_Var_Fnorm'] = torch.mean(torch.stack(obs_cov_FNorms))
        metrics['avg_gen_rayleigh'] = avg_rayleigh_score
        metrics['avg_reg_gen_rayleigh'] = avg_reg_rayleigh_score
        # We assume the optimizer is set (as default) to minimize the loss. Thus, we return the negative of the score.
        return -avg_reg_rayleigh_score, metrics

    @staticmethod
    def compute_statistical_metrics(Z_uncentered):
        """
        For a random variable Z this functions makes the following process: Center the samples, compute the covariance
        matrix `covZZ`, compute the Frobenius norm of the covariance matrix `covZZ_Fnorm`, and a metric indicating the
        independence/orthogonality of the dimensions of Z (i.e., the norm of the difference
        between the covariance matrix and the identity matrix, divided by dimension of the obs, so a value of 1 will
        indicate complete independece/orthogonality of dimensions).
        TODO: If samples are drawn from a symmetric random process we have that
          Expected value of variables is invariant to symmetry transformations. Also, The covariance matrix is
          a linear operator that commutes with the group actions (same as Koopman operator) thus, in a
          "symmetry enabled" basis (exposing isotypic components) the covariance matrix is expected to
          be block-diagonal. We can exploit that known structure to reduce empirical estimation errors a sort of
          structural regularizaiton, one would say .
        TODO: There is difference in numerical error between numpy and torch processing
          complex numbers. Despite having both double precision. Documentation shows a warning sign indicating for
          complex operations, saying we should use Cuda 11.6 (not sure how impactfull this is). We should follow.
        :param Z_uncentered: Batched samples of the random variable Z, shape (M, dim(z)) | M = batch_size
        :return:
            - Z_centered: (torch.Tensor) Centered samples of Z (i.e., Z_uncentered - mean(Z_uncentered))
            - covZZ: (torch.Tensor) Covariance matrix of Z
            - covZZ_Fnorm: (float) Frobenius norm of the covariance matrix of Z
            - Z_dependence: (float) Metric measuring orthogonality/independence of dimensions of Z.
                0 means complete independece of dimensions.
        """
        assert Z_uncentered.dim() == 2, "We expect a batch of samples of shape (M, dim(z)) | M = batch_size"
        Z_mean = torch.mean(Z_uncentered, dim=0, keepdim=True)
        Z_centered = Z_uncentered - Z_mean
        # Compute covariance per sample in batch
        covZZ = torch.matmul((Z_centered.conj()[:, :, None]), Z_centered[:, None, :])
        # Average over the M samples in batch (i.e. shape(covZZ) = (M, dim(z), dim(z))
        covZZ = torch.mean(covZZ, dim=0)
        # Compute Frobenius norm of the covariance matrix
        covZZ_Fnorm = torch.norm(covZZ, p='fro')
        # Compute regularization term encouraging for orthogonality/independence of dimensions of obs vector
        dimZ = covZZ.shape[0]
        orthogonality_error = covZZ - torch.eye(covZZ.shape[0], device=covZZ.device)
        # Scale the regularization term with 1/dim**2 to obtain some invariance to the dimension of obs vector.
        Z_dependence = torch.norm(orthogonality_error, p='fro')**2  #/ (dimZ ** 2)
        return Z_centered, covZZ, covZZ_Fnorm, Z_dependence

    def get_metric_labels(self) -> Iterable[str]:
        return []

    def batch_unpack(self, batch):
        return self.state_ctrl_to_x(batch)

    def state_ctrl_to_x(self, batch):
        """
        Mapping from batch of ClosedLoopDynamics data points to NN model input-output data points
        """
        inputs = torch.concatenate([batch[STATES], batch[CTRLS]], dim=2)
        return inputs

    def batch_pack(self, x):
        return self.x_to_state_crtl(x)

    def get_hparams(self):
        return {'encoder': self.encoder.get_hparams(),
                'pred_horizon': self.pred_horizon}
