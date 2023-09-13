import logging
from functools import reduce
from typing import Optional

import torch
from escnn.group import Representation
from torch import Tensor

log = logging.getLogger(__name__)


def compute_projection_score(cov_x, cov_y, cov_xy):
    """ Computes the projection score of the covariance matrices. Maximizing this score is equivalent to maximizing
    the correlation between x and y

    The projection score is defined as:
        P := ( ||cov_x_inv @ cov_xy @ cov_y_inv||_HS )^2
    Args:
        cov_x: (time, features, features) or (features, features)
        cov_y: (time, features, features) or (features, features)
        cov_xy: (time, features, features) or (features, features)

    Returns:
        Score value: (time, 1) or (1,) depending on the input shape.
    """
    score = torch.linalg.lstsq(cov_x, cov_xy).solution  # cov_x_inv @ cov_xy
    score = score @ torch.linalg.pinv(cov_y, hermitian=True)  # cov_x_inv @ cov_xy @ cov_y_inv
    score = torch.linalg.matrix_norm(score, ord='fro') ** 2  # ||cov_x_inv @ cov_xy @ cov_y_inv||_HS^2
    return score


def compute_spectral_score(cov_x, cov_y, cov_xy):
    """ Computes the spectral score of the covariance matrices. This is a looser bound on the projection score, but a
    more numerically stable metric, since it avoids computing the inverse of the covariance matrices.

    The spectral score is defined as:
        S := (||cov_xy||_HS)^2 / (||cov_x||_2 / ||cov_y||_2 )
    Args:
        cov_x: (time, features, features) or (features, features)
        cov_y: (time, features, features) or (features, features)
        cov_xy: (time, features, features) or (features, features)

    Returns:
        Score value: (time, 1) or (1,) depending on the input shape.
    """
    score = torch.linalg.matrix_norm(cov_xy, ord='fro') ** 2  # == ||cov_xy|| 2, HS
    score = score / torch.linalg.matrix_norm(cov_x, ord=2, dim=(-2, -1))  # ||cov_xy|| 2, HS / ||cov_x||
    score = score / torch.linalg.matrix_norm(cov_y, ord=2, dim=(-2, -1))  # ||cov_xy|| 2, HS / (||cov_x|| * ||cov_y||)
    return score


def regularization_orthonormality(cov_x):
    """ Regularization term to enforce orthonormality of the feature/ space
    Args:
        cov_x: (time, features, features) or (features, features)
    Returns:
        regularization term: (time, 1) or (1,) depending on the input shape.
    """
    assert cov_x.shape[-1] == cov_x.shape[-2], f"Covariance matrix is assumed to be in the last two dimensions"
    identity = torch.eye(cov_x.shape[-1], device=cov_x.device)
    reg = torch.linalg.matrix_norm(identity - cov_x, ord='fro', dim=(-2, -1))
    return reg


def regularization_2(cov_x, cov_y, rank):
    r1 = rank + torch.trace(cov_x @ torch.log(cov_x.abs()) - cov_x)
    r2 = rank + torch.trace(cov_y @ torch.log(cov_y.abs()) - cov_y)
    return (r1 + r2).mean()


def covariance(X: Tensor, Y: Tensor):
    """ Computes the batched covariance of two tensors.
    Args:
        X: (batch, time, features)
        Y: (batch, time, features)
    Returns:
        Covariance Matrix of shape (features, features)
    """
    assert X.shape == Y.shape and len(X.shape) == 3, f"(batch, time, features) != {X.shape} / {Y.shape}"
    n_samples = X.shape[0]
    # [t=time, b=batch, o=state_dim] x [t=time, a=state_dim, b=batch] -> [t=time, o=state_dim, a=state_dim]
    #                                                   [b, t, o] -> [t, o, b]            [b, t, o] -> [t, o, b]
    CovXY = torch.einsum('...ob,...ba->...oa', torch.permute(X, dims=(1, 2, 0)), torch.permute(Y, dims=(1, 0, 2)))
    return CovXY / n_samples


def empirical_cov_cross_cov(state_traj: Tensor,
                            state_traj_prime: Optional[Tensor] = None,
                            representation: Optional[Representation] = None,
                            cov_window_size: Optional[int] = None,
                            debug: bool = False) -> (Tensor, Tensor):
    """ Compute empirical approximation of the covariance and cross-covariance operators for a trajectory of states.
    This function computes the empirical approximation of the covariance and cross-covariance operators in batched
    matrix operations for efficiency.
    Args:
        state_traj: (batch, time_horizon, state_dim) trajectory of states in main function space.
        state_traj_prime: (batch, time_horizon, state_dim) trajectory of states in auxiliary function space.
        representation (optional Representation): Group representation on the state space. If provided, the empirical
         covariance and cross-covariance operators will be improved using the group average trick:
         Cov(0,i) = 1/|G| Σ_g ∈ G (ρ(g) Cov(x_0, x'_i) ρ(g)^-1), ensuring that the empirical operators are equivariant:
         Cov(0,i) ρ(g) = ρ(g) Cov(0,i) ∀ g ∈ G.
        debug: (bool) If True, check that the empirical operators are equivariant. Defaults to False.
    Returns:
        CCov: (time_horizon, time_horizon, state_dim, state_dim) Tensor containing all the Cross-Covariance
         empirical operators between the states in the main trajectory and states in the auxiliary trajectory.
         Each entry of the tensor is a (state_dim, state_dim) covariance estimate.
         Such that CCov(i,j) = Cov(x_i, x'_j) ∀ i, j in [0, time_horizon], j >= i.
        Cov: (time_horizon, state_dim, state_dim) Tensor containing all the Covariance empirical operators between time
         steps on the main state space. Cov(i) = Cov(x_i, x_i) ∀ i in [0, time_horizon]
        Cov_prime: (time_horizon, state_dim, state_dim) Tensor containing all the Covariance empirical operators between
         time steps on the auxiliary state space. Cov_prime(i) = Cov(x'_i, x'_i) ∀ i in [0, time_horizon]
    """
    assert len(state_traj.shape) == 3, f"state_traj: {state_traj.shape}. Expected (batch, time_horizon, state_dim)"
    assert state_traj_prime is None or state_traj_prime.shape == state_traj.shape
    assert cov_window_size is None or isinstance(cov_window_size, int)
    num_samples, time_horizon, state_dim = state_traj.shape
    dtype, device = state_traj.dtype, state_traj.device
    pred_horizon = time_horizon - 1
    unique_function_space = state_traj_prime is None

    if unique_function_space:  # If function space has a transfer-invariant density, no auxiliary function.
        state_traj_prime = state_traj
    # Parallel computation. Serial faster as it avoids copying the tensor many times.
    # Expand state_traj to have an extra dimension for time_horizon
    state_traj_block = state_traj.permute(1, 0, 2)  # (T, batch_size, state_dim)
    state_traj_block = state_traj_block.unsqueeze(1).expand(-1, time_horizon, -1, -1)  # (T, T', batch_size, state_dim)
    state_traj_block_prime = state_traj_prime.permute(1, 0, 2)  # (T, batch_size, state_dim)
    state_traj_block_prime = state_traj_block_prime.unsqueeze(1).expand(-1, time_horizon, -1, -1)

    # Compute in a single tensor (parallel) operation all cross-covariance between time steps in original and auxiliary
    # state trajectories.
    # Cov[i, j] := Cov(x_i, x'_j)
    CCov = torch.einsum('...ob,...ba->...oa',  # -> (T, T', state_dim, state_dim)
                        state_traj_block.permute(0, 1, 3, 2),  # (T, T', state_dim, batch_size)
                        state_traj_block_prime.permute(1, 0, 2, 3)  # (T', T, batch_size, state_dim)
                        ) / num_samples

    if unique_function_space:  # If same funtion space Cov(t,t) is the diagonal of the cross-Cov matrix.
        Cov = CCov[range(time_horizon), range(time_horizon)]
        Cov_prime = Cov
    else:  # If diff function spaces, we need to compute the Covariance matrices for each state space.
        #      (T, state_dim, batch_size) @ (T, batch_size, state_dim) -> (T, state_dim, state_dim)
        Cov = state_traj.permute(1, 2, 0) @ state_traj.permute(1, 0, 2) / num_samples
        Cov_prime = state_traj_prime.permute(1, 2, 0) @ state_traj_prime.permute(1, 0, 2) / num_samples

    if debug:  # Sanity checks. These should be true by construction
        for t in range(min(pred_horizon, cov_window_size)):
            assert torch.allclose(CCov[0, t], state_traj[:, 0, :].T @ state_traj_prime[:, t, :] / num_samples), \
                f"Max error {torch.max(torch.abs(CCov[0, t] - state_traj[:, 0, :].T @ state_traj_prime[:, t, :]))}"
            assert torch.allclose(Cov[t], state_traj[:, t, :].T @ state_traj[:, t, :] / num_samples), \
                f"Max error {torch.max(torch.abs(Cov[t] - state_traj[:, t, :].T @ state_traj[:, t, :]))}"
            assert torch.allclose(Cov_prime[t], state_traj_prime[:, t, :].T @ state_traj_prime[:, t, :] / num_samples)

    if representation is not None:
        # We can improve the empirical estimates by understanding that the theoretical operators are equivariant:
        # ρ(g) Cov(X,Y) ρ(g)^T = Cov(ρ(g)X, ρ(g)Y) = Cov(X, Y) = CovXY
        # Thus we can apply the "group-average" trick to improve the estimate:
        # CovXY = 1/|G| Σ_g ∈ G (ρ(g) Cov(X,Y) ρ(g)^T) (see https://arxiv.org/abs/1111.7061)
        # This is a costly operation but is equivalent to doing data augmentation of the state space samples,
        # with all group elements, and then computing the empirical covariance operators.
        # Furthermore, we can apply this operation in parallel to all Cov Ops for numerical efficiency in GPU.
        orbit_cross_Cov = [CCov]
        orbit_Cov, orbit_Cov_prime = [Cov], [Cov_prime]
        for h in representation.group.generators:  # Generators of the symmetry group. We only need these.
            # Compute each:      ρ(g) Cov(X,Y) ρ(g)^T   | ρ(g)^T = ρ(~g) = ρ(g^-1)
            rep_g = torch.tensor(representation(h), dtype=dtype, device=device)
            rep_g_inv = torch.tensor(representation(~h), dtype=dtype, device=device)
            #                                        t,l=time, n,m,a,o=state_dim
            orbit_cross_Cov.append(torch.einsum('na,ltao,om->ltnm', rep_g, CCov, rep_g_inv))
            orbit_Cov.append(torch.einsum('na,tao,om->tnm', rep_g, Cov, rep_g_inv))
            orbit_Cov_prime.append(torch.einsum('na,tao,om->tnm', rep_g, Cov_prime, rep_g_inv))

        # Compute group average:  1/|G| Σ_g ∈ G (ρ(g) Cov(X,Y) ρ(g)^T).
        CCov = torch.mean(torch.stack(orbit_cross_Cov, dim=0), dim=0)
        Cov = torch.mean(torch.stack(orbit_Cov, dim=0), dim=0)
        Cov_prime = torch.mean(torch.stack(orbit_Cov_prime, dim=0), dim=0)

        if debug:  # Check commutativity/equivariance of the empirical estimates of all Covariance operators
            for g in representation.group.elements:
                rep_h = torch.tensor(representation(g), dtype=dtype, device=device)
                cov_rep = torch.einsum('na,ltao->ltno', rep_h, CCov)  # t,l=time, n,m,a,o=state_dim
                rep_cov = torch.einsum('ltao,om->ltam', CCov, rep_h)
                window = min(pred_horizon, cov_window_size)
                assert torch.allclose(cov_rep[0, :window], rep_cov[0, :window], atol=1e-5), \
                    f"Max equivariance error {torch.max(torch.abs(cov_rep[0, :] - rep_cov[0, :]))}"
                # Check now commutativity of Cov and Cov_prime
                cov_rep = torch.einsum('na,tao->tno', rep_h, Cov)  # t=time, n,m,a,o=state_dim
                rep_cov = torch.einsum('tao,om->tam', Cov, rep_h)
                assert torch.allclose(cov_rep, rep_cov, atol=1e-5)
                cov_rep = torch.einsum('na,tao->tno', rep_h, Cov_prime)  # t=time, n,m,a,o=state_dim
                rep_cov = torch.einsum('tao,om->tam', Cov_prime, rep_h)
                assert torch.allclose(cov_rep, rep_cov, atol=1e-5)

    return CCov, Cov, Cov_prime


def chapman_kolmogorov_regularization(CCov: Tensor, ck_window_length: int = 3, debug: bool = False):
    """ Compute the Chapman-Kolmogorov regularization using the cross-covariance operators between distinct time steps.

    This regularization aims at exploitation Markov Assumption of a linear dynamical system. Specifically it computes:
    ||Cov(t, t+d) - Cov(t, t+1) Cov(t+1, t+2) ... Cov(t+d-1, t+d) || ∀ t in [0, pred_horizon-2],
    d in [2, min(pred_horizon, ck_window_length)].

    The function thus computes all possible regularization terms of the provided initial state and state trajectory,
    withing the provided window length `ck_window_length`.

    TODO: Vectorize this function to do operations in parallel.
    See more in:
    [1] Vladimir Kostic, Pietro Novelli, Riccardo Grazzi, Karim Lounici, and Massimiliano Pontil.
    “Deep Projection Networks for Learning Time-Homogeneous Dynamical Systems.” arXiv, July 19,
    2023. https://doi.org/10.48550/arXiv.2307.09912.

    Args:
        CCov: (time_horizon, time_horizon, state_dim, state_dim) Tensor containing all the Cross-Covariance
         empirical operators between the states in the main trajectory and states in the auxiliary trajectory.
         Each entry of the tensor is a (state_dim, state_dim) covariance estimate.
         Such that CCov(i,j) = Cov(x_i, x'_j) ∀ i, j in [0, time_horizon], j >= i.
        ck_window_length: (int) Maximum window length to compute the regularization term. Defaults to 2.
        debug: (bool) Whether to print debug information on the CK scores computed. Defaults to False.
    Returns:
        ck_error: (time_horizon - 1,) Average CK error per `dt` time steps. That is:
         ck_error[dt - 2] = avg(|| Cov(t, t+dt) - Cov(t, t+1) Cov(t+1, t+2) ... Cov(t+dt-1, t+dt) ||) |
            ∀ t in [0, time_horizon - 2], dt in [2, min(time_horizon - 2, ck_window_length)]
    """
    assert (len(CCov.shape) == 4 and CCov.shape[0] == CCov.shape[1]
            and CCov.shape[2] == CCov.shape[3]), f"Expected Cov_t_dt of shape (T, T, state_dim, state_dim)"
    time_horizon = CCov.shape[0]
    dtype, device = CCov.dtype, CCov.device

    # Generate upper triangular matrix that will contain the CK error values at each position, such that:
    # ck_errors[i, j] = || Cov(i, j) - Cov(i, i+1) Cov(i+1, i+2) ... Cov(j-1, j) || | j >= i+2
    ck_reg = torch.fill(torch.zeros((time_horizon, time_horizon), dtype=dtype, device=device), torch.nan)

    # Minimum number of steps to compute the CK regularization term
    # ck_errors[t, t+2] = || Cov(t, t+2) - Cov(t, t+1) Cov(t+1, t+2) ||
    min_dt = 2
    for ts in range(0, time_horizon - 2):  # ts ∈ [0, time_horizon - 2]
        chain_cov = None
        chain_test = []  # te ∈ [ts + 2, min(pred_horizon, ck_window)]
        max_dt = min(ck_window_length + 1, time_horizon - ts)
        for dt in range(min_dt, max_dt):
            te = ts + dt  # te ∈ [ts + 2, min(pred_horizon, ts + ck_window)]
            # Compute the covariance chain using Dynamic Programming (i.e., do not repeat computations)
            # Cov(ts, ts+1), Cov(ts+1, ts+2), ... Cov(te-1, te)
            if chain_cov is None:  # chain_cov = Cov(ts, ts+1), Cov(ts+1, ts+2)
                chain_cov = CCov[ts, ts + 1] @ CCov[ts + 1, ts + 2]
                chain_test.extend([(ts, ts + 1), (ts + 1, ts + 2)])
            else:  # chain_cov *= Cov(te-1, te)
                chain_cov = chain_cov @ CCov[te - 1, te]
                chain_test.append((te - 1, te))
            # Select the target covariance as the operator in the upper right if the [ts : te, ts : te] block.
            target_cov = CCov[ts, te]  # Cov(ts, te)

            # || Cov(ts, te) - Cov(ts, ts+1), Cov(ts+1, ts+2), ... Cov(te-1, te) ||_2
            ck_reg[ts, te] = torch.linalg.matrix_norm(chain_cov - target_cov, ord='fro')

    # a = ck_reg.detach().cpu().numpy()
    if debug:
        # Test largest chain scenario
        target_cov = CCov[0, ck_window_length]
        chain_cov = reduce(torch.matmul, [CCov[0 + i, 1 + i] for i in range(ck_window_length)])
        ck_reg_true = torch.linalg.matrix_norm(chain_cov - target_cov, ord='fro')
        assert torch.allclose(ck_reg_true, ck_reg[0, ck_window_length]), \
            f"max ck error {torch.max(torch.abs(ck_reg_true - ck_reg[0, 3]))}"

    ck_avg_error = []
    max_dt = min(ck_window_length + 1, time_horizon)
    for dt in range(min_dt, max_dt):
        ck_avg_error.append(torch.mean(torch.diagonal(ck_reg, offset=dt)))

    return Tensor(ck_avg_error)


def compute_chain_spectral_corr_scores(CCov: Tensor, Cov: Tensor, Cov_prime: Tensor,
                                       window_size: Optional[int] = None, debug: bool = False):
    """ Compute the spectral and correlation scores using the cross-covariance operators between distinct time steps.

    Args:
        CCov: (time_horizon, time_horizon, state_dim, state_dim) Tensor containing all the Cross-Covariance
         empirical operators between the states in the main trajectory and states in the auxiliary trajectory.
         Each entry of the tensor is a (state_dim, state_dim) covariance estimate.
         Such that CCov(i,j) = Cov(x_i, x'_j) ∀ i, j in [0, time_horizon], j >= i.
        Cov: (time_horizon, state_dim, state_dim) Tensor containing all the Covariance empirical operators between time
         steps on the main state space. Cov(i) = Cov(x_i, x_i) ∀ i in [0, time_horizon]
        Cov_prime: (time_horizon, state_dim, state_dim) Tensor containing all the Covariance empirical operators between
         time steps on the auxiliary state space. Cov_prime(i) = Cov(x'_i, x'_i) ∀ i in [0, time_horizon]
        window_size: (int) Maximum window length to compute the spectral score. Defaults to None, in which case the
         score is computed for all window_size > j >= i
        debug: (bool) Whether to print debug information on the scores computed. Defaults to False.
    Returns:
        spectral_scores: (time_horizon - 1) Tensor containing the average spectral score between time steps separated
         apart by a shift of `dt` [steps/time]. That is:
            spectral_score[dt - 1] = avg(||Cov(x_i, x'_i+dt)||_HS^2/(||Cov(x_i, x_i)||_2*||Cov(x'_i+dt, x'_i+dt)||_2))
             | ∀ i in [0, time_horizon - dt], dt in [1, min(time_horizon - i, window_size)]
        corr_scores: (time_horizon - 1) Tensor containing the correlation scores between time steps separated
         apart by a shift of `dt` [steps/time]. That is:
            corr_score[dt - 1] = avg(||Cov(x_i, x_i)^-1 Cov(x_i, x'_i+dt) Cov(x'_i+dt, x'_i+dt)^-1||_HS^2)
             | ∀ i in [0, time_horizon - dt], dt in [1, min(time_horizon - i, window_size)]
    """
    assert (len(CCov.shape) == 4 and CCov.shape[0] == CCov.shape[1]
            and CCov.shape[2] == CCov.shape[3]), f"Expected Cov_t_dt of shape (T, T, state_dim, state_dim)"
    assert len(Cov.shape) == 3 and Cov.shape[0] == Cov.shape[1], f"Expected Cov of shape (T, state_dim, state_dim)"
    assert Cov.shape == Cov_prime.shape, f"Expected Cov_prime of shape (T, state_dim, state_dim)"

    time_horizon = CCov.shape[0]

    # Compute the norm of the diagonal of the covariance matrices is a single parallel operation.
    norm_Cov = torch.linalg.matrix_norm(Cov, ord=2, dim=(-2, -1))  # norm_Cov_t[i] = ||Cov(x_i, x_i)||_2
    norm_Cov_prime = torch.linalg.matrix_norm(Cov_prime, ord=2, dim=(-2, -1))  # norm_Cov_t[i] = ||Cov(x'_i, x'_i)||_2

    # Compute the HS norm of the Cross-Covariance operators in a single parallel operation.
    # norm_CCov[i, j] = ||Cov(x_i, x'_j)||_HS
    norm_CCov = torch.linalg.matrix_norm(CCov, ord='fro', dim=(-2, -1))

    # Compute all the Spectral scores between time steps separated by `dt`
    # spectral_score[dt - 1] = ||Cov(x_i, x'_i+dt)||_HS^2 / (||Cov(x_i, x_i)||_2 * ||Cov(x'_i+dt, x'_i+dt)||_2) |
    #                                        i in [0, time_horizon - dt], dt in [1, min(time_horizon - i, window_size)]
    # This operation can be performed in vectorized form for every dt. Note that the cross covariance operators
    # between all time steps separated by a dt are the Cov(x_i, x'_i+dt) in the `dt` diagonal of the CCov matrix.
    # max_dt = min(time_horizon, window_size + 1)
    max_dt = time_horizon
    spectral_scores, corr_scores = [], []
    for dt in range(1, max_dt):
        cov_X_idx = range(time_horizon - dt)
        cov_Y_idx = range(dt, time_horizon)
        norm_CovX, norm_CovY, = norm_Cov[cov_X_idx], norm_Cov_prime[cov_Y_idx]
        norm_CovXY = norm_CCov[cov_X_idx, cov_Y_idx]  # Get elements of the `dt` diagonal of the CCov matrix
        spectral_scores_dt = norm_CovXY ** 2 / (norm_CovX * norm_CovY)
        assert len(
            spectral_scores_dt) == time_horizon - dt, (f"Expected {time_horizon - dt} scores, "
                                                       f"got {len(spectral_scores_dt)}")
        spectral_scores.append(spectral_scores_dt)
        # Compute Correlation score
        CovX, CovY = Cov[cov_X_idx], Cov_prime[cov_Y_idx]
        CovXY = CCov[cov_X_idx, cov_Y_idx]
        p_score = compute_projection_score(cov_x=CovX, cov_y=CovY, cov_xy=CovXY)
        corr_scores.append(p_score)

    if debug:  # Check vectorized operations are equivalent to sequential operations
        for dt in range(1, max_dt):
            s_scores_dt = spectral_scores[dt - 1]  # ||Cov(t, t+dt)||_HS^2 / (||Cov(t, t)||_2 * ||Cov(t+dt,t+dt)||_2)
            p_scores_dt = corr_scores[dt - 1]
            for t in range(time_horizon - dt):
                covX, covY = Cov[t], Cov_prime[t + dt]
                covXY = CCov[t, t + dt]
                exp = s_scores_dt[t]
                real = compute_spectral_score(cov_x=covX, cov_y=covY, cov_xy=covXY)
                assert torch.allclose(exp, real, atol=1e-5), f"Spectral scores do not match {exp}!={real}"
                exp = p_scores_dt[t]
                real = compute_projection_score(cov_x=covX, cov_y=covY, cov_xy=covXY)
                assert torch.allclose(exp, real, atol=1e-5, rtol=1e-5), f"Correlation scores do not match {exp}!={real}"

    # Compute the average spectral score per each dt in [1, min(time_horizon, window_size))
    for idx, (s_scores, c_scores) in enumerate(zip(spectral_scores, corr_scores)):
        spectral_scores[idx] = torch.mean(s_scores)
        corr_scores[idx] = torch.mean(c_scores)

    return Tensor(spectral_scores), Tensor(corr_scores)


def compute_chain_projection_scores(CCov: Tensor, Cov: Tensor, Cov_prime: Tensor,
                                    window_size: Optional[int] = None, debug: bool = False):
    """ Compute the projection scores using the cross-covariance operators between distinct time steps.
    Args:
        CCov: (time_horizon, time_horizon, state_dim, state_dim) Tensor containing all the Cross-Covariance
         empirical operators between the states in the main trajectory and states in the auxiliary trajectory.
         Each entry of the tensor is a (state_dim, state_dim) covariance estimate.
         Such that CCov(i,j) = Cov(x_i, x'_j) ∀ i, j in [0, time_horizon], j >= i.
        Cov: (time_horizon, state_dim, state_dim) Tensor containing all the Covariance empirical operators between time
         steps on the main state space. Cov(i) = Cov(x_i, x_i) ∀ i in [0, time_horizon]
        Cov_prime: (time_horizon, state_dim, state_dim) Tensor containing all the Covariance empirical operators between
         time steps on the auxiliary state space. Cov_prime(i) = Cov(x'_i, x'_i) ∀ i in [0, time_horizon]
        debug: (bool) Whether to print debug information on the scores computed. Defaults to False.
    Returns:
        projection_scores: (time_horizon, time_horizon) Tensor containing the projection scores between all pairs of
            states i, j in [0, time_horizon], j >= i.
    """
    assert (len(CCov.shape) == 4 and CCov.shape[0] == CCov.shape[1]
            and CCov.shape[2] == CCov.shape[3]), f"Expected Cov_t_dt of shape (T, T, state_dim, state_dim)"
    assert (len(Cov.shape) == 3 and Cov.shape[0] == Cov.shape[1], f"Expected Cov of shape (T, state_dim, state_dim)")
    assert Cov.shape == Cov_prime.shape, f"Expected Cov_prime of shape (T, state_dim, state_dim)"
    time_horizon, state_dim, _ = Cov.shape

    # Compute the inversion and square root of the covariance matrices in a vectorized fashion.
    # Cov_sqrt_inv, cond_Cov = batch_matrix_sqrt_inv(Cov, epsilon=1e-5)
    # Cov_sqrt_inv_prime, cond_Cov_prime = batch_matrix_sqrt_inv(Cov_prime, epsilon=1e-5)

    max_dt = min(time_horizon, window_size + 1)
    corr_scores = []
    for dt in range(1, max_dt):
        cov_X_idx = range(time_horizon - dt)
        cov_Y_idx = range(dt, time_horizon)
        # CovX_sqrt_inv, CovY_sqrt_inv, = Cov_sqrt_inv[cov_X_idx], Cov_sqrt_inv_prime[cov_Y_idx]
        CovX, CovY = Cov[cov_X_idx], Cov_prime[cov_Y_idx]
        CovXY = CCov[cov_X_idx, cov_Y_idx]
        # corr_matrix = CovX_sqrt_inv @ CovXY @ CovY_sqrt_inv
        # corr_score = torch.linalg.matrix_norm(corr_matrix, ord='fro', dim=(-2, -1)) ** 2 / state_dim**2
        p_score = compute_projection_score(cov_x=CovX, cov_y=CovY, cov_xy=CovXY)
        corr_scores.append(p_score)

    if debug:  # Check vectorized operations are equivalent to sequential operations
        for dt in range(1, max_dt):
            s_scores_dt = corr_scores[dt - 1]
            for t in range(time_horizon - dt):
                covX, covY = Cov[t], Cov_prime[t + dt]
                covXY = CCov[t, t + dt]
                exp = s_scores_dt[t]
                real = compute_projection_score(cov_x=covX, cov_y=covY, cov_xy=covXY)
                assert torch.allclose(exp, real, atol=1e-5), f"Spectral scores do not match {exp}!={real}"

    # Compute the average spectral score per each dt in [1, min(time_horizon, window_size))
    for idx, scores in enumerate(corr_scores):
        corr_scores[idx] = torch.mean(scores)

    return Tensor(corr_scores)


def obs_state_space_metrics(obs_state_traj: Tensor,
                            obs_state_traj_prime: Optional[Tensor],
                            representation: Optional[Representation] = None,
                            max_ck_window_length: int = 2):
    """ Compute the metrics of an observable space with expected linear dynamics.

    This function computes the metrics of an observable space with expected linear dynamics. Specifically,
    Args:
        obs_state_traj (batch, time_horizon, obs_state_dim): trajectory of states
        obs_state_traj_prime (batch, time_horizon, obs_state_dim): Auxiliary trajectory of states
        representation: Symmetry representation on the observable space. If provided, the empirical covariance and
            cross-covariance operators will be improved using the group average trick
        max_ck_window_length: Maximum window length to compute the Chapman-Kolmogorov regularization term.
        ck_w: Weight of the Chapman-Kolmogorov regularization term.
    Returns:
        Dictionary containing:
            - spectral_score: (time_horizon - 1) Tensor containing the average spectral score between time steps separated
             apart by a shift of `dt` [steps/time]. That is:
                spectral_score[dt - 1] = avg(||Cov(x_i, x'_i+dt)||_HS^2/(||Cov(x_i, x_i)||_2*||Cov(x'_i+dt, x'_i+dt)||_2))
                 | ∀ i in [0, time_horizon - dt], dt in [1, min(time_horizon - i, window_size)]
            - corr_score: (time_horizon - 1) Tensor containing the correlation scores between time steps separated
             apart by a shift of `dt` [steps/time]. That is:
                corr_score[dt - 1] = avg(||Cov(x_i, x_i)^-1 Cov(x_i, x'_i+dt) Cov(x'_i+dt, x'_i+dt)^-1||_HS^2)
                 | ∀ i in [0, time_horizon - dt], dt in [1, min(time_horizon - i, window_size)]
            - orth_reg: (time_horizon) Tensor containing the orthonormality regularization term for each time step.
                That is orth_reg[t] = || Cov(t,t) - I ||_2
            - ck_reg: (time_horizon - 1,) Average CK error per `dt` time steps. That is:
                ck_error[dt - 2] = avg(|| Cov(t, t+dt) - Cov(t, t+1) Cov(t+1, t+2) ... Cov(t+dt-1, t+dt) ||) |
                ∀ t in [0, time_horizon - 2], dt in [2, min(time_horizon - 2, ck_window_length)]
            - cov_cond_num: (float) Average condition number of the Covariance matrices.
    """
    debug = log.level == logging.DEBUG  # TODO: remove default debug
    # Compute the empirical covariance and cross-covariance operators, ensuring that operators are equivariant.
    # Cov[i, j'] := Cov(i, j')  | t in [0, pred_horizon], i,j in [0, pred_horizon], j >= i
    CCov, Cov, Cov_prime = empirical_cov_cross_cov(state_traj=obs_state_traj, state_traj_prime=obs_state_traj_prime,
                                                   representation=representation, cov_window_size=max_ck_window_length,
                                                   debug=debug)
    # Orthonormality regularization terms for ALL time steps in horizon
    # reg_orthonormal[t] = || Cov(x_i, x_i) - I || | t in [0, pred_horizon]
    orthonormality_Cov = regularization_orthonormality(Cov)
    orthonormality_Cov_prime = regularization_orthonormality(Cov_prime)
    reg_orthonormal = (orthonormality_Cov + orthonormality_Cov_prime) / 2.0

    cond_num_Cov = torch.linalg.cond(torch.cat((Cov, Cov_prime), dim=0), p=2)

    # Compute the Projection, Spectral and Orthonormality regularization terms for ALL time steps in horizon.
    # spectral_scores[dt - 1] := ||Cov(t, t+dt)||^2_HS / (||Cov(t)|| ||Cov(t+d)||) | dt in [1, time_horizon)
    # corr_scores[dt - 1] := ||Cov(t)^-1 Cov(t, t+dt) Cov(t+d)^-1||^2_HS | dt in [1, time_horizon)
    spectral_scores, corr_scores = compute_chain_spectral_corr_scores(CCov=CCov,
                                                                      Cov=Cov, Cov_prime=Cov_prime, debug=debug)
    if debug:
        assert (corr_scores > spectral_scores).all(), "Correlation scores should be upper bound of spectral scores"

    # Compute the Chapman-Kolmogorov regularization scores for all possible step transitions. In return, we get:
    # ck_regularization[i,j] = || Cov(i, j) - ( Cov(i, i+1), ... Cov(j-1, j) ) ||_2  | j >= i + 2
    ck_regularization = chapman_kolmogorov_regularization(CCov=CCov,
                                                          ck_window_length=max_ck_window_length,
                                                          debug=debug)

    return dict(orth_reg=reg_orthonormal,
                ck_reg=ck_regularization,
                spectral_score=spectral_scores,
                corr_score=corr_scores,
                cov_cond_num=torch.mean(cond_num_Cov),
                # projection_score_t=torch.nanmean(projection_score_t, dim=0, keepdim=True),  # (batch, time)
                # spectral_score_t=torch.nanmean(spectral_score_t, dim=0, keepdim=True)       # (batch, time)
                )


def forecasting_loss_and_metrics(
        state_gt: Tensor, state_pred: Tensor) -> (Tensor, dict[str, Tensor]):
    # Compute state squared error over time and the infinite norm of the state dimension over time.
    l2_loss = torch.norm(state_gt - state_pred, p=2, dim=-1)
    time_steps = state_gt.shape[1]
    metrics = {}
    metrics['pred_loss_t'] = l2_loss
    return l2_loss.mean(), metrics


def batch_matrix_sqrt_inv(C, epsilon=None):
    """ Compute the inverse square root of a batch of matrices.
    Args:
        C: (batch, d, d) Tensor containing a batch of matrices to compute the inverse square root.
        epsilon: (float) Small constant to add to the eigenvalues to avoid numerical instability.
    Returns:
        C_inv_sqrt: (batch, d, d) Tensor containing the inverse square root of the input matrices.
    """
    # Perform batched eigenvalue decomposition
    eigenvalues, Q = torch.linalg.eigh(C)  # C = Q @ Lambda @ Q^T
    cond_num = eigenvalues[:, -1] / eigenvalues[:, 0]
    if epsilon is None:
        epsilon = torch.finfo(eigenvalues.dtype).eps
    # Handle degenerate matrices by adding a small constant
    eigenvalues = torch.where(eigenvalues > epsilon, eigenvalues, epsilon)

    # Compute Lambda^(-1/2)
    Lambda_inv_sqrt = 1.0 / torch.sqrt(eigenvalues).unsqueeze(-1)

    # Perform batched matrix multiplication to compute C^(-1/2)
    # Intermediate multiplication: eigenvectors * Lambda_inv_sqrt
    intermediate = Q * Lambda_inv_sqrt

    # Final multiplication: intermediate @ eigenvectors^T
    C_inv_sqrt = torch.matmul(intermediate, Q.transpose(-1, -2))

    return C_inv_sqrt, cond_num
