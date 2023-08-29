from functools import reduce
from typing import Optional

import numpy as np
import torch
from escnn.group import Representation


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


def covariance(X: torch.Tensor, Y: torch.Tensor):
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
    CovXY = torch.einsum('tob,tba->toa', torch.permute(X, dims=(1, 2, 0)), torch.permute(Y, dims=(1, 0, 2)))
    return CovXY / n_samples


def empirical_cov_cross_cov(state_0: torch.Tensor,
                            next_states: torch.Tensor,
                            representation: Optional[Representation] = None,
                            debug: bool = False) -> (torch.Tensor, torch.Tensor):
    """ Compute empirical approximation of the covariance and cross-covariance operators for a trajectory of states.
    This function computes the empirical approximation of the covariance and cross-covariance operators in batched
    matrix operations for efficiency.
    Args:
        state_0: (batch, state_dim) Initial state
        next_states: (batch, pred_horizon, state_dim) future trajectory of states of length `pred_horizon`
        representation (optional Representation): Group representation on the state space. If provided, the empirical
         covariance and cross-covariance operators will be improved using the group average trick:
         Cov_t0_ti = 1/|G| Σ_g ∈ G (ρ(g) Cov(X_0, X_i) ρ(g)^T), ensuring that the empirical operators are equivariant:
         Cov_t0_ti ρ(g) = ρ(g) Cov_t0_ti <==> Cov(ρ(g)X_0, ρ(g)X_i) = ρ(g) Cov_t0_ti ρ(g)^* = Cov_t0_ti
        debug: (bool) If True, check that the empirical operators are equivariant. Defaults to False.
    Returns:
        Cov_t_tdt: (pred_horizon + 1, pred_horizon + 1, state_dim, state_dim) Tensor containing all the Covariance
         and Cross-Covariance empirical operators between the states in the trajectory. Each entry of the tensor is a
         (state_dim, state_dim) covariance estimate.
         Such that Cov_t_tdt[i, j] = Cov(X_i, X_j) ∀ i, j in [0, pred_horizon + 1], j >= i.
    """
    assert len(state_0.shape) == 2, f"state_0 should be of shape (batch, state_dim) but is {state_0.shape}"
    num_samples, state_dim = state_0.shape
    dtype, device = state_0.dtype, state_0.device
    pred_horizon = next_states.shape[1]
    # Apply batch operations to the entire trajectory of observations
    state_traj = torch.cat([state_0.unsqueeze(1), next_states], dim=1)

    # This tensor will hold all the Covariance operators between time steps of the trajectory. Meaning that at the
    # location Cov_t[i, j] := Cov(X_i, X_j) will have the covariance operator between the states at time i and j.
    # This matrix will be upper triangular with all lower entries set to nan.
    Cov_t_tdt = torch.zeros((pred_horizon + 1, pred_horizon + 1, state_dim, state_dim), dtype=dtype, device=device)

    # Compute all the Cov Operators on the diagonal, that is all Cov ops between the same time step:
    # Cov_t[i] = E[(X_i - E[X_i])(X_i - E[X_i])^T] | t in [0, pred_horizon + 1]
    Cov_t = covariance(X=state_traj, Y=state_traj)
    # Store the results on the diagonal of the Cov_t_tdt matrix
    Cov_t_tdt[torch.arange(pred_horizon + 1), torch.arange(pred_horizon + 1)] = Cov_t

    # Compute all the Cross-Covariance operators between distinct time steps.
    # Cov_t_ti[i] := Cov(X_t, X_t+1), ... Cov(X_t, X_H) | t in [0, pred_horizon]
    for t in range(pred_horizon):
        # To do this we "duplicate" the state at time t for the next `pred_horizon - t` times compute in batch ops.
        obs_state_t = state_traj[:, t, :].unsqueeze(1).expand(-1, pred_horizon - t, -1)
        next_states_t = state_traj[:, t + 1:, :]
        assert obs_state_t.shape == next_states_t.shape, f"{obs_state_t.shape} != {next_states_t.shape}"
        Cov_t_ti = covariance(X=obs_state_t, Y=next_states_t)
        # Store the results in the Cov_t_tdt matrix
        Cov_t_tdt[t, t + 1:] = Cov_t_ti

    if debug:  # Sanity checks. These should be true by construction
        # Check Covariance Operator at initial timestep
        assert torch.allclose(Cov_t_tdt[0, 0], state_0.T @ state_0 / num_samples, atol=1e-6)
        # Check Cross-Covariance operator ...
        assert torch.allclose(Cov_t_tdt[0, pred_horizon],
                              state_0.T @ state_traj[:, pred_horizon, ...] / num_samples, atol=1e-6)

    if representation is not None:
        # We can improve the empirical estimates by understanding that the theoretical operators are equivariant:
        # ρ(g) Cov(X,Y) ρ(g)^T = Cov(ρ(g)X, ρ(g)Y) = Cov(X, Y) = CovXY
        # Thus we can apply the "group-average" trick to improve the estimate:
        # CovXY = 1/|G| Σ_g ∈ G (ρ(g) Cov(X,Y) ρ(g)^T) (see https://arxiv.org/abs/1111.7061)
        # This is a costly operation but is equivalent to doing data augmentation of the state space samples,
        # with all group elements, and then computing the empirical covariance operators.
        # Furthermore, we can apply this operation in parallel to all Cov Ops for numerical efficiency in GPU.
        orbit_Cov_t_tdt = [Cov_t_tdt]
        for h in representation.group.generators:  # Generators of the symmetry group. We only need these.
            # Compute each:      ρ(g) Cov(X,Y) ρ(g)^T   | ρ(g)^T = ρ(~g) = ρ(g^-1)
            orbit_Cov_t_tdt.append(torch.einsum('na,ltao,om->ltnm',  # t,l=time, n,m,a,o=state_dim
                                                torch.tensor(representation(h), dtype=dtype, device=device),
                                                Cov_t_tdt,
                                                torch.tensor(representation(~h), dtype=dtype, device=device)))

        # Compute group average:  1/|G| Σ_g ∈ G (ρ(g) Cov(X,Y) ρ(g)^T).
        Cov_t_tdt = torch.mean(torch.stack(orbit_Cov_t_tdt, dim=0), dim=0)

        if debug:  # Check commutativity/equivariance of the empirical estimates of all Covariance operators
            for g in representation.group.elements:
                rep_h = torch.tensor(representation(g), dtype=dtype, device=device)
                cov_rep = torch.einsum('na,ltao->ltno', rep_h, Cov_t_tdt)  # t,l=time, n,m,a,o=state_dim
                rep_cov = torch.einsum('ltao,om->ltam', Cov_t_tdt, rep_h)
                assert torch.allclose(cov_rep[0, :], rep_cov[0, :], atol=1e-5), \
                    f"Max equivariance error {torch.max(torch.abs(cov_rep[0, :] - rep_cov[0, :]))}"

    return Cov_t_tdt


def chapman_kolmogorov_regularization(Cov_t_dt: torch.Tensor,
                                      ck_window_length: int = 2,
                                      representation: Optional[Representation] = None,
                                      cov_t0_ti: Optional[torch.Tensor] = None,
                                      debug: bool = False):
    """ Compute the Chapman-Kolmogorov regularization using the cross-covariance operators between distinct time steps.

    This regularization aims at exploitation Markov Assumption of a linear dynamical system. Specifically it computes:
    ||Cov(X_t, X_t+d) - Cov(X_t, X_t+1) Cov(X_t+1, X_t+2) ... Cov(X_t+d-1, X_t+d) || ∀ t in [0, pred_horizon-2],
    d in [2, min(pred_horizon, ck_window_length)].

    The function thus computes all possible regularization terms of the provided initial state and state trajectory,
    withing the provided window length `ck_window_length`.

    See more in:
    [1] Vladimir Kostic, Pietro Novelli, Riccardo Grazzi, Karim Lounici, and Massimiliano Pontil.
    “Deep Projection Networks for Learning Time-Homogeneous Dynamical Systems.” arXiv, July 19,
    2023. https://doi.org/10.48550/arXiv.2307.09912.

    Args:
        Cov_t_dt: (time_horizon, time_horizon, state_dim, state_dim) Tensor containing in all empirical covariance
         operators between states in a trajectory of length `time_horizon`. Each entry of the tensor is assumed to be:
            Cov_t_dt[i, j] = Cov(X_i, X_j) ∀ i, j in [0, time_horizon], j >= i.
        ck_window_length: (int) Maximum window length to compute the regularization term. Defaults to 2.
        representation (optional Representation): Group representation on the state space. If provided, the empirical
            covariance and cross-covariance operators will be improved using the group average trick.
        cov_t0_ti: (pred_horizon, state_dim, state_dim) Cross-covariance operators between the initial state and the
         next states in the trajectory. Defaults to None, in which case these operators are computed.
        debug: (bool) Whether to print debug information on the CK scores computed. Defaults to False.
    Returns:

    """
    assert (len(Cov_t_dt.shape) == 4 and Cov_t_dt.shape[0] == Cov_t_dt.shape[1]
            and Cov_t_dt.shape[2] == Cov_t_dt.shape[3]), f"Expected Cov_t_dt of shape (T, T, state_dim, state_dim)"
    time_horizon = Cov_t_dt.shape[0]
    dtype, device = Cov_t_dt.dtype, Cov_t_dt.device

    # Generate upper triangular matrix that will contain the CK error values at each position, such that:
    # ck_errors[i, j] = || Cov(X_i, X_j) - Cov(X_i, X_i+1) Cov(X_i+1, X_i+2) ... Cov(X_j-1, X_j) || | j >= i+2
    ck_errors = torch.fill(torch.zeros((time_horizon, time_horizon), dtype=dtype, device=device), torch.nan)

    if debug:
        print("\t\t Chapman-Kolmogorov Scores")

    # Minimum number of steps to compute the CK regularization term
    # ck_errors[t, t+2] = || Cov(X_t, X_t+2) - Cov(X_t, X_t+1) Cov(X_t+1, X_t+2) ||
    min_steps = 2
    for ts in range(0, time_horizon - 2):  # ts ∈ [0, time_horizon - 2]
        chain_cov = None
        chain_test = []  # te ∈ [ts + 2, min(pred_horizon, ck_window)]
        max_dt = min(ck_window_length, time_horizon - ts)
        for dt in range(min_steps, max_dt):
            te = ts + dt  # te ∈ [ts + 2, min(pred_horizon, ts + ck_window)]
            # Compute the covariance chain using Dynamic Programming (i.e., do not repeat computations)
            # Cov(X_ts, X_ts+1), Cov(X_ts+1, X_ts+2), ... Cov(X_te-1, X_te)
            if chain_cov is None:  # chain_cov = Cov(X_ts, X_ts+1), Cov(X_ts+1, X_ts+2)
                chain_cov = Cov_t_dt[ts, ts + 1] @ Cov_t_dt[ts + 1, ts + 2]
                chain_test.extend([(ts, ts + 1), (ts + 1, ts + 2)])
            else:  # chain_cov *= Cov(X_te-1, X_te)
                chain_cov = chain_cov @ Cov_t_dt[te - 1, te]
                chain_test.append((te - 1, te))
            # Select the target covariance as the operator in the upper right if the [ts : te, ts : te] block.
            target_cov = Cov_t_dt[ts, te]  # Cov(X_ts, X_te)

            # || Cov(X_ts, X_te) - Cov(X_ts, X_ts+1), Cov(X_ts+1, X_ts+2), ... Cov(X_te-1, X_te) ||_2
            ck_errors[ts, te] = torch.linalg.matrix_norm(chain_cov - target_cov, ord='fro')

            if debug:
                # chain_test = np.diag(index_matrix[r_min:r_max, c_min:c_max])
                target_test = (ts, te)
                print(f"{ck_errors[ts, te]:.3f} \t = || " + '·'.join([f"Cov{c}" for c in chain_test]) +
                      f" \t-\t Cov{target_test} ||")

    return ck_errors


def compute_chain_spectral_scores(Cov_t_dt: torch.Tensor, debug: bool = False):
    """ Compute the spectral scores using the cross-covariance operators between distinct time steps.

    Args:
        Cov_t_dt: (time_horizon, time_horizon, state_dim, state_dim) Tensor containing in all empirical covariance
         operators between states in a trajectory of length `time_horizon`. Each entry of the tensor is assumed to be:
         Cov_t_dt[i, j] = Cov(X_i, X_j) ∀ i, j in [0, time_horizon], j >= i.
        debug: (bool) Whether to print debug information on the scores computed. Defaults to False.
    Returns:
        spectral_scores: (time_horizon, time_horizon) Tensor containing the spectral scores between all pairs of states
            i, j in [0, time_horizon], j >= i.
    """
    assert (len(Cov_t_dt.shape) == 4 and Cov_t_dt.shape[0] == Cov_t_dt.shape[1]
            and Cov_t_dt.shape[2] == Cov_t_dt.shape[3]), f"Expected Cov_t_dt of shape (T, T, state_dim, state_dim)"
    time_horizon = Cov_t_dt.shape[0]
    dtype, device = Cov_t_dt.dtype, Cov_t_dt.device

    spectral_scores = torch.fill(torch.zeros((time_horizon, time_horizon), dtype=dtype, device=device), torch.nan)
    # Compute the norm of the diagonal of the covariance matrices is a single parallel operation.
    Cov_t = Cov_t_dt[range(time_horizon), range(time_horizon)]
    norm_Cov_t = torch.linalg.matrix_norm(Cov_t, ord=2, dim=(-2, -1))  # norm_Cov_t[i] = ||Cov(X_i, X_i)||_2

    # Compute the HS norm of the Cross-Covariance operators in a single parallel operation.
    # norm_Cov_t_dt[i, j] = ||Cov(X_i, X_j)||_HS
    norm_Cov_t_dt = torch.linalg.matrix_norm(Cov_t_dt, ord='fro', dim=(-2, -1))

    for ts in range(time_horizon):
        for te in range(ts + 1, time_horizon):
            norm_CovX, norm_CovY, = norm_Cov_t[ts], norm_Cov_t[te]
            spectral_scores[ts, te] = norm_Cov_t_dt[ts, te] ** 2 / (norm_CovX * norm_CovY)
            # TODO: # Shall we scale ( / state_dim)?

    if debug:
        for ts in range(time_horizon):
            for te in range(ts + 1, time_horizon):
                exp = spectral_scores[ts, te]
                real = compute_spectral_score(cov_x=Cov_t[ts], cov_y=Cov_t[te], cov_xy=Cov_t_dt[ts, te])
                assert torch.allclose(exp, real, atol=1e-5), f"Spectral scores do not match {exp}!={real}"

    return spectral_scores


def compute_chain_projection_scores(Cov_t_dt: torch.Tensor, debug: bool = False):
    """ Compute the projection scores using the cross-covariance operators between distinct time steps.

    Args:
        Cov_t_dt: (time_horizon, time_horizon, state_dim, state_dim) Tensor containing in all empirical covariance
         operators between states in a trajectory of length `time_horizon`. Each entry of the tensor is assumed to be:
         Cov_t_dt[i, j] = Cov(X_i, X_j) ∀ i, j in [0, time_horizon], j >= i.
        debug: (bool) Whether to print debug information on the scores computed. Defaults to False.
    Returns:
        projection_scores: (time_horizon, time_horizon) Tensor containing the projection scores between all pairs of
            states i, j in [0, time_horizon], j >= i.
    """
    assert (len(Cov_t_dt.shape) == 4 and Cov_t_dt.shape[0] == Cov_t_dt.shape[1]
            and Cov_t_dt.shape[2] == Cov_t_dt.shape[3]), f"Expected Cov_t_dt of shape (T, T, state_dim, state_dim)"
    time_horizon = Cov_t_dt.shape[0]
    dtype, device = Cov_t_dt.dtype, Cov_t_dt.device

    projection_scores = torch.fill(torch.zeros((time_horizon, time_horizon), dtype=dtype, device=device), torch.nan)
    # Compute the norm of the diagonal of the covariance matrices is a single parallel operation.
    Cov_t = Cov_t_dt[range(time_horizon), range(time_horizon)]
    Cov_t_inv = torch.linalg.pinv(Cov_t, hermitian=True)

    for ts in range(time_horizon):
        for te in range(ts + 1, time_horizon):
            projection_scores[ts, te] = torch.linalg.matrix_norm(
                Cov_t_inv[ts] @ Cov_t_dt[ts, te] @ Cov_t_inv[te], ord='fro') ** 2

    if debug:
        for ts in range(time_horizon):
            for te in range(ts + 1, time_horizon):
                exp = projection_scores[ts, te]
                real = compute_projection_score(cov_x=Cov_t[ts], cov_y=Cov_t[te], cov_xy=Cov_t_dt[ts, te])
                # assert torch.abs((exp - real) / real) < 0.1, f"Projection scores do not match {exp}!={real}"

    return projection_scores
