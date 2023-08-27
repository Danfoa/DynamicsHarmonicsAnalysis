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
    score = torch.linalg.matrix_norm(score, ord='fro')  # ||cov_x_inv @ cov_xy @ cov_y_inv||_HS^2
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
    score = torch.linalg.matrix_norm(cov_xy, ord='fro')  # == ||cov_xy|| 2, HS
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
                            cross_cov_only: bool = False,
                            check_equivariance: bool = False) -> (torch.Tensor, torch.Tensor):
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
        cross_cov_only: (bool) If True, only the cross-covariance operator is computed. Defaults to False.
        check_equivariance: (bool) If True, check that the empirical operators are equivariant. Defaults to False. 
    Returns:
        Cov_t: (batch, pred_horizon + 1, state_dim, state_dim) Empirical approximation of the covariance operator.
         Cov_t[t] = E[(X_t - E[X_t])(X_t - E[X_t])^T], being t in [0, pred_horizon + 1]
        Cov_t0_ti: (batch, pred_horizon, state_dim, state_dim) Empirical approximation of the cross-covariance operator.
         Cov_t0_ti[t] := Cov(X_0, X_t+1) = E[(X_0 - E[X_0])(X_t+1 - E[X_t+1])^T], being t in [0, pred_horizon]
    """
    obs_state_0 = state_0[:, None, :]  # Expand the time dimension
    pred_horizon = next_states.shape[1]
    # Apply batch operations to the entire trajectory of observations
    obs_state_traj = torch.cat([obs_state_0, next_states], dim=1)

    if not cross_cov_only:
        # Compute all the covariance matrix at each time step in a single batched operation
        # Cov_t[i] = E[(X_i - E[X_i])(X_i - E[X_i])^T] | t in [0, pred_horizon + 1]
        Cov_t = covariance(X=obs_state_traj, Y=obs_state_traj)
        # assert torch.allclose(Cov_t[0], obs_state_init.T @ obs_state_init, atol=1e-6)          # TODO: Test suit

    # Compute all the cross-covariance matrices between the initial state and all other states in batched ops.
    # To do this we duplicate the initial state `pred_horizon` times and do the batched operation
    obs_state_0d = obs_state_0.expand(-1, pred_horizon, -1)
    # Cov_t0_ti[i] := Cov(X_0, X_t+1) = E[(X_0 - E[X_0])(X_t+1 - E[X_t+1])^T] | t in [0, pred_horizon]
    Cov_t0_ti = covariance(X=obs_state_0d, Y=next_states)

    if representation is not None:
        # We can improve the empirical estimates by understanding that the TRUE operators are equivariant operators
        # ρ(g) Cov(X,Y) ρ(g)^T = Cov(ρ(g)X, ρ(g)Y) = Cov(X, Y) = CovXY
        # Thus we can apply the "group-average" trick to improve the estimate
        # CovXY = 1/|G| Σ_g ∈ G (ρ(g) Cov(X,Y) ρ(g)^T).
        # This is a costly op, but we do it in batched form and only for the group generators
        gens = representation.group.generators  # Generators of the symmetry group
        # Its more efficient if we apply this to Cov_t and CovXY at the same time
        tmp_equiv_op = torch.cat([Cov_t, Cov_t0_ti], dim=0) if not cross_cov_only else Cov_t0_ti
        # Compute each:      ρ(g) Cov(X,Y) ρ(g)^T   | ρ(g)^T = ρ(~g) = ρ(g^-1)
        Gtmp_equiv_op = [torch.einsum('na,tao,om->tnm',  # t=time, n,m,a,o=state_dim
                                      torch.tensor(representation(h), dtype=Cov_t0_ti.dtype, device=Cov_t0_ti.device),
                                      tmp_equiv_op,
                                      torch.tensor(representation(~h), dtype=Cov_t0_ti.dtype, device=Cov_t0_ti.device))
                         for h in gens]
        # Compute group average:  1/|G| Σ_g ∈ G (ρ(g) Cov(X,Y) ρ(g)^T).
        equiv_op_avg = torch.mean(torch.stack([tmp_equiv_op] + Gtmp_equiv_op, dim=0), dim=0)

        if not cross_cov_only:
            # Split back the covariance and cross-covariance operators
            Cov_t, Cov_t0_ti = torch.split(equiv_op_avg, [Cov_t.shape[0], Cov_t0_ti.shape[0]], dim=0)
        else:
            Cov_t0_ti = equiv_op_avg

        if check_equivariance:  # Check commutativity/equivariance
            for g in representation.group.elements:
                rep_h = torch.tensor(representation(g), dtype=Cov_t.dtype, device=Cov_t.device)
                for t in range(pred_horizon):
                    assert torch.allclose(rep_h @ Cov_t0_ti[t], Cov_t0_ti[t] @ rep_h, atol=1e-5), \
                        f"Max error {torch.max(torch.abs(rep_h @ Cov_t0_ti[t] - Cov_t0_ti[t] @ rep_h))}"
    if cross_cov_only:
        return Cov_t0_ti

    return Cov_t, Cov_t0_ti


def chapman_kolmogorov_regularization(state_0: torch.Tensor,
                                      next_states: torch.Tensor,
                                      ck_window_length: int = 2,
                                      representation: Optional[Representation] = None,
                                      cov_t0_ti: Optional[torch.Tensor] = None,
                                      verbose: bool = False):
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
        state_0: (batch, state_dim) Initial state
        next_states: (batch, pred_horizon, state_dim) future trajectory of states of length `pred_horizon`
        ck_window_length: (int) Maximum window length to compute the regularization term. Defaults to 2.
        representation (optional Representation): Group representation on the state space. If provided, the empirical
            covariance and cross-covariance operators will be improved using the group average trick.
        cov_t0_ti: (pred_horizon, state_dim, state_dim) Cross-covariance operators between the initial state and the
         next states in the trajectory. Defaults to None, in which case these operators are computed.
        verbose: (bool) Whether to print debug information on the CK scores computed. Defaults to False.
    Returns:

    """

    assert len(next_states.shape) == 3 and len(state_0.shape) == 2
    state_dim = state_0.shape[-1]
    pred_horizon = next_states.shape[1]

    # All CK terms can be computed by constructing a matrix of size (pred_horizon + 1) x (pred_horizon + 1) in which
    # each entry in the position i, j = Cov(X_i, X_j) for i < j. Then the transition chain of cross covariances
    # Cov(X_i, X_i+1) Cov(X_i+1, X_i+2) ... Cov(X_i+d-1, X_i+d) is the diagonal of the matrix starting at position
    # (i, i+1) and ending at position (i+d-1, i+d).
    cross_cov_ops = np.empty((pred_horizon + 1, pred_horizon + 1), dtype=object)
    for t in range(0, pred_horizon - 1):
        if t == 0:  # Initial condition
            cov_t_ti = cov_t0_ti if cov_t0_ti is not None else empirical_cov_cross_cov(state_0=state_0,
                                                                                       next_states=next_states,
                                                                                       representation=representation,
                                                                                       cross_cov_only=True)
        else:
            cov_t_ti = empirical_cov_cross_cov(state_0=next_states[:, t - 1, :],
                                               next_states=next_states[:, t:, :],
                                               representation=representation,
                                               cross_cov_only=True)
        cross_cov_ops[t, t + 1:] = [(c,) for c in cov_t_ti]
    assert np.all([a is None for a in np.diag(cross_cov_ops)]), "Diagonal (Cov(X_t,X_t) ∀ t) should be empty"

    if verbose:
        print("\t\t Chapman-Kolmogorov Scores")
        # index_matrix = np.empty((pred_horizon + 1, pred_horizon + 1), dtype=object)
        # for i, j in np.ndindex(index_matrix.shape):
        #     index_matrix[i, j] = (i, j)

    ck_scores = []
    min_steps = 2
    for t_start in range(0, pred_horizon - min_steps):
        chain_cov = None
        chain_test = []
        for t_end in range(t_start + min_steps, min(pred_horizon - 1, t_start + ck_window_length + 1)):
            r_min, r_max = t_start, t_end - 1
            c_min, c_max = t_start + 1, t_end
            # Compute the covariance chain using Dynamic Programming (i.e., do not repeat computations)
            # Cov(X_t, X_t+1), Cov(X_t+1, X_t+2), ... Cov(X_te-1, X_te)
            if chain_cov is None:
                chain_cov = cross_cov_ops[r_min, c_min][0] @ cross_cov_ops[r_max, c_max][0]
                chain_test.extend([(r_min, c_min), (r_max, c_max)])
            else:
                chain_cov = chain_cov @ cross_cov_ops[r_max, c_max][0]
                chain_test.append((r_max, c_max))
            # Select the target covariance as the operator in the upper right
            target_cov = cross_cov_ops[t_start, t_end][0]  # Cov(X_ts, X_te)
            # || Cov(X_ts, X_te) - Cov(X_ts, X_ts+1), Cov(X_ts+1, X_ts+2), ... Cov(X_te-1, X_te) ||_2
            ck_scores += [torch.linalg.matrix_norm(chain_cov - target_cov, ord='fro')]

            if verbose:
                # chain_test = np.diag(index_matrix[r_min:r_max, c_min:c_max])
                target_test = (t_start, t_end)
                print(f"{ck_scores[-1]:.3f} \t = || " + '·'.join([f"Cov{c}" for c in chain_test]) +
                      f" \t-\t Cov{target_test} ||")

    return ck_scores
