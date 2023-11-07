from typing import Optional, Union

import numpy as np
import torch.linalg
from escnn.group import Representation
from torch import Tensor


def represent_linear_map_in_basis(basis_linear_map: np.ndarray, in_linear_map: np.ndarray) -> np.ndarray:
    """
    Represent a given linear map in terms of a basis set of linear maps.

    Args:
    basis_linear_map (ndarray): An array of shape (basis_dimension, out_dim, in_dim) representing the basis of linear
    maps.
    in_linear_map (ndarray): An array of shape (out_dim, in_dim) representing the input linear map.

    Returns:
    ndarray: Coefficients of the input linear map in the basis.

    Raises:
    ValueError: If the input linear map cannot be represented in the given basis.
    """

    # Flatten the basis linear maps and input linear map to column vectors
    basis_dim, out_dim, in_dim = basis_linear_map.shape
    flat_basis = basis_linear_map.reshape(basis_dim, -1)
    flat_in_map = in_linear_map.reshape(-1, 1)

    # Initialize an array to store the coefficients
    coefficients = np.zeros(basis_dim)

    # Calculate the coefficients using vector projection formula
    for i in range(basis_dim):
        numerator = np.dot(flat_basis[i], flat_in_map)
        denominator = np.dot(flat_basis[i], flat_basis[i])

        # Protect against division by zero (shouldn't happen for a basis)
        if denominator == 0:
            raise ValueError("Basis vector has zero magnitude. Invalid basis.")

        coefficients[i] = numerator / denominator

    # Reconstruct the input linear map using the calculated coefficients
    reconstructed_map = np.sum(coefficients[:, None, None] * basis_linear_map, axis=0)

    # Calculate the reconstruction error
    error = np.linalg.norm(reconstructed_map - in_linear_map)

    # If the error is zero (or very close to zero), the map can be represented in the basis
    if error > 1e-10:
        raise ValueError("The input linear map cannot be represented in the given basis.")

    return coefficients


def full_rank_lstsq_symmetric(X: Tensor,
                              Y: Tensor,
                              rep_X: Optional[Representation] = None,
                              rep_Y: Optional[Representation] = None,
                              bias: bool = True) -> [Tensor, Union[Tensor, None]]:
    """ Compute the least squares solution of the linear system Y = A·X + B.

    If the representation is provided the empirical transfer operator is improved using the group average trick to
    enforce equivariance considering that:
                        rep_Y(g) y = A rep_X(g) x
                    rep_Y(g) (A x) = A rep_X(g) x
                        rep_Y(g) A = A rep_X(g)
            rep_Y(g) A rep_X(g)^-1 = A                | forall g in G.

    TODO: Parallelize
    Args:
        X: (|x|, n_samples) Data matrix of the initial states.
        Y: (|y|, n_samples) Data matrix of the next states.
        rep_X: Map from group elements to matrices of shape (|x|,|x|) transforming x in X.
        rep_Y: Map from group elements to matrices of shape (|y|,|y|) transforming y in Y.
        bias: Whether to include a bias term in the linear model.
    Returns:
        A: (|y|, |x|) Least squares solution of the linear system `Y = A·X + B`.
        B: Bias vector of dimension (|y|, 1). Set to None if bias=False.
    """

    A, B = full_rank_lstsq(X, Y, bias=bias)
    if rep_X is None or rep_Y is None:
        return A, B
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
            rep_X_g_inv = torch.from_numpy(rep_X(~g)).to(dtype=X.dtype, device=X.device)
            rep_Y_g = torch.from_numpy(rep_Y(g)).to(dtype=X.dtype, device=X.device)
            A_g = rep_Y_g @ A @ rep_X_g_inv
        A_G.append(A_g)
    A_G = torch.stack(A_G, dim=0)
    A_G = torch.mean(A_G, dim=0)

    if bias:
        # Bias can only be present in the dimension of the output space associated with the trivial representation of G.
        B_G = torch.zeros_like(B)
        dim = 0
        for irrep_id in rep_Y.irreps:
            irrep = group.irrep(*irrep_id if isinstance(irrep_id, tuple) else (irrep_id,))
            if irrep == group.trivial_representation:
                B_G[dim] = B[dim]
            dim += irrep.size
        return A_G.to(dtype=X.dtype, device=X.device), B_G.to(dtype=X.dtype, device=X.device)
    return A_G.to(dtype=X.dtype, device=X.device), None


def full_rank_lstsq(X: Tensor, Y: Tensor, driver='gelsd', bias=True) -> [Tensor, Union[Tensor, None]]:
    """Compute the least squares solution of the linear system `X' = A·X + B`. Assuming full rank X and A.
    Args:<
        X: (|x|, n_samples) Data matrix of the initial states.
        Y: (|y|, n_samples) Data matrix of the next states.
    Returns:
        A: (|y|, |x|) Least squares solution of the linear system `X' = A·X`.
        B: Bias vector of dimension (|y|, 1). Set to None if bias=False.
    """
    assert (
            X.ndim == 2 and Y.ndim == 2 and X.shape[1] == Y.shape[1]
    ), f"X: {X.shape}, Y: {Y.shape}. Expected (|x|, n_samples) and (|y|, n_samples) respectively."

    if bias:
        # In order to solve for the bias in the same least squares problem we need to augment the data matrix X, with an
        # additional dimension of ones. This is equivalent to switching to Homogenous coordinates
        X_aug = torch.cat([X, torch.ones((1, X.shape[1]), device=X.device, dtype=X.dtype)], dim=0)
    else:
        X_aug = X

    # Torch convention uses Y:(n_samples, |y|) and X:(n_samples, |x|) to solve the least squares
    # problem for `Y = X·A`, instead of our convention `Y = A·X`. So we have to do the appropriate transpose.
    result = torch.linalg.lstsq(X_aug.T.detach().cpu().to(dtype=torch.double),
                                Y.T.detach().cpu().to(dtype=torch.double), rcond=None, driver=driver)
    A_sol = result.solution.T.to(device=X.device, dtype=X.dtype)
    if bias:
        assert A_sol.shape == (Y.shape[0], X.shape[0] + 1)
        # Extract the matrix A and the bias vector B
        A, B = A_sol[:, :-1], A_sol[:, [-1]]
        return A.to(dtype=X.dtype, device=X.device), B.to(dtype=X.dtype, device=X.device)
    else:
        assert A_sol.shape == (Y.shape[0], X.shape[0])
        A, B = A_sol, None
        return A.to(dtype=X.dtype, device=X.device), B


def matrix_average_trick(
        A: Union[np.ndarray, torch.Tensor],
        Bs: Union[list[np.ndarray], list[torch.Tensor]]
        ) -> Union[np.ndarray, torch.Tensor]:
    if isinstance(A, np.ndarray):
        return sum([B @ A @ B.conj().T for B in Bs]) / len(Bs)
    else:
        return torch.sum([B @ A @ B.conj().T for B in Bs]) / len(Bs)
