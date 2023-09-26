import importlib
import json
import pathlib
import warnings
from typing import Union

import numpy as np
import torch.nn

import math

from itertools import chain, combinations

from torch import Tensor


def powerset(iterable):
    "Return the list of all subsets of the input iterable"
    s = list(iterable)
    return list(chain.from_iterable(combinations(s, r) for r in range(len(s)+1)))


class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def matrix_average_trick(
        A: Union[np.ndarray, torch.Tensor],
        Bs: Union[list[np.ndarray], list[torch.Tensor]]
        ) -> Union[np.ndarray, torch.Tensor]:
    if isinstance(A, np.ndarray):
        return sum([B @ A @ B.conj().T for B in Bs]) / len(Bs)
    else:
        return torch.sum([B @ A @ B.conj().T for B in Bs]) / len(Bs)


def best_rectangular_grid(n):
    best_pair = (1, n)
    best_perimeter = sum(best_pair)
    for i in range(2, int(math.sqrt(n)) + 1):
        if n % i == 0:
            new_pair = (i, n // i)
            if (sum(new_pair) < best_perimeter):
                best_pair = new_pair
                best_perimeter = sum(new_pair)

    return best_pair


def class_from_name(module_name, class_name):
    # load the module, will raise ImportError if module cannot be loaded
    m = importlib.import_module(module_name)
    # get the class, will raise AttributeError if class cannot be found
    c = getattr(m, class_name)
    return c


def print_dict(d: dict, sort=False):
    str = []
    d_sorted = dict(sorted(d.items())) if sort else d
    for k, v in d_sorted.items():
        str.append(f"{k}={v}")
    return "-".join(str)


def append_dictionaries(dict1, dict2, recursive=True):
    import torch
    result = {}
    for k in set(dict1) | set(dict2):
        item1, item2 = dict1.get(k, 0), dict2.get(k, 0)
        if isinstance(item1, list) and (isinstance(item2, int) or isinstance(item2, float)):
            result[k] = item1 + [item2]
        elif isinstance(item1, int) or isinstance(item1, float):
            result[k] = [item1, item2]
        elif isinstance(item1, torch.Tensor) and isinstance(item2, torch.Tensor):
            # try:
            result[k] = torch.hstack((item1, item2))
            # except RuntimeError as e:
                # result[k] = torch.cat((torch.unsqueeze(item1, 0), torch.unsqueeze(item2, 0)))
        elif isinstance(item1, dict) and isinstance(item2, dict) and recursive:
            result[k] = append_dictionaries(item1, item2)
    return result


def flatten_dict(d: dict, prefix=''):
    a = {}
    for k, v in d.items():
        if isinstance(v, dict):
            a.update(flatten_dict(v, prefix=f"{k}/"))
        else:
            a[f"{prefix}{k}"] = v
    return a


def check_if_resume_experiment(ckpt_call):
    ckpt_path = pathlib.Path(ckpt_call.dirpath).joinpath(ckpt_call.CHECKPOINT_NAME_LAST + ckpt_call.FILE_EXTENSION)
    best_path = pathlib.Path(ckpt_call.dirpath).joinpath(ckpt_call.filename + ckpt_call.FILE_EXTENSION)

    terminated = False
    if best_path.exists() and not ckpt_path.exists():
        terminated = True
    elif ckpt_path.exists() and best_path.exists():
        terminated = False

    return terminated, ckpt_path, best_path


def compare_dictionaries(dict1: dict, dict2: dict):
    """
    Recursively compare dictionaries which entries might be other dictionaries.
    Any entry in one dictionary that is not present in the other dictionary needs to be reported.
    Args:
        dict1: A tree of dictionaries.
        dict2: A tree of dictionaries.

    Returns: The dictionary containing only the keys that are different between the two dictionaries. In all levels of
    the tree. Each entry of this difference dictionary is a tuple (val1, val2) when val1 != val2.
    """
    diff = {}
    for key in dict1.keys():
        if key not in dict2.keys():
            diff[key] = (dict1[key], None)
        else:
            if isinstance(dict1[key], dict):
                inner_diff = compare_dictionaries(dict1[key], dict2[key])
                if len(inner_diff) > 0:
                    diff[key] = inner_diff
            elif isinstance(dict1[key], np.ndarray):
                if not np.allclose(dict1[key] - dict2[key], 0, rtol=1e-6, atol=1e-6):
                    diff[key] = (dict1[key], dict2[key])
            else:
                if dict1[key] != dict2[key]:
                    diff[key] = (dict1[key], dict2[key])

    for key in dict2.keys():
        if key not in dict1.keys():
            diff[key] = (None, dict2[key])

    return diff


def random_orthogonal_matrix(n):
    """Generate a random n x n orthogonal matrix."""
    random_matrix = np.random.randn(n, n)
    Q, _ = np.linalg.qr(random_matrix)
    return Q


def companion_matrix(eig):
    """Return a companion matrix given a single eigenvalue."""
    if np.imag(eig) == 0:  # Real eigenvalue
        return np.array([[-np.abs(eig)]])

    # For complex eigenvalues
    real_part = np.real(eig)
    # imag_part = np.imag(eig)

    return np.array([[2 * -np.abs(real_part), -np.abs(eig) ** 2],
                     [1             ,             0]])


def random_well_conditioned_invertible_matrix(n, perturbation_scale=0.1):
    """
    Generate a nearly well-conditioned matrix of size n x n.

    Args:
    - n (int): Dimension of the matrix.
    - perturbation_scale (float): Magnitude of the perturbation.
        Small values (e.g., 0.1) will ensure the resulting matrix is well-conditioned.

    Returns:
    - numpy.ndarray: Generated matrix.
    """
    # Start with identity matrix
    T = np.eye(n)

    # Create low-rank perturbation
    for _ in range(n // 2):
        u = np.random.randn(n, 1)
        v = np.random.randn(n, 1)
        perturbation = u @ v.T
        T += perturbation_scale * perturbation

    print(f"Condition number of invertible matrix cond(T): {np.linalg.cond(T)}")
    if np.linalg.cond(T) > 10:
        warnings.warn(f"Condition number of invertible matrix cond(T): {np.linalg.cond(T)}")
    return T


def find_combinations(target_dim, irreps_dims, idx=0):
    """# A = np.array([[0, 1], [-2, -1]]).

    # target_dim = 8
    # irreps_dims = [2, 1]
    # combinations = find_combinations(target_dim, irreps_dims)
    # combinations = [tuple(sorted(c)) for c in combinations]
    # combinations = list(set(combinations))  # remove duplicates
    #
    """
    # Base case: if target is 0, we're done
    if target_dim == 0:
        return [[]]
    if idx == len(irreps_dims):
        return []

    current_dim = irreps_dims[idx]
    max_irreps = target_dim // current_dim

    results = []

    # Try to fit as many of the current irrep dimension into the target as possible
    for num_irreps in range(max_irreps + 1):
        remaining_dim = target_dim - current_dim * num_irreps
        sub_combinations = find_combinations(remaining_dim, irreps_dims, idx + 1)
        for comb in sub_combinations:
            results.append([current_dim] * num_irreps + comb)

    return results


import re


def format_scientific(text):
    """
    Formats any number in a string to scientific notation with 2 significant figures.

    Parameters:
    - text (str): The input text

    Returns:
    - str: Text with numbers formatted in scientific notation
    """

    # Function to replace each number with its scientific notation form
    def replace_number(match):
        number = float(match.group())
        return "{:.1e}".format(number)

    # Regular expression to find numbers after "=" and replace them
    str = re.sub(r"(?<=\=)(\d+\.\d+|\d+\.|\.\d+|\d+)", replace_number, text)
    # Remove the following keywords from the string
    keywords = ['model.', 'hydra.', 'system.']
    for keyword in keywords:
        str = str.replace(keyword, '')
    return str

def traj_from_states(state: Tensor, next_state: Tensor) -> Tensor:
    """
    Concatenate states and next_states into a trajectory.
    Args:
        state: (batch_size, state_dim)
        next_state: (batch_size, state_dim) or (batch_size, time, state_dim)

    Returns:
        state_trajectory: (batch_size, time, state_dim)
    """

    if len(state.shape) == 2:
        state = torch.unsqueeze(state, dim=1)
    if len(next_state.shape) == 2:
        next_state = torch.unsqueeze(next_state, dim=1)

    if next_state is not None:
        state_trajectory = torch.cat([state, next_state], dim=1)
    else:
        state_trajectory = state

    return state_trajectory

def states_from_traj(state_trajectory: Tensor) -> Tensor:
    """
    Split a trajectory into states and next_states.
    Args:
        state_trajectory: (batch_size, time, state_dim)
    Returns:
        state: (batch_size, state_dim)
        next_state: (batch_size, state_dim)
    """
    assert len(state_trajectory.shape) == 3, f"{state_trajectory.shape} - Expected (batch_size, time, state_dim)"
    state = state_trajectory[:, 0, :]
    next_state = state_trajectory[:, 1:, :]
    return state, next_state


def batched_to_flat_trajectory(x):
    """
    Converts a 3D tensor of shape (batch, time, dim) to a 2D tensor (batch * time, dim).
    Ensures that the ordering of samples in time is not shuffled. The tensor is first made contiguous
    in memory to ensure that reshaping does not affect the data order.

    Args:
        x (torch.Tensor): Input tensor of shape (batch, time, dim).

    Returns:
        torch.Tensor: Reshaped tensor of shape (batch * time, dim).
    """
    x_contiguous = x.contiguous()  # Needed for reshaping not to mess with the time order.
    x_reshaped = x_contiguous.view(-1, x_contiguous.size(-1))
    return x_reshaped


def flat_to_batched_trajectory(x_reshaped, batch_size, state_dim):
    """
    Reshapes a 2D tensor back to a 3D tensor with potentially new feature dimension.
    Infers the time dimension from the shape of the 2D tensor.

    Args:
        x_reshaped (torch.Tensor): Input tensor of shape (batch * time, new_dim).
        batch_size (int): The original batch size.
        state_dim (int): The new feature dimension size.

    Returns:
        torch.Tensor: Reshaped tensor of shape (batch, time, new_dim).

    The time dimension is inferred from the total number of elements in the 2D tensor
    and the known batch size. The tensor is reshaped back to its original 3D form.
    """
    total_elements = x_reshaped.size(0)
    assert total_elements % batch_size == 0, f"Total elements {total_elements} not divisible by batch size {batch_size}"
    time_dim = total_elements // batch_size
    new_shape = (batch_size, time_dim, state_dim)
    x_original = x_reshaped.view(new_shape)
    return x_original


# Test the utility functions with emphasis on time order preservation
batch_size, time_steps, old_dim = 3, 5, 4
new_dim = 6

# Create a sequential tensor to easily check time order
x = torch.Tensor([[[i for _ in range(old_dim)] for i in range(time_steps)] for _ in range(batch_size)])

# Convert 3D -> 2D
x_reshaped = batched_to_flat_trajectory(x)

# Simulate some transformation
transformation = torch.nn.Linear(old_dim, new_dim)
x_transformed = transformation(x_reshaped)

# Convert 2D -> 3D
x_reconstructed = flat_to_batched_trajectory(x_transformed, batch_size, new_dim)

# Check if time order is preserved
time_order_preserved = all(
    (x[i, j, 0] == x_reconstructed[i, j, 0]).item() for i in range(batch_size) for j in range(time_steps))

x, x_reconstructed, time_order_preserved
