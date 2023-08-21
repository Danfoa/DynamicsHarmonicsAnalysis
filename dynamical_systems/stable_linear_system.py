from collections import OrderedDict
from pathlib import Path
from typing import Optional

import chart_studio
import escnn
import numpy as np
import scipy
from escnn.group import Representation, directsum
import morpho_symm
from morpho_symm.groups.isotypic_decomposition import cplx_isotypic_decomposition

from data.dynamics_dataset import MarkovDataStructure
from utils.mysc import (companion_matrix, matrix_average_trick, random_orthogonal_matrix,
                        random_well_conditioned_invertible_matrix)
from utils.plotting import plot_system_2D, plot_system_3D


def sample_initial_condition(state_dim, P=None, z=None):
    x0 = np.random.uniform(-1, 1, size=state_dim)
    if P is not None:
        violation = P @ x0 < z
        is_constraint_violated = np.any(violation)
        while is_constraint_violated:
            # Take first violation and update the point.
            dim_violated = np.argwhere(violation).flatten()[0]
            normal_vect = P[dim_violated, :]
            # Get unitary normal plane to the constraint hyperplane
            normal_vect /= np.linalg.norm(normal_vect)
            # Project x onto the constraint hyperplane
            violation_vector = (x0 @ normal_vect) * normal_vect
            # Remove the projected motion from dx
            x0 = x0 - violation_vector

            violation = P @ x0 < z
            is_constraint_violated = np.any(violation)
    return x0


def stable_lin_dynamics(rep: Optional[Representation] = None, state_dim: int = -1,
                        stable_eigval_prob: float = 0.2, time_constant=1, min_period=0.5, max_period=None):
    """

    Args:
        state_dim: Dimension of the state of the linear system x[t+dt] = A x[t] + eta
        rep: Representation of the group of linear transformations of the state space.
        time_constant: Time constant of the slowest transient eigenvalue of A. Represents the decay rate of the
            slowest transient mode in seconds. This value implies the slowest transient dynamics vanishes to 36.8%
            of its maximum initial value after one time constant, and to 0.5% after 5 time constants.
        min_period: Determines the minimum period of oscillation/rotation of the fastest transient mode in seconds.
        max_period: Determines the maximum period of oscillation/rotation of the slowest transient mode in seconds.

    Returns:
    """
    assert min_period > 0, f"Negative minimum period {min_period}"
    assert max_period is None or max_period > min_period, f"Negative maximum period {max_period}"
    if max_period is None:
        max_period = 2 * time_constant

    irreps_space_dyn = []
    max_freq = (1 / min_period) / (2 * np.pi)
    min_freq = (1 / max_period) * (2 * np.pi)

    assert rep is not None or state_dim > 0, f"Either a representation or a state dimension must be provided"

    def sample_stable_eigval() -> complex:
        if np.random.uniform() < stable_eigval_prob:
            real_part = 0   # Stable eigenvalue
        else:
            real_part = -1 * 1 / np.random.uniform(time_constant, time_constant/5)
        imag_part = np.random.uniform(min_freq, max_freq)  # Rotations per second
        return complex(real_part, imag_part)

    if rep is not None:
        G = rep.group
        for re_irrep_id in rep.irreps:
            re_irrep = G.irrep(*re_irrep_id)
            type = re_irrep.type
            basis = re_irrep.endomorphism_basis()
            basis_dim = len(basis)

            unique_params = []
            if type == "R":  # Only Isomorphism are scalar multiple of the identity
                unique_params = [sample_stable_eigval().real for _ in range(basis_dim)]
            elif type == "C":  # Realification is achieved by setting [re(eig1), im(eig1), ...]
                for _ in range(basis_dim // 2):
                    eigval = sample_stable_eigval()
                    unique_params.extend([eigval.real, eigval.imag])
            elif type == "H":  # Realification is achieved by setting [re(eig1), im_i(eig1), im_j(eig1), im_k(eig1),...]
                unique_params = []
                for _ in range(basis_dim // 4):
                    unique_params.append(sample_stable_eigval().real)
                    unique_params.extend([sample_stable_eigval().imag for _ in range(3)])
            else:
                raise NotImplementedError(f"What is this representation type:{type}? Dunno.")
            basis_vects = [e * basis_mat for e, basis_mat in zip(unique_params, basis)]
            iso = np.sum(basis_vects, axis=0)  # Isomorphism
            irreps_space_dyn.append(iso)
    else:
        subspace_dims = [1] + [2] * ((state_dim - 1) // 2) if state_dim % 2 == 1 else [2] * (state_dim // 2)
        for subspace_dim in subspace_dims:
            # If n is odd, add one real eigenvalue with negative real part
            if subspace_dim == 1:
                irreps_space_dyn.append(np.diag([sample_stable_eigval().real]))
            if subspace_dim == 2:
                irreps_space_dyn.append(companion_matrix(sample_stable_eigval()))

    A = scipy.linalg.block_diag(*irreps_space_dyn)

    # Perform random well conditioned perturbation to proportionally lose the orthonormality of the basis of the
    # state space. This will make the system more difficult to learn. And replicate what is experienced in practise when
    # the assumption of orthogonality of our observables may not hold.
    # T = random_well_conditioned_invertible_matrix(state_dim, perturbation_scale=orth_scale)
    # Ap = T @ A @ np.linalg.inv(T)

    return A


def stable_equivariant_lin_dynamics(rep_X: Representation, time_constant=1, min_period=0.5, max_period=None):
    assert min_period > 0, f"Negative minimum period {min_period}"
    assert max_period is None or max_period > min_period, f"Negative maximum period {max_period}"
    if max_period is None:
        max_period = 2 * time_constant

    state_dim = rep_X.size
    rep_X, Q_iso = identify_isotypic_spaces(rep_X)

    isotypic_reps = rep_X.attributes['isotypic_reps']

    # Define the linear dynamical system as a function of smaller linear dynamical system each evolving an
    # Isotypic subspace of the observable state space.
    iso_space_dyn = []
    for isotypic_id, rep_iso in isotypic_reps.items():
        A_iso = stable_lin_dynamics(rep_iso,
                                    time_constant=5,
                                    stable_eigval_prob=1/(state_dim + 1),
                                    min_period=min_period,
                                    max_period=max_period)
        # Enforce G-equivariance
        A_G_iso = matrix_average_trick(A_iso, [rep_iso(g) for g in G.elements])

        # Test commutativity / G-equivariance
        for g in G.generators:
            assert np.allclose(rep_iso(g) @ A_G_iso, A_G_iso @ rep_iso(g))
        iso_space_dyn.append(A_G_iso)

    A_G = scipy.linalg.block_diag(*iso_space_dyn)

    # Apply an arbitrary change of basis to the system matrix, to lose the isotypic basis.
    T = random_orthogonal_matrix(state_dim)
    A_G = T @ A_G @ T.T
    rep_X = Representation(G, irreps=rep_X.irreps, name=rep_X.name, change_of_basis=T @ rep_X.change_of_basis)

    # Test commutativity / G-equivariance
    for g in G.generators:
        assert np.allclose(rep_X(g) @ A_G, A_G @ rep_X(g)), f"G-equiv err:{np.max((rep_X(g) @ A_G) - (A_G @ rep_X(g)))}"

    # Ensure stability
    eigvals = np.linalg.eigvals(A_G)
    # assert np.all(np.real(eigvals) < 0), f"Unstable eigenvalues: {eigvals}"
    return A_G, rep_X


def evolve_linear_dynamics(A, x0, dt, T, sigma=0.1, P=None, constraint_offset=None):
    """
    Evolve the stochastic `x[t+dt] = Ax[t] + eta` system using Euler-Maruyama. Cimplying with the linear
    constraints `P @ x[t] >= z`. Equivalent to imposing some Hyperplanes in the state space that cannot be crossed.

    Parameters:
    - A: System matrix.
    - z: RHS of the constraint.
    - x0: Initial condition.
    - dt: Time step.
    - T: Total simulation time.
    - sigma: Noise intensity.

    Returns:
    - trajectory: List of state vectors over time.
    """
    x_dim = x0.shape[0]
    assert A.shape == (x_dim, x_dim), f"System matrix A must be of shape ({x_dim}, {x_dim})"
    assert P is None or P.shape[-1] == x_dim, f"Constraint matrix P: {P.shape} must be of shape (ANY, {x_dim})"
    num_steps = int(T / dt) - 1
    t = np.linspace(0, T, num_steps)

    x = x0.copy()
    trajectory = [x0]

    for _ in range(num_steps):
        dx = (A @ x) + (sigma * np.sqrt(dt) * np.random.normal(size=x.shape))
        x_offset = (dx * dt)
        x_next = x + x_offset
        if P is not None:
            violation = P @ x_next < constraint_offset
            is_constraint_violated = np.any(violation)
            while is_constraint_violated:
                # Take first violation and update the point.
                dim_violated = np.argwhere(violation).flatten()[0]
                normal_vect = P[dim_violated, :]
                # Get unitary normal plane to the constraint hyperplane
                normal_vect /= np.linalg.norm(normal_vect)
                # Project the gradient to the hyperplane surface
                violation_grad = (x_offset @ normal_vect) * normal_vect
                x_offset -= violation_grad
                x_next = x + x_offset
                # Check for more violations
                violation = P @ x_next < constraint_offset
                is_constraint_violated = np.any(violation)
        x = x_next
        trajectory.append(x)

    return t, np.array(trajectory)


def find_combinations(target_dim, irreps_dims, idx=0):
    """
    # A = np.array([[0, 1], [-2, -1]])

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


def identify_isotypic_spaces(rep: Representation) -> OrderedDict[tuple: Representation]:
    """
    Identify the isotypic subspaces of a representation. See Isotypic Basis for more details (TODO).
    Args:
        rep (Representation): Input representation in any arbitrary basis.

    Returns: A `Representation` with a change of basis exposing an Isotypic Basis (a.k.a symmetry enabled basis).
        The instance of the representation contains an additional parameter `isotypic_subspaces` which is an
        `OrderedDict` of representations per each isotypic subspace. The keys are the active irreps' ids associated
        with each Isotypic subspace.
    """

    potential_irreps = rep.group.irreps()
    isotypic_subspaces_indices = {irrep.id: [] for irrep in potential_irreps}
    rep.irreps[0] == rep.irreps[0]
    for irrep in potential_irreps:
        for index, rep_irrep_id in enumerate(rep.irreps):
            if G.irrep(*rep_irrep_id) == irrep:
                isotypic_subspaces_indices[rep_irrep_id].append(index)

    # Remove inactive Isotypic Spaces
    for irrep in potential_irreps:
        if len(isotypic_subspaces_indices[irrep.id]) == 0:
            del isotypic_subspaces_indices[irrep.id]

    # Each Isotypic Space will be indexed by the irrep it is associated with.
    active_isotypic_reps = {}
    for irrep_id, indices in isotypic_subspaces_indices.items():
        # if indices are not consecutive numbers raise an error
        if not np.all(np.diff(indices) == 1):
            raise NotImplementedError("TODO: Add permutations needed to handle this case")
        irrep_dim = G.irrep(*irrep_id).size
        multiplicities = len(indices)
        active_isotypic_reps[irrep_id] = Representation(group=rep.group,
                                                        irreps=[irrep_id] * multiplicities,
                                                        name=f'IsoSubspace {irrep_id}',
                                                        change_of_basis=np.identity(irrep_dim * multiplicities))

    # Impose canonical order on the Isotypic Subspaces.
    # If the trivial representation is active it will be the first Isotypic Subspace.
    # Then sort by dimension of the space from smallest to largest.
    ordered_isotypic_reps = OrderedDict(sorted(active_isotypic_reps.items(), key=lambda item: item[1].size))
    if G.trivial_representation.id in ordered_isotypic_reps.keys():
        ordered_isotypic_reps.move_to_end(G.trivial_representation.id, last=False)

    # Compute the decomposition of Real Irreps into Complex Irreps and store this information in irrep.attributes
    # cplx_irrep_i(g) = Q_re2cplx @ re_irrep_i(g) @ Q_re2cplx^-1`
    for irrep_id in ordered_isotypic_reps.keys():
        re_irrep = G.irrep(*irrep_id)
        cplx_subreps, Q_re2cplx = cplx_isotypic_decomposition(G, re_irrep)
        re_irrep.is_cplx_irrep = len(cplx_subreps) == 1
        G.irrep(*irrep_id).attributes['cplx_irreps'] = cplx_subreps
        G.irrep(*irrep_id).attributes['Q_re2cplx'] = Q_re2cplx

    new_rep = directsum(list(ordered_isotypic_reps.values()),
                        name=rep.name + '-Iso',
                        change_of_basis=None)  # TODO: Check for additional permutations
    new_rep.supported_nonlinearities = rep.supported_nonlinearities
    new_rep.attributes['isotypic_reps'] = ordered_isotypic_reps
    return new_rep, rep.change_of_basis


if __name__ == '__main__':
    np.set_printoptions(precision=3)

    order = 4
    subgroups_ids = dict(C2=('cone', 1),
                         Tetrahedral=('fulltetra',),
                         Octahedral=(True, 'octa',),
                         Icosahedral=(True, 'ico',),
                         Cyclic=(False, False, order),
                         Dihedral=(False, True, order),
                         )

    # Select the group of the domain
    G_domain = escnn.group.O3()
    # Select the subgroup  of the dynamics of the system
    G_id = subgroups_ids['Cyclic']
    G, g_dynamics_2_Gsub_domain, g_domain_2_g_dynamics = G_domain.subgroup(G_id)

    # Define the state representation.
    # rep_X = G.regular_representation # + G.irrep(1)
    rep_X = G.irrep(1) +  G.irrep(0)

    # Generate stable equivariant linear dynamics withing a range of fast and slow dynamics
    state_dim = rep_X.size
    time_constant = 5  # [s] Maximum time constant of the system.
    A_G, rep_X = stable_equivariant_lin_dynamics(rep_X,
                                                 time_constant=time_constant,
                                                 min_period=time_constant/3,
                                                 max_period=time_constant*2)

    for _ in range(30):
        A_G, rep_X = stable_equivariant_lin_dynamics(rep_X,
                                                     time_constant=time_constant,
                                                     min_period=time_constant / 3,
                                                     max_period=time_constant * 2)
        eigvals = np.linalg.eigvals(A_G)
        re = np.real(eigvals)
        if len(re[re < 0]) > 0:
            t_constant = 1 / np.max(np.abs(re[re < 0]))
        else:
            t_constant = 'stable'
        print(f"Time constant: {t_constant}")

    # Generate hyperplanes that constraint outer region of space
    n_constraints = 3
    normal_planes = np.random.rand(state_dim, n_constraints)
    # Orthonormalize the column space of the constraint matrix, so hyperplanes are orthogonal to each other
    normal_planes = scipy.linalg.orth(normal_planes).T
    P_symm, offset = None, None
    for normal_plane in normal_planes:
        normal_orbit = np.vstack([np.linalg.det(rep_X(g)) * (rep_X(g) @ normal_plane) for g in G.elements])
        offset_orbit = np.asarray([-np.random.uniform(0.3, 1.0)] * normal_orbit.shape[0])
        P_symm = np.vstack((P_symm, normal_orbit)) if P_symm is not None else normal_orbit
        offset = np.concatenate((offset, offset_orbit)) if offset is not None else offset_orbit

    # Generate trajectories of the system dynamics
    dt = 0.01
    T = 4 * time_constant
    sigma = 5
    n_trajs = 30
    state_trajs = []
    for _ in range(n_trajs):
        # Sample initial condition
        x0 = sample_initial_condition(state_dim, P_symm, offset)
        t, state_traj = evolve_linear_dynamics(A_G, x0, dt, T, sigma, P=P_symm, constraint_offset=offset)
        state_trajs.append(state_traj)


    state_trajs = np.asarray(state_trajs)
    # plot_system_2D(A, state_traj, P=P, z=z_val)
    if state_dim == 2:
        fig = plot_system_2D(A_G, state_trajs, P=P_symm, z_constraint=offset)
    elif state_dim == 3:
        fig = plot_system_3D(A_G, state_trajs, P=P_symm, z_constraint=offset)
    fig.show()
    fig.write_html('test.html')
    chart_studio.tools.set_credentials_file(username='danfoa', api_key='YOUR_API_KEY')

    # # Save MarkovDataset
    # data = MarkovDataStructure(
    #     description="Stable linear system with stochastic additive noise",
    #     dynamics_parameters=dict(
    #         transition_matrix=A_G,
    #         constraint_matrix=P_symm,
    #         constraint_vector=offset,
    #         noise_std=sigma,
    #         dt=dt,
    #         time_constant=time_constant,
    #         ),
    #     measurements=dict(
    #         state=state_trajs[0].shape[-1]
    #         ),
    #     state_measurements=['state'],
    #     group_representations=dict(state=rep_X),
    #     measurements_representations=dict(state='state'),
    #     recordings=dict(
    #         state=state_trajs,
    #         )
    #     )
    #
    # path_to_data = Path(__file__).parents[1] / 'data'
    # assert path_to_data.exists(), f"Invalid Dataset path {path_to_data.absolute()}"
    # path_to_data = path_to_data / 'linear_systems'
    # path_to_data.mkdir(exist_ok=True)
    #
    # file_name = f"n_trajs={n_trajs}-noise_std={sigma}_time-constant={time_constant}_T-dt={T}s-{dt}s.zip"
    # path_to_file = path_to_data / file_name
    #
    # data.save_to_file(path_to_file)