from pathlib import Path
from typing import Optional

import escnn
import numpy as np
import scipy
from escnn.group import Representation

from data.DynamicsRecording import DynamicsRecording
from utils.mysc import companion_matrix, matrix_average_trick, random_orthogonal_matrix
from utils.plotting import plot_system_2D, plot_system_3D
from utils.representation_theory import identify_isotypic_spaces


def sample_initial_condition(state_dim, P=None, z=None):
    """."""
    MIN_DISTANCE_FROM_ORIGIN = 0.4

    direction = points = np.random.randn(state_dim)
    direction = direction / np.linalg.norm(points)
    # We want to sample initial conditions away from the zero, as we want to capture the transient dynamics in our
    # data. We use a Beta distribution to sample distance from the origin.
    alpha, beta = 6, 1  #
    distance_from_origin = np.random.beta(alpha, beta)  # P(x > 0.5) = 0.98 for alpha=6, beta=1
    distance_from_origin = max(distance_from_origin, MIN_DISTANCE_FROM_ORIGIN)  # Truncate unlikely low values
    x0 = distance_from_origin * direction

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
        if np.linalg.norm(x0) < MIN_DISTANCE_FROM_ORIGIN:  # If sample is too close to zero ignore it.
            x0 = sample_initial_condition(state_dim, P=P, z=z)
    return x0


def stable_lin_dynamics(rep: Optional[Representation] = None,
                        state_dim: int = -1,
                        stable_eigval_prob: float = 0.2,
                        time_constant=1,
                        min_period=0.5,
                        max_period=None):
    """Generates a stable equivariant linear dynamical within a range of stable and transient dynamics.

    Args:
    ----
    rep: Representation of the group of linear transformations of the state space.
    state_dim: Dimension of the state of the linear system x[t+dt] = A x[t] + eta
    stable_eigval_prob: Probability of an eigenspace of the system being a steady state mode (i.e. re(eigval)=0)
    time_constant: Time constant of the slowest transient eigenvalue of A. Represents the decay rate of the
        slowest transient mode in seconds. This value implies the slowest transient dynamics vanishes to 36.8%
    min_period: Determines the minimum period of oscillation/rotation of the fastest transient mode in seconds.
    max_period: Determines the maximum period of oscillation/rotation of the slowest transient mode in seconds.

    Returns:
    -------
        A (np.ndarray): System matrix of shape (state_dim, state_dim).
    """
    assert min_period > 0, f"Negative minimum period {min_period}"
    assert max_period is None or max_period > min_period, f"Negative maximum period {max_period}"
    if max_period is None:
        max_period = 2 * time_constant

    irreps_space_dyn = []
    max_freq = (1 / min_period) / (2 * np.pi)
    min_freq = (1 / max_period) * (2 * np.pi)

    assert rep is not None or state_dim > 0, "Either a representation or a state dimension must be provided"

    def sample_stable_eigval() -> complex:
        if np.random.uniform() < stable_eigval_prob:
            real_part = 0  # Stable eigenvalue
        else:
            real_part = -1 * 1 / np.random.uniform(time_constant, time_constant / 5)
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
    """ TODO """
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
                                    stable_eigval_prob=1 / (state_dim + 1),
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
    # steady_state_eigvals = eigvals[np.isclose(eigvals, 0)]
    transient_eigvals = eigvals[np.logical_not(np.isclose(eigvals, np.zeros_like(eigvals)))]
    assert np.all(np.real(transient_eigvals) <= 0), f"Unstable eigenvalues: {eigvals}"
    # Compute the empirical time constant of the system:
    time_constants = [1 / np.abs(eigval) for eigval in transient_eigvals]
    slowest_time_constant = np.max(time_constants) if len(transient_eigvals) > 0 else np.inf
    print(f"Empirical time constants MIN:{np.min(time_constants)}, MAX:{np.max(time_constants)}")
    return A_G, rep_X, slowest_time_constant


def evolve_linear_dynamics(A: np.ndarray, init_state: np.ndarray, dt: float, sim_time: float, noise_std=0.1,
                           constraint_matrix=None, constraint_offset=None):
    """Evolve a stochastic linear system `dx/dt = A x[t] + eta` with linear inequality state constraints `Px[t] >=z`.

    Args:
    ----
    A (np.ndarray): System matrix of shape (state_dim, state_dim)
    init_state (np.ndarray): Initial state of shape (state_dim, )
    dt (float): Time step of the simulation
    sim_time (float): Total simulation time
    noise_std (float): Standard deviation of the additive Gaussian noise
    constraint_matrix (np.ndarray): Matrix of shape (n_constraints, state_dim) defining the hyperplanes normal vects
    constraint_offset (np.ndarray): of shape (n_constraints,) defining the hyperplanes offset from the origin

    Returns:
    -------
        t (np.ndarray): Time vector of shape (num_steps, )
        trajectory (np.ndarray): Trajectory of the system of shape (num_steps, state_dim)
    """
    x_dim = init_state.shape[0]
    assert A.shape == (x_dim, x_dim), f"System matrix A must be of shape ({x_dim}, {x_dim})"
    assert constraint_matrix is None or constraint_matrix.shape[
        -1] == x_dim, f"Constraint matrix P: {constraint_matrix.shape} must be of shape (ANY, {x_dim})"
    num_steps = int(sim_time / dt) - 1
    t = np.linspace(0, sim_time, num_steps)

    x = init_state.copy()
    trajectory = [init_state]

    for _ in range(num_steps):
        dx = (A @ x) + (noise_std * np.random.normal(size=x.shape))
        x_offset = (dx * dt)
        x_next = x + x_offset
        if constraint_matrix is not None:
            violation = constraint_matrix @ x_next < constraint_offset
            is_constraint_violated = np.any(violation)
            while is_constraint_violated:
                # Take first violation and update the point.
                dim_violated = np.argwhere(violation).flatten()[0]
                normal_vect = constraint_matrix[dim_violated, :]
                # Get unitary normal plane to the constraint hyperplane
                normal_vect /= np.linalg.norm(normal_vect)
                # Project the gradient to the hyperplane surface
                violation_grad = (x_offset @ normal_vect) * normal_vect
                x_offset -= violation_grad
                x_next = x + x_offset
                # Check for more violations
                violation = constraint_matrix @ x_next < constraint_offset
                is_constraint_violated = np.any(violation)
        x = x_next
        trajectory.append(x)

    return t, np.array(trajectory)

# def gen_recordings_stable_equiv_lin_system():
#     pass

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
    rep_X = G.irrep(1) + G.irrep(0)

    # Generate stable equivariant linear dynamics withing a range of fast and slow dynamics
    state_dim = rep_X.size
    max_time_constant = 5  # [s] Maximum time constant of the system.
    A_G, rep_X, time_constant = stable_equivariant_lin_dynamics(rep_X,
                                                                time_constant=max_time_constant,
                                                                min_period=max_time_constant / 3,
                                                                max_period=max_time_constant * 2)
    time_constant = max_time_constant * 2 if np.isinf(time_constant) else time_constant
    # Generate hyperplanes that constraint outer region of space
    n_constraints = 2
    P_symm, offset = None, None
    if n_constraints > 0:
        normal_planes = np.random.rand(state_dim, n_constraints)
        # Orthonormalize the column space of the constraint matrix, so hyperplanes are orthogonal to each other
        normal_planes = scipy.linalg.orth(normal_planes).T
        for normal_plane in normal_planes:
            normal_orbit = np.vstack([np.linalg.det(rep_X(g)) * (rep_X(g) @ normal_plane) for g in G.elements])
            offset_orbit = np.asarray([-np.random.uniform(0.0, 0.8)] * normal_orbit.shape[0])
            P_symm = np.vstack((P_symm, normal_orbit)) if P_symm is not None else normal_orbit
            offset = np.concatenate((offset, offset_orbit)) if offset is not None else offset_orbit

    # Generate trajectories of the system dynamics
    dt = 0.01
    T = 4 * time_constant
    sigma = 0.5
    n_trajs = 100
    state_trajs = []
    for _ in range(n_trajs):
        # Sample initial condition
        x0 = sample_initial_condition(state_dim, P_symm, offset)
        t, state_traj = evolve_linear_dynamics(A_G, x0, dt, T, sigma, constraint_matrix=P_symm,
                                               constraint_offset=offset)
        state_trajs.append(state_traj)

    state_trajs = np.asarray(state_trajs)

    # Save the recordings to train test val splits
    path_2_system = Path(__file__).parents[1] / 'data'
    assert path_2_system.exists(), f"Invalid Dataset path {path_2_system.absolute()}"
    path_2_system = (path_2_system / 'linear_systems' / f"{state_dim:d}-dim" / f"{G.name}" /
                     f"{n_constraints:d}-constraints" /
                     f"noise_std={sigma}_time-constant={max_time_constant:.1f}_T-dt={T:.1f}s-{dt:.3f}s")
    if path_2_system.exists():
        import shutil

        shutil.rmtree(path_2_system)
    path_2_system.mkdir(parents=True, exist_ok=True)

    # Split the trajectories into train (70%) test (15%) and validation (15%) sets.
    train_idx = range(0, int(0.7 * n_trajs))
    val_idx = range(int(0.7 * n_trajs), int(0.85 * n_trajs))
    test_idx = range(int(0.85 * n_trajs), n_trajs)

    for partition, idx in zip(['train', 'val', 'test'], [train_idx, val_idx, test_idx]):
        # Save DynamicsDataset
        data = DynamicsRecording(
            description="Stable linear system with stochastic additive noise",
            dynamics_parameters=dict(
                transition_matrix=A_G,
                constraint_matrix=P_symm,
                constraint_vector=offset,
                noise_std=sigma,
                dt=dt,
                time_constant=max_time_constant,
                group=dict(subgroup_id=G_id),
                ),
            measurements=dict(state=state_trajs[0].shape[-1]),
            state_measurements=['state'],
            reps_irreps=dict(state=rep_X.irreps),  # Store the irreps composing the measurements representations
            reps_change_of_basis=dict(state=rep_X.change_of_basis),  # Store the change of basis matrices
            measurements_representations=dict(state='state'),
            recordings=dict(state=state_trajs[idx]))

        path_to_file = path_2_system / f"n_trajs={len(state_trajs[idx])}-{partition}"

        data.save_to_file(path_to_file)
        # data2 = MarkovDynamicsRecording.load_from_file(path_to_file)

    if state_dim == 2:
        fig = plot_system_2D(A_G, state_trajs[test_idx], P=P_symm, z_constraint=offset)
    elif state_dim == 3:
        fig = plot_system_3D(A_G, state_trajs[test_idx], P=P_symm, z_constraint=offset)

    fig.write_html(path_2_system / 'test_trajectories.html')

    print(f"Recordings saved to {path_2_system}")