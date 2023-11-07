import itertools
import math
import shutil
from pathlib import Path
from typing import Optional

import escnn
from lightning import seed_everything
from morpho_symm.utils.rep_theory_utils import isotypic_decomp_representation, isotypic_basis
import numpy as np
import scipy
from escnn.group import Representation, directsum
from tqdm import tqdm

from data.DynamicsRecording import DynamicsRecording
from utils.mysc import companion_matrix, random_orthogonal_matrix
from utils.linear_algebra import matrix_average_trick
from utils.plotting import plot_system_2D, plot_system_3D, plot_trajectories


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

    trials = 500
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

            trials -= 1
            if trials == 0:
                raise RuntimeError("Too constrained.")
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
        max_period = time_constant

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


def stable_equivariant_lin_dynamics(rep_X: Representation,
                                    time_constant=1,
                                    min_period=0.5,
                                    max_period=None,
                                    n_constraints: int = 0,
                                    multiplicity: int = 1):
    """ TODO """
    assert min_period > 0, f"Negative minimum period {min_period}"
    assert max_period is None or max_period > min_period, f"Negative maximum period {max_period}"
    if max_period is None:
        max_period = 2 * time_constant

    state_dim = rep_X.size * multiplicity
    iso_reps, iso_space_dims = isotypic_basis(representation=rep_X,
                                              multiplicity=multiplicity,
                                              prefix='State')

    # Trivial Isotypic subspace
    tr_iso_idx = list(iso_space_dims.get(G.trivial_representation.id, None))

    # Define the linear dynamical system as a function of smaller linear dynamical system each evolving an
    # Isotypic subspace of the observable state space.
    iso_space_dyn = []
    iso_space_constraints = []
    iso_constraint_id = []
    for isotypic_id, rep_iso in iso_reps.items():

        iso_state_dim = rep_iso.size
        A_iso = stable_lin_dynamics(rep_iso,
                                    time_constant=time_constant,
                                    stable_eigval_prob=0.5,
                                    min_period=min_period,
                                    max_period=max_period)
        # Enforce G-equivariance
        elements = G.elements if not G.continuous else G.testing_elements(len(G.representations))
        A_G_iso = matrix_average_trick(A_iso, [rep_iso(g) for g in elements])
        for g in elements:
            assert np.allclose(rep_iso(g) @ A_G_iso, A_G_iso @ rep_iso(g))
        iso_space_dyn.append(A_G_iso)

        # Add constraints to the Isotypic subspace enforcing the selected symmetry group.
        # Inequality hyperplane constraints are defined as P x >= z
        P_symm, z_constraint = None, None
        iso_irrep_id = rep_iso.irreps[0]
        if n_constraints > 0 and iso_irrep_id != G.trivial_representation.id and rep_iso.size > 1 :
            # Orthonormalize the column space of the constraint matrix, so hyperplanes are orthogonal to each other
            normal_planes = np.random.rand(iso_state_dim, n_constraints)
            normal_planes = np.round(normal_planes, 4)
            normal_planes = scipy.linalg.orth(normal_planes).T  # (n_constraints, iso_state_dim)
            # Ensure unique normal vectors
            normal_planes = np.unique(normal_planes, axis=0)
            for normal_plane in normal_planes:
                normal_orbit = np.vstack([np.linalg.det(rep_iso(g)) * (rep_iso(g) @ normal_plane) for g in G.elements])
                normal_orbit = np.round(normal_orbit, 7)
                # Ensure unique normal vectors
                normal_orbit = np.unique(normal_orbit, axis=0)
                # Fix point of linear systems is the origin
                offset_orbit = np.asarray([-np.random.uniform(0.1, 0.5)] * normal_orbit.shape[0])
                # Stack the symmetric hyperplane constraints into the list of constraints for this Isptypic subspace
                P_symm = np.vstack((P_symm, normal_orbit)) if P_symm is not None else normal_orbit
                z_constraint = np.concatenate((z_constraint, offset_orbit)) if z_constraint is not None else offset_orbit
        if P_symm is not None:
            iso_space_constraints.append((P_symm, z_constraint))
            iso_constraint_id.extend([isotypic_id] * P_symm.shape[0])
        else:  # Append zero constraints that will be removed later
            iso_space_constraints.append((np.zeros((1, iso_state_dim)), np.zeros(1)))
            iso_constraint_id.append(isotypic_id)

    A_G = scipy.linalg.block_diag(*iso_space_dyn)
    z_G = np.concatenate([z for _, z in iso_space_constraints]) if n_constraints > 0 else None
    # Stack constraints of Isotypic Spaces into a constraint matrix of shape (n_constraints, state_dim)
    P_G = scipy.linalg.block_diag(*[P for P, _ in iso_space_constraints]) if n_constraints > 0 else None
    if P_G is not None:
        # Remove zero constraints from P_G and Z_G. Get indices of zero row vectors in P_G and remove them
        zero_rows = np.where(~P_G.any(axis=1))[0]
        P_G = np.delete(P_G, zero_rows, axis=0)
        z_G = np.delete(z_G, zero_rows, axis=0)
        iso_constraint_id = [id for i, id in enumerate(iso_constraint_id) if i not in zero_rows]
        # Add component of the constraint plane along the trivial representations to break 180 deg rotational symmetry
        # And introduce a non-linearity in the dynamics that relates the different Isotypic subspaces.
        for constraint_id in set(iso_constraint_id):
            G_constraint_idx = np.where(np.asarray(iso_constraint_id) == constraint_id)[0]
            tr_plane_component = np.random.uniform(-1, 1, size=(1, len(tr_iso_idx)))
            P_G[np.ix_(G_constraint_idx, tr_iso_idx)] = tr_plane_component
        # Change constraints to the new basis

    # Apply an arbitrary change of basis to the system matrix, to lose the isotypic basis.
    Q = random_orthogonal_matrix(state_dim)
    A_G = Q @ A_G @ Q.T
    # P_G x >= z_G -> (P_G @ T^-1) T x >= z_G
    P_G = P_G @ Q.T if P_G is not None else None

    # Define the representation of the system in the new basis.
    rep_state_iso = directsum([iso_rep for iso_rep in iso_reps.values()],
                              name=f"{rep_X.name}-IsoBasis",
                              change_of_basis=Q)

    # Test commutativity / G-equivariance
    elements = G.elements if not G.continuous else G.testing_elements(len(G.representations))
    for g in elements:
        assert rep_state_iso(g).shape == (state_dim, state_dim), f"Invalid rep shape {rep_state_iso(g).shape}"
        assert np.allclose(rep_state_iso(g) @ A_G, A_G @ rep_state_iso(g)), \
            f"G-equiv err:{np.max((rep_state_iso(g) @ A_G) - (A_G @ rep_state_iso(g)))}"

    # Ensure stability
    eigvals = np.linalg.eigvals(A_G)
    # steady_state_eigvals = eigvals[np.isclose(eigvals, 0)]
    transient_eigvals = eigvals[np.logical_not(np.isclose(np.real(eigvals), np.zeros_like(eigvals)))]
    assert np.all(np.real(transient_eigvals) <= 0), f"Unstable eigenvalues: {eigvals}"

    # Compute the empirical time constant of the system:
    time_constants = [1 / np.abs(eigval) for eigval in transient_eigvals]
    fastest_time_constant = np.min(time_constants) if len(transient_eigvals) > 0 else np.inf
    # Compute the longest period of oscillation of the system:
    periods = [2 * np.pi / np.abs(np.imag(eigval)) for eigval in eigvals if not np.isinf(eigval)]
    fastest_period = np.min(periods)
    print(f"System has eigenvalues \n {eigvals}")
    print(f"Empirical periods MIN:{np.min(periods)}, MAX:{np.max(periods)}")
    if len(time_constants) > 0:
        print(f"Empirical time constants MIN:{np.min(time_constants)}, MAX:{np.max(time_constants)}")
    # print(f"Empirical time constants MIN:{np.min(time_constants)}, MAX:{np.max(time_constants)}")

    return A_G, P_G, z_G, rep_state_iso, fastest_period, fastest_time_constant


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


if __name__ == '__main__':
    np.set_printoptions(precision=3)
    seed_everything(120)
    order = 3
    subgroups_ids = dict(C2=('cone', 1),
                         Tetrahedral=('fulltetra',),
                         Octahedral=(True, 'octa',),
                         Icosahedral=(True, 'ico',),
                         Cyclic=(False, False, order),
                         Dihedral=(False, True, order),
                         SO2=(False, False, -1))

    # Select the group of the domain
    G_domain = escnn.group.O3(maximum_frequency=10)
    # Select the subgroup  of the dynamics of the system
    G_id = subgroups_ids['Cyclic']
    # G_id = subgroups_ids['Dihedral']
    G, g_dynamics_2_Gsub_domain, g_domain_2_g_dynamics = G_domain.subgroup(G_id)

    rep_X = G.regular_representation  # directsum([irrep] * multiplicity, name="State Space")  # + G.irrep(1)

    # Parameters of the state space.
    for n_constraints in [1]: # [0, 1]:
        # Define the state representation.
        for multiplicity in [2]:
            # Generate stable equivariant linear dynamics withing a range of fast and slow dynamics
            max_time_constant = 5  # [s] Maximum time constant of the system.
            min_period = max_time_constant / 3  # [s] Minimum period of oscillation of the fastest transient mode.
            max_period = max_time_constant * 2  # [s] Maximum period of oscillation of the slowest transient mode.

            A_G, P_G, z_G, rep_state, fastest_period, fastest_time_constant = stable_equivariant_lin_dynamics(
                rep_X,
                time_constant=max_time_constant,
                min_period=min_period,
                max_period=max_period,
                n_constraints=n_constraints,
                multiplicity=multiplicity)

            state_dim = rep_state.size
            # Fastest time constants determines the fastest transient dynamics of the system. We want to capture it.
            if np.isinf(fastest_time_constant):  # Stable system on limit cycle. no transient dynamics.
                T = fastest_period  # Simulate until the slowest stable mode has completed a full period.
            else:  # System has transient dynamics that vanish to 36.8% in fastest_time_constant seconds.
                T = max(6 * fastest_time_constant, fastest_period)  # Required time for this transient dynamics to vanish.
            dt = T * 0.005  # Sample time to obtain 200 samples per trajectory

            # Generate trajectories of the system dynamics
            n_trajs = 170
            trajs_per_noise_level = []
            for noise_level in tqdm(range(10), desc="noise level"):
                sigma = T * 0.0025 * noise_level
                state_trajs = []
                for _ in range(n_trajs):
                    # Sample initial condition
                    x0 = sample_initial_condition(state_dim, P_G, z_G)
                    t, state_traj = evolve_linear_dynamics(
                        A_G, x0, dt, T, sigma, constraint_matrix=P_G, constraint_offset=z_G)
                    state_trajs.append(state_traj)
                trajs_per_noise_level.append(np.asarray(state_trajs))

            for noise_level, state_trajs in tqdm(enumerate(trajs_per_noise_level), desc="saving recordings"):
                # Save the recordings to train test val splits
                path_2_system = Path(__file__).parents[1]
                assert path_2_system.exists(), f"Invalid Dataset path {path_2_system.absolute()}"
                path_2_system = (path_2_system / 'linear_system' / f"group={G.name}-dim={state_dim:d}" /
                                 f"n_constraints={n_constraints:d}" /
                                 f"f_time_constant={fastest_time_constant:.1f}[s]-frames={state_trajs.shape[1]:d}"
                                 f"-horizon={state_trajs.shape[1] * dt:.1f}[s]" /
                                 f"noise_level={noise_level:d}")
                if path_2_system.exists():
                    shutil.rmtree(path_2_system)

                path_2_system.mkdir(parents=True, exist_ok=True)

                assert len(state_trajs) == n_trajs, f"Invalid number of trajectories {len(state_traj)}"
                # Split the trajectories into train (70%) test (15%) and validation (15%) sets.
                train_idx = range(0, int(0.7 * n_trajs))
                val_idx = range(int(0.7 * n_trajs), int(0.85 * n_trajs))
                test_idx = range(int(0.85 * n_trajs), n_trajs)

                fig = None
                for partition, idx in zip(['val', 'test', 'train'], [val_idx, test_idx, train_idx]):
                    # Orbit of trajectories
                    G_trajs = state_trajs[idx]
                    # For validation and test sets, augment the trajectories to properly evaluate symmetry performance
                    if partition == 'val' or partition == 'test':
                        # Augment validation and test sets with all group elements
                        elements = G.elements if not G.continuous else G.testing_elements(len(G.representations))
                        # elements.remove(G.identity)  # Weird bug
                        for g in elements:
                            if g == G.identity: continue
                            g_trajs = np.einsum('ij, ...j -> ...i', rep_state(g), state_trajs[idx])
                            G_trajs = np.concatenate([G_trajs, g_trajs], axis=0)

                    n_trajs_part = G_trajs.shape[0]
                    traj_length = G_trajs.shape[1]
                    # Save DynamicsDataset
                    data = DynamicsRecording(
                        description="Stable linear system with stochastic additive noise",
                        info=dict(num_traj=n_trajs_part,
                                  trajectory_length=traj_length),
                        dynamics_parameters=dict(
                            transition_matrix=A_G,
                            constraint_matrix=P_G,
                            constraint_vector=z_G,
                            noise_std=sigma,
                            dt=dt,
                            time_constant=max_time_constant,
                            time_constant_dt_ratio=max_time_constant / dt,
                            n_constraints=n_constraints,
                            group=dict(subgroup_id=G_id, group_name=G.name, group_order=G.order()),
                            ),
                        state_obs=('state',),
                        obs_representations=dict(state=rep_state),
                        recordings=dict(state=np.asarray(G_trajs, dtype=np.float32)))

                    assert data.obs_dims['state'] == state_dim
                    path_to_file = path_2_system / f"n_trajs={n_trajs_part}-{partition}"

                    data.save_to_file(path_to_file)
                    # data2 = MarkovDynamicsRecording.load_from_file(path_to_file)

                    if state_dim == 2:
                        fig = plot_system_2D(G_trajs, P=P_G, z_constraint=z_G,
                                             num_trajs_to_show=-1, legendgroup=partition, fig=fig)
                    elif state_dim == 3:
                        colormap = {'train': 'Gray', 'val': 'Agsunset', 'test': 'Viridis'}
                        fig = plot_system_3D(A=A_G,
                                             trajectories=G_trajs,
                                             fig=fig,
                                             constraint_matrix=P_G,
                                             constraint_offset=z_G,
                                             traj_colorscale=colormap[partition],
                                             legendgroup=partition)
                    else:
                        colormap = {'train': 'Set3', 'val': 'Plotly', 'test': 'Dark2'}
                        if partition == 'test':
                            fig = plot_trajectories(trajs=G_trajs[:20],
                                                    fig=fig,
                                                    main_legend_label=partition,
                                                    colorscale=colormap[partition])

                    if fig is not None:
                        fig.write_html(path_2_system / 'test_trajectories.html')
                if noise_level == 2 and fig is not None:
                    fig.show()
    # fig.show()
    print(f"Recordings saved to {path_2_system}")
