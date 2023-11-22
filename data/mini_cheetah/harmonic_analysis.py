from pathlib import Path

import numpy as np
from escnn.group import Representation

from data.DynamicsRecording import DynamicsRecording, get_train_test_val_file_paths
import morpho_symm
from morpho_symm.utils.robot_utils import load_symmetric_system
from morpho_symm.utils.rep_theory_utils import isotypic_decomp_representation, irreps_stats

from utils.plotting import plot_trajectories


def decom_signal_into_isotypic_components(signal: np.ndarray, rep: Representation):
    rep_iso = isotypic_decomp_representation(rep)
    Q_iso2orig = rep_iso.change_of_basis  # Change of basis from isotypic basis to original basis
    Q_orig2iso = rep_iso.change_of_basis_inv  # Change of basis from original basis to isotypic basis
    assert signal.shape[-1] == rep.size, f"Expected signal shape to be (..., {rep.size}) got {signal.shape}"

    signal_iso = np.einsum('...ij,...j->...i', Q_orig2iso, signal)

    isotypic_representations = rep_iso.attributes['isotypic_reps']

    # Compute the dimensions of each isotypic subspace
    cum_dim = 0
    iso_comp_dims = {}
    for irrep_id, iso_rep in isotypic_representations.items():
        iso_space_dims = range(cum_dim, cum_dim + iso_rep.size)
        iso_comp_dims[irrep_id] = iso_space_dims
        cum_dim += iso_rep.size

    # Separate the signal into isotypic components, by masking the signal outside of each isotypic subspace
    iso_comp_signals = {irrep_id: signal_iso[..., iso_comp_dims[irrep_id]] for irrep_id in
                        isotypic_representations.keys()}
    iso_comp_signals_orig_basis = {}
    # Compute the signals of each isotypic component in the original basis
    for irrep_id, _ in isotypic_representations.items():
        iso_dims = iso_comp_dims[irrep_id]
        Q_isocomp2orig = Q_iso2orig[:, iso_dims]  # Change of basis from isotypic component basis to original basis
        iso_comp_signals_orig_basis[irrep_id] = np.einsum('...ij,...j->...i',
                                                          Q_isocomp2orig,
                                                          iso_comp_signals[irrep_id])

    # Check that the sum of the isotypic components is equal to the original signal
    rec_signal = np.sum([iso_comp_signals_orig_basis[irrep_id] for irrep_id in isotypic_representations.keys()], axis=0)
    assert np.allclose(rec_signal, signal), \
        f"Reconstructed signal is not equal to original signal. Error: {np.linalg.norm(rec_signal - signal)}"

    return iso_comp_signals, iso_comp_signals_orig_basis

def decom_signal_into_isotypic_components2(signal: np.ndarray, rep: Representation):
    G = rep.group
    Q_irr2orig = rep.change_of_basis  # Change of basis from isotypic basis to original basis
    Q_orig2irr = rep.change_of_basis_inv  # Change of basis from original basis to isotypic basis
    assert signal.shape[-1] == rep.size, f"Expected signal shape to be (..., {rep.size}) got {signal.shape}"

    signal_iso = np.einsum('...ij,...j->...i', Q_orig2irr, signal)

    iso_irreps_ids, counts, indices = irreps_stats(rep.irreps)
    # Compute the dimensions of each isotypic subspace
    cum_dim = 0
    iso_comp_dims = {irrep_id: [] for irrep_id in iso_irreps_ids}
    for irrep_id in rep.irreps:
        irrep = G.irrep(*irrep_id)
        iso_space_dims = list(range(cum_dim, cum_dim + irrep.size))
        iso_comp_dims[irrep_id].extend(iso_space_dims)
        cum_dim += irrep.size

    # Separate the signal into isotypic components, by masking the signal outside of each isotypic subspace
    iso_comp_signals = {irrep_id: signal_iso[..., iso_comp_dims[irrep_id]] for irrep_id in iso_irreps_ids}
    iso_comp_signals_orig_basis = {}
    # Compute the signals of each isotypic component in the original basis
    for irrep_id in iso_irreps_ids:
        iso_dims = iso_comp_dims[irrep_id]
        Q_isocomp2orig = Q_irr2orig[:, iso_dims]  # Change of basis from isotypic component basis to original basis
        iso_comp_signals_orig_basis[irrep_id] = np.einsum('...ij,...j->...i',
                                                          Q_isocomp2orig,
                                                          iso_comp_signals[irrep_id])

    # Check that the sum of the isotypic components is equal to the original signal
    rec_signal = np.sum([iso_comp_signals_orig_basis[irrep_id] for irrep_id in iso_irreps_ids], axis=0)
    assert np.allclose(rec_signal, signal), \
        f"Reconstructed signal is not equal to original signal. Error: {np.linalg.norm(rec_signal - signal)}"

    return iso_comp_signals, iso_comp_signals_orig_basis


if __name__ == "__main__":
    path_to_data = Path(__file__).parent
    assert path_to_data.exists(), f"Invalid Dataset path {path_to_data.absolute()}"
    recording_name = 'forest'
    # Find all dynamic systems recordings
    path_to_data /= Path('recordings') / recording_name
    # path_to_data = Path('/home/danfoa/Projects/koopman_robotics/data/linear_system/group=C3-dim=6/n_constraints=1/')
    path_to_dyn_sys_data = set([a.parent for a in list(path_to_data.rglob('*test.pkl'))])
    # Select a dynamical system
    # Obtain the training, testing and validation file paths containing distinct trajectories of motion.
    train_data, test_data, val_data = get_train_test_val_file_paths(path_to_dyn_sys_data.pop())
    train_data = train_data[0]  # Get the first file path
    dyn_recording = DynamicsRecording.load_from_file(train_data)

    robot, G = load_symmetric_system(robot_name='mini_cheetah-c2')
    q0 = robot._q0

    rep_Qjs = G.representations['Q_js']
    rep_TqQjs = G.representations['TqQ_js']
    rep_R3 = G.representations['Rd']
    rep_R3_pseudo = G.representations['Rd_pseudo']
    rep_E3 = G.representations['Ed']

    time_window = range(50, 550)
    q_js_t_unit_circle = dyn_recording.recordings['q_js'][0][time_window]
    q_js_t = np.reshape(q_js_t_unit_circle, (-1, 12, 2))
    v_js_t = dyn_recording.recordings['v_js'][0][time_window]
    tau_js_t = dyn_recording.recordings['torques'][0][time_window]

    # ensure the cos(theta) and sin(theta) are are in fact in the unit circle
    norm = np.linalg.norm(q_js_t, axis=-1)
    assert np.allclose(norm, 1), f"Reshape probably took the wrong order of the axes."
    q_js_t = np.arctan2(q_js_t[..., 1], q_js_t[..., 0])

    iso_irrep_ids, _, _ = irreps_stats(rep_TqQjs.irreps)
    q_js_t_iso_signal, q_js_t_iso_comps = decom_signal_into_isotypic_components(q_js_t, rep_TqQjs)
    v_js_t_iso_signal, v_js_t_iso_comps = decom_signal_into_isotypic_components(v_js_t, rep_TqQjs)
    tau_js_t_iso_signal, tau_js_t_iso_comps = decom_signal_into_isotypic_components(tau_js_t, rep_TqQjs)

    # Compute approximate work done by the torques on the system
    work_t = np.sum(v_js_t * tau_js_t, axis=-1)
    work_iso_decomp_t = {}  # Decomposition of the work into work done by isotypic components forces.
    for irrep_id, v_js_t_iso in v_js_t_iso_signal.items():
        tau_js_t_iso = tau_js_t_iso_signal[irrep_id]
        work_iso_dims = v_js_t_iso * tau_js_t_iso
        work_iso_t = np.sum(v_js_t_iso * tau_js_t_iso, axis=-1)
        work_iso_decomp_t[irrep_id] = work_iso_t
        # Check that computing the work in the original basis is equivalent to computing the work in the isotypic basis
        # v_js_t_iso_comp = v_js_t_iso_comps[irrep_id]
        # tau_js_t_iso_comp = tau_js_t_iso_comps[irrep_id]
        # assert np.allclose(work_iso_t, np.sum(v_js_t_iso_comp * tau_js_t_iso_comp, axis=-1))

    work_iso_trajs = np.concatenate([work_iso[None, :, None] for work_iso in work_iso_decomp_t.values()], axis=0)
    
    fig = plot_trajectories(trajs=work_iso_trajs,
                            # secondary_trajs=work[None, :, None],
                            main_legend_label="Work_Iso",
                            colorscale='G10',
                            plot_error=False)

    # fig = plot_trajectories(trajs=work_t[None, :, None],
    #                         main_legend_label="Work",
    #                         colorscale='Set2',
    #                         plot_error=False,
    #                         fig=fig)

    # Compute the Kinetic Energy of the system and the kinetic energy of each isotypic component.
    from pinocchio import pinocchio_pywrap as pin
    kin_energy_iso_t = {}
    for irrep_id in iso_irrep_ids:
        q_js_t_iso = q_js_t_iso_comps[irrep_id]
        v_js_t_iso = v_js_t_iso_comps[irrep_id]
        kin_energy_iso_t[irrep_id] = np.empty(q_js_t_iso.shape[0])
        for t in range(q_js_t_iso.shape[0]):
            # Get state and velocity at time t
            q_js_iso = q_js_t_iso[t]
            v_js_iso = v_js_t_iso[t]
            # Convert joint angle positions to unit circle format [cos(theta), sin(theta)]
            q_unit_circle = np.vstack([np.cos(q_js_iso), np.sin(q_js_iso)]).reshape(-1)
            q = np.concatenate([q0[:7], q_unit_circle])
            v = np.concatenate([np.zeros(6), v_js_iso])
            kin_energy = pin.computeKineticEnergy(robot.pinocchio_robot.model,
                                                  robot.pinocchio_robot.data,
                                                  q,
                                                  v)
            kin_energy_iso_t[irrep_id][t] = kin_energy

    import plotly.express as px
    df = px.data.gapminder()

    kin_energy_iso_trajs = np.concatenate([kin_energy_iso[None, :, None] for kin_energy_iso in kin_energy_iso_t.values()], axis=0)
    # fig = plot_trajectories(trajs=kin_energy_iso_trajs,
    #                         # secondary_trajs=work[None, :, None],
    #                         main_legend_label="Work_Iso",
    #                         colorscale='G10',
    #                         plot_error=False)

    # fig = plot_trajectories(trajs=work_t,
    #                         main_legend_label="Work",
    #                         colorscale='Set2',
    #                         plot_error=False,
    #                         fig=fig)

    # fig = plot_trajectories(trajs=work_iso_trajs,
    #                         # secondary_trajs=work[None, :, None],
    #                         main_legend_label="Work_Iso",
    #                         colorscale='G10',
    #                         plot_error=False)
    #
    # fig = plot_trajectories(trajs=work_t,
    #                         main_legend_label="Work",
    #                         colorscale='Set2',
    #                         plot_error=False,
    #                         fig=fig)

    fig.show()
    # Plot the data
