from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
from escnn.group import Representation
from morpho_symm.utils.rep_theory_utils import irreps_stats, isotypic_decomp_representation
from morpho_symm.utils.robot_utils import load_symmetric_system
from pinocchio import pinocchio_pywrap as pin
from plotly.subplots import make_subplots

from data.DynamicsRecording import DynamicsRecording, get_train_test_val_file_paths


def decom_signal_into_isotypic_components(signal: np.ndarray, rep: Representation):
    rep_iso = isotypic_decomp_representation(rep)
    Q_iso2orig = rep_iso.change_of_basis  # Change of basis from isotypic basis to original basis
    Q_orig2iso = rep_iso.change_of_basis_inv  # Change of basis from original basis to isotypic basis
    assert signal.shape[-1] == rep.size, f"Expected signal shape to be (..., {rep.size}) got {signal.shape}"

    signal_iso = np.einsum("...ij,...j->...i", Q_orig2iso, signal)

    isotypic_representations = rep_iso.attributes["isotypic_reps"]

    # Compute the dimensions of each isotypic subspace
    cum_dim = 0
    iso_comp_dims = {}
    for irrep_id, iso_rep in isotypic_representations.items():
        iso_space_dims = range(cum_dim, cum_dim + iso_rep.size)
        iso_comp_dims[irrep_id] = iso_space_dims
        cum_dim += iso_rep.size

    # Separate the signal into isotypic components, by masking the signal outside of each isotypic subspace
    iso_comp_signals = OrderedDict()
    for irrep_id, _ in isotypic_representations.items():
        iso_dims = iso_comp_dims[irrep_id]
        iso_comp_signals[irrep_id] = signal_iso[..., iso_dims]

    iso_comp_signals_orig_basis = OrderedDict()
    # Compute the signals of each isotypic component in the original basis
    for irrep_id, _ in isotypic_representations.items():
        iso_dims = iso_comp_dims[irrep_id]
        Q_isocomp2orig = Q_iso2orig[:, iso_dims]  # Change of basis from isotypic component basis to original basis
        iso_comp_signals_orig_basis[irrep_id] = np.einsum(
            "...ij,...j->...i", Q_isocomp2orig, iso_comp_signals[irrep_id]
        )

    # Check that the sum of the isotypic components is equal to the original signal
    rec_signal = np.sum([iso_comp_signals_orig_basis[irrep_id] for irrep_id in isotypic_representations.keys()], axis=0)
    assert np.allclose(rec_signal, signal), (
        f"Reconstructed signal is not equal to original signal. Error: {np.linalg.norm(rec_signal - signal)}"
    )

    return iso_comp_signals, iso_comp_signals_orig_basis


if __name__ == "__main__":
    desired_font_size = 22
    xaxis_style = dict(
        titlefont=dict(size=desired_font_size),  # Set desired font size for x-axis title
        tickfont=dict(size=desired_font_size - 8),  # Set desired font size for x-axis
        showline=False,
        linewidth=0.1,
        gridcolor="rgba(0,0,0,0.01)",
    )
    yaxis_style = dict(
        titlefont=dict(size=desired_font_size - 8),  # Set desired font size for y-axis title
        tickfont=dict(size=desired_font_size - 8),  # Set desired font size for y-axis
        gridcolor="rgba(0,0,0,0.01)",
        showline=True,
        linewidth=0.1,
        linecolor="rgba(0,0,0,0.1)",
    )
    # Set legend box horizontal and vertical position
    legend_style = dict(
        x=1.0,
        y=1.0,
        font=dict(size=desired_font_size - 8),
        bgcolor="rgba(1.0,1.0,1.0,0.7)",  # Set background color to 0.4 alpha,
        orientation="v",
    )
    layout_config = dict(
        width=300,
        height=200,
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=10, r=10, t=10, b=10),
        legend=legend_style,
        xaxis=xaxis_style,
        yaxis=yaxis_style,
        autosize=False,
    )

    robot, G = load_symmetric_system(robot_name="mini_cheetah-k4")
    q0 = robot._q0

    rep_Qjs = G.representations["Q_js"]
    rep_TqQjs = G.representations["TqQ_js"]
    rep_R3 = G.representations["Rd"]
    rep_R3_pseudo = G.representations["Rd_pseudo"]
    rep_E3 = G.representations["Ed"]

    path_to_data = Path(__file__).parent
    assert path_to_data.exists(), f"Invalid Dataset path {path_to_data.absolute()}"

    recording_name = "concrete_difficult_slippery"
    # recordings = ['concrete_pronking', 'concrete_galloping', 'forest', 'rock_road', 'sidewalk',
    # 'concrete_difficult_slippery', 'sidewalk', 'air_jumping_gait', 'air_walking_gait']
    # recordings = ['concrete_pronking', 'forest', 'concrete_difficult_slippery', 'sidewalk', 'air_jumping_gait',
    # 'air_walking_gait']
    # recordings = ['concrete_pronking', 'forest']  # 'forest', 'rock_road', 'sidewalk', 'concrete_difficult_slippery']
    # recordings = ['concrete_pronking', 'forest', 'rock_road', 'sidewalk', 'concrete_difficult_slippery']
    # Get a list of all directories in path_to_data
    recordings = [a.name for a in (path_to_data / "recordings").iterdir() if a.is_dir()]

    df = None

    for recording_name in recordings:
        # Find all dynamic systems recordings
        record_path = path_to_data / Path("recordings") / recording_name
        # path_to_data = Path('/home/danfoa/Projects/koopman_robotics/data/linear_system/group=C3-dim=6/n_constraints
        # =1/')
        path_to_dyn_sys_data = set([a.parent for a in list(record_path.rglob("*test.pkl"))])
        # Select a dynamical system
        # Obtain the training, testing and validation file paths containing distinct trajectories of motion.
        train_data, test_data, val_data = get_train_test_val_file_paths(path_to_dyn_sys_data.pop())
        train_data = train_data[0]  # Get the first file path
        dyn_recording = DynamicsRecording.load_from_file(train_data)
        dt = np.array([0.002])  #  dyn_recording.dynamics_parameters['dt']

        time_window = range(dyn_recording.recordings["joint_pos"][0].shape[0])
        time_window = list(time_window)[:4000]
        # time_window = range(dyn_recording.recordings['joint_pos'][0].shape[0])
        joint_pos_t = dyn_recording.recordings["joint_pos"][0][time_window]
        joint_vel_t = dyn_recording.recordings["joint_vel"][0][time_window]
        joint_torques_t = dyn_recording.recordings["joint_torques"][0][time_window]

        # Transform from unit circle to joint angle
        joint_angle_t = np.reshape(joint_pos_t, (-1, 12, 2))
        joint_angle_t = np.arctan2(joint_angle_t[..., 1], joint_angle_t[..., 0])

        iso_irrep_ids, _, _ = irreps_stats(rep_TqQjs.irreps)
        iso_joint_angle_t_iso_basis, iso_joint_angle_t_ori_basis = decom_signal_into_isotypic_components(
            joint_angle_t, rep_TqQjs
        )
        iso_joint_vel_iso_basis, iso_joint_vel_ori_basis = decom_signal_into_isotypic_components(joint_vel_t, rep_TqQjs)
        iso_joint_torques_iso_basis, iso_joint_torques_ori_basis = decom_signal_into_isotypic_components(
            joint_torques_t, rep_TqQjs
        )

        iso_space_names = {
            iso_irrep_id: r"$\mathbb{G}^{(%d)}$" % i for i, iso_irrep_id in enumerate(iso_joint_vel_iso_basis.keys())
        }

        # Compute approximate work done by the torques on the system
        work_iso_decomp_t = {}  # Decomposition of the work into work done by isotypic components forces.
        for irrep_id, v_js_t_iso in iso_joint_vel_ori_basis.items():
            tau_js_t_iso = iso_joint_torques_ori_basis[irrep_id]
            work_iso_dims = v_js_t_iso * tau_js_t_iso
            work_iso_t = np.sum(v_js_t_iso * tau_js_t_iso, axis=-1)
            # This should be equivalent to the work computed in the iso basis itself.
            tau_js_t_iso_basis = iso_joint_torques_iso_basis[irrep_id]
            v_js_t_iso_basis = iso_joint_vel_iso_basis[irrep_id]
            work_iso_t_iso_basis = np.sum(v_js_t_iso_basis * tau_js_t_iso_basis, axis=-1)
            assert np.allclose(work_iso_t, work_iso_t_iso_basis)
            work_iso_decomp_t[iso_space_names[irrep_id]] = work_iso_t
        work_t = np.sum(joint_vel_t * joint_torques_t, axis=-1)
        # Ensure the sum of the Iso Work is equal to the total work
        sum_work_t = np.sum(list(work_iso_decomp_t.values()), axis=0)
        assert np.allclose(work_t, sum_work_t, atol=1e-3, rtol=1e-4), f"Error {np.max(np.abs(work_t - sum_work_t))}"

        # Create a data frame with the decomposition of the work into isotypic components
        work_dist_df = work_iso_decomp_t
        # work_dist_df['total'] = work_t
        work_dist_df["time"] = time_window * dt
        # for irrep_id in iso_irrep_ids:
        #     work_dist_df[iso_space_names[irrep_id]] /= work_t
        work_dist_df = pd.DataFrame.from_dict(work_dist_df)
        # Sort the columns of the dataframe by the total work done by each isotypic component
        ordered_iso_comps = sorted(work_dist_df.columns, key=lambda x: -np.linalg.norm(work_dist_df[x]))
        color_pallete = px.colors.qualitative.Dark2
        iso_comp_colors = {iso_comp: color_pallete[i] for i, iso_comp in enumerate(work_iso_decomp_t.keys())}

        work_df_long = work_dist_df.melt(
            id_vars=["time"], value_vars=ordered_iso_comps, var_name="Source", value_name="Work"
        )

        # Compute the total Kinetic Energy
        kin_energy_t = np.zeros(len(time_window))
        for t in range(len(time_window)):
            q_js = joint_pos_t[t]
            v_js = joint_vel_t[t]
            # Convert joint angle positions to unit circle format [cos(theta), sin(theta)]
            q = np.concatenate([q0[:7], q_js])
            v = np.concatenate([np.zeros(6), v_js])
            kin_energy = pin.computeKineticEnergy(robot.pinocchio_robot.model, robot.pinocchio_robot.data, q, v)
            kin_energy_t[t] = kin_energy

        # Compute the Kinetic Energy of the system and the kinetic energy of each isotypic component.
        kin_energy_iso_decomp_t = {}
        for irrep_id in iso_irrep_ids:
            q_js_t_iso = iso_joint_angle_t_ori_basis[irrep_id]
            v_js_t_iso = iso_joint_vel_ori_basis[irrep_id]
            kin_energy_iso_decomp_t[iso_space_names[irrep_id]] = np.zeros(q_js_t_iso.shape[0])
            for t in range(q_js_t_iso.shape[0]):
                # Get state and velocity at time t
                q_js_iso = q_js_t_iso[t]
                v_js_iso = v_js_t_iso[t]
                # Convert joint angle positions to unit circle format [cos(theta), sin(theta)]
                q_unit_circle = np.vstack([np.cos(q_js_iso), np.sin(q_js_iso)]).reshape(-1)
                q = np.concatenate([q0[:7], q_unit_circle])
                v = np.concatenate([np.zeros(6), v_js_iso])
                kin_energy = pin.computeKineticEnergy(robot.pinocchio_robot.model, robot.pinocchio_robot.data, q, v)
                kin_energy_iso_decomp_t[iso_space_names[irrep_id]][t] = kin_energy

        sum_kin_e_t = np.sum(list(kin_energy_iso_decomp_t.values()), axis=0)
        err = np.abs(kin_energy_t - sum_kin_e_t) / (kin_energy_t + 1e-9)

        trajs = np.array([e for k, e in kin_energy_iso_decomp_t.items()])
        trajs = np.expand_dims(trajs, axis=-1)
        max_energy = np.max(trajs)

        fig = make_subplots(rows=len(kin_energy_iso_decomp_t), cols=1, shared_xaxes=True)
        from plotly import graph_objects as go

        row_count = 1
        for iso_comp_id, energy_t in kin_energy_iso_decomp_t.items():
            # Add a line with an area filled between the line and 0
            fig.add_trace(
                go.Scatter(
                    x=np.asarray(time_window) * dt,
                    y=energy_t,
                    mode="lines",
                    line=dict(color=iso_comp_colors[iso_comp_id]),
                    name=iso_comp_id,
                    legendgroup=iso_comp_id,
                    fill="tozeroy",
                    fillcolor=iso_comp_colors[iso_comp_id],
                    # Use spline interpolation
                    line_shape="spline",
                    showlegend=True,
                ),
                row=row_count,
                col=1,
            )
            # Set the y axis label as the iso component name
            fig.update_yaxes(range=[0, max_energy * 1.1], nticks=2, row=row_count, col=1, **yaxis_style)
            row_count += 1
        fig.update_layout(
            layout_config,
            template="plotly_white",
        )
        # Ensure the y axis has only two ticks, the max energy and 0
        fig.update_xaxes(**xaxis_style, title="Time[s]", row=len(kin_energy_iso_decomp_t), col=1)
        # Set a unique y label for all subplots centered in the middle of the figure
        fig.update_yaxes(**yaxis_style, title="Kinetic Energy [J]", row=len(kin_energy_iso_decomp_t) - 1, col=1)

        base_path = Path(__file__).parent.parent.parent / "media" / "images" / "umich_dataset"

        file_path = base_path / f"{recording_name}_kin_energy_decomposition_{G.name}"
        fig.write_image(file_path.with_suffix(".svg"))
        print(f"Saved figure to {file_path}")
