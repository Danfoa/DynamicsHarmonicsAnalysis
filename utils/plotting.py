import copy
from collections.abc import Iterable
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.figure import Figure
from plotly import figure_factory as ff, graph_objects as go
from plotly.subplots import make_subplots

from utils.mysc import best_rectangular_grid


def plot_system_2D(A, trajectories: Union[list, np.ndarray], P=None, z_constraint=None):
    trajs = np.asarray(trajectories)
    if trajs.ndim == 2:
        trajs = np.expand_dims(trajs, axis=0)

    x_min, x_max = np.min(trajs[:, :, 0]), np.max(trajs[:, :, 0])
    y_min, y_max = np.min(trajs[:, :, 1]), np.max(trajs[:, :, 1])

    bound = max(np.abs(max(y_max, x_max)), np.abs(min(y_min, x_min)))
    traj_len = trajs.shape[1]
    x = np.linspace(x_min, x_max, 20)
    y = np.linspace(y_min, y_max, 20)

    X, Y = np.meshgrid(x, y)
    UV = np.array([A @ np.array([xi, yi]) for xi, yi in zip(X.ravel(), Y.ravel())])
    U = UV[:, 0].reshape(X.shape)
    V = UV[:, 1].reshape(Y.shape)
    # Normalize U and V for adjustable arrow size based on desired maximum length
    norm = np.sqrt(U ** 2 + V ** 2)
    max_length = 0.05 * bound
    scale_factor = max_length / np.max(norm)
    U = (U / (norm + 1e-10)) * scale_factor
    V = (V / (norm + 1e-10)) * scale_factor

    # Gradient field
    fig = ff.create_quiver(X, Y, U, V,
                           name='Deterministic dynamics',
                           opacity=0.3,
                           line_width=1)

    # Hyperplanes
    # Constraint hyperplanes
    if P is not None:
        # Hyperplanes
        for i, row in enumerate(P):
            # y(x) = mx + b
            m = -row[0] / row[1] if row[1] != 0 else 0
            b = (z_constraint[i] / row[1]) if row[1] != 0 else 0
            y_constraint = lambda x: float(m * x + b)
            a = [[-bound, y_constraint(-bound)], [bound, y_constraint(bound)]]
            constraint_line = np.array([[-bound, y_constraint(-bound)], [bound, y_constraint(bound)]])
            # Plot the constraint line in red
            fig.add_trace(go.Scatter(x=constraint_line[:, 0], y=constraint_line[:, 1], mode='lines',
                                     line=dict(color='red'), name=f'constraint:{i}'))

    # Trajectories
    for traj in trajs:
        # Create a gradient based on the progression of time
        alpha_scale = np.linspace(0.0, 1, traj_len)  # Values from 0.1 (10% opacity) to 1 (100% opacity)
        # Define the colormap with a gradient in alpha
        fig.add_trace(go.Scatter(x=traj[:, 0], y=traj[:, 1], mode='markers+lines', opacity=0.8, showlegend=False,
                                 marker=dict(color=alpha_scale, size=2, colorscale='Viridis')))
    # Make plot square and axis equal
    fig.update_layout(
        plot_bgcolor='rgba(245, 245, 245, 1)',
        paper_bgcolor='rgba(245, 245, 245, 1)',
        xaxis=dict(
            range=[-bound * 1.1, bound * 1.1],
            scaleratio=1,
            scaleanchor="y"
            ),
        yaxis=dict(
            range=[-bound * 1.1, bound * 1.1],
            scaleratio=1
            )
        )
    return fig


def plot_system_3D(trajectories, secondary_trajectories=None, A=None, constraint_matrix=None, constraint_offset=None, fig=None, initial_call=False,
                   flow_field_colorscale='Blues', traj_colorscale='Viridis',
                   init_state_color='red', initial_point_radius=3, title=''):
    trajs = np.asarray(trajectories)
    assert trajs.shape[-1] == 3, "Trajectories must be 3D"
    if trajs.ndim == 2:  # Add a trajectory index dimension if it is missing
        trajs = np.expand_dims(trajs, axis=0)

    # Compute bounds of the plot
    # assert trajs.shape[-1] == A.shape[0]
    x_min, x_max = np.min(trajs[:, :, 0]), np.max(trajs[:, :, 0])
    y_min, y_max = np.min(trajs[:, :, 1]), np.max(trajs[:, :, 1])
    z_min, z_max = np.min(trajs[:, :, 2]), np.max(trajs[:, :, 2])
    bound = max(abs(x_max), abs(x_min), abs(y_max), abs(y_min), abs(z_max), abs(z_min))
    traj_len = trajs.shape[1]

    if fig is None:
        initial_call = True
        fig = go.Figure()

    if initial_call:
        if A is not None:
            x = np.linspace(-bound, bound, 10)
            y = np.linspace(-bound, bound, 10)
            z = np.linspace(-bound, bound, 10)

            X, Y, Z = np.meshgrid(x, y, z)
            UVW = np.array([A @ np.array([xi, yi, zi]) for xi, yi, zi in zip(X.ravel(), Y.ravel(), Z.ravel())])
            U = UVW[:, 0].reshape(X.shape)
            V = UVW[:, 1].reshape(Y.shape)
            W = UVW[:, 2].reshape(Z.shape)
            scale_factor = 0.5 * bound / np.max(np.sqrt(U ** 2 + V ** 2 + W ** 2))
            U *= scale_factor
            V *= scale_factor
            W *= scale_factor
            # Gradient field (quiver)
            fig.add_trace(go.Cone(x=X.ravel(), y=Y.ravel(), z=Z.ravel(),
                                  u=U.ravel(), v=V.ravel(), w=W.ravel(), opacity=0.2,
                                  sizemode="absolute", colorscale=flow_field_colorscale, showscale=False))
        # Constraint hyperplanes
        if constraint_matrix is not None:
            for i, row in enumerate(constraint_matrix):
                if row[2] != 0:  # Ensure z-coefficient isn't zero
                    z_coord = lambda x, y: (-row[0] * x - row[1] * y + constraint_offset[i]) / row[2] if row[
                                                                                                             2] != 0 \
                        else 0
                    lower_left_coord = [-bound, -bound, z_coord(-bound, -bound)]
                    lower_right_coord = [bound, -bound, z_coord(bound, -bound)]
                    upper_left_coord = [-bound, bound, z_coord(-bound, bound)]
                    upper_right_coord = [bound, bound, z_coord(bound, bound)]
                    plane_corners = np.array([lower_left_coord, lower_right_coord, upper_right_coord, upper_left_coord])
                    fig.add_trace(go.Mesh3d(x=plane_corners[:, 0], y=plane_corners[:, 1],
                                            z=plane_corners[:, 2], opacity=0.2))

    # Trajectory plotting (for both initial and subsequent calls)
    for traj_num, traj in enumerate(trajs):
        alpha_scale = np.linspace(0.3, 1, traj_len)
        fig.add_trace(go.Scatter3d(x=traj[:, 0], y=traj[:, 1], z=traj[:, 2], mode='lines', opacity=0.5,
                                   showlegend=False, name=f'traj{traj_num}',
                                   line=dict(color=alpha_scale, colorscale=traj_colorscale, width=6, )))

        # Initial point with customizable color and radius
        fig.add_trace(go.Scatter3d(x=[traj[0, 0]], y=[traj[0, 1]], z=[traj[0, 2]], mode='markers', showlegend=False,
                                   marker=dict(size=initial_point_radius, color=init_state_color)))

        if secondary_trajectories is not None:
            traj = secondary_trajectories[traj_num]
            alpha_scale = np.linspace(0.3, 1, traj_len)
            fig.add_trace(go.Scatter3d(x=traj[:, 0], y=traj[:, 1], z=traj[:, 2], mode='lines+markers', opacity=0.2,
                                       showlegend=False, name=f'pred{traj_num}',
                                       line=dict(color=alpha_scale, colorscale=traj_colorscale, width=6),
                                       marker=dict(size=initial_point_radius/2)))

            # Initial point with customizable color and radius
            fig.add_trace(go.Scatter3d(x=[traj[0, 0]], y=[traj[0, 1]], z=[traj[0, 2]], mode='markers', showlegend=False,
                                       opacity=0.5, marker=dict(size=initial_point_radius, color=init_state_color)))


    # Layout
    # current_bound = fig.layout.scene.xaxis.range[1]
    if initial_call or (bound * 1.1 > fig.layout.scene.xaxis.range[1]):
        # Update layout only if the new bound is larger than the current bound
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis=dict(range=[-bound * 1.1, bound * 1.1]),
                yaxis=dict(range=[-bound * 1.1, bound * 1.1]),
                zaxis=dict(range=[-bound * 1.1, bound * 1.1]),
                aspectratio=dict(x=1, y=1, z=1)
                )
            )
    return fig


def rgba(color, alpha):
    """
    Convert a hex color code to RGBA format with a specified alpha (opacity).

    Parameters:
    - color (str): Hex color string e.g., '#FF00FF'.
    - alpha (float): Opacity value between 0 and 1.

    Returns:
    - str: RGBA color string e.g., 'rgba(255,0,255,0.5)'.
    """
    color = color.lstrip('#')
    r, g, b = tuple(int(color[i:i + 2], 16) for i in (0, 2, 4))
    return f'rgba({r},{g},{b},{alpha})'


from plotly import colors


# Integrating the aggregated error calculation into the latest version of plot_trajectories function
def plot_trajectories(trajs, secondary_trajs=None, fig=None, colorscale='Prism',
                      main_style=None, secondary_style=None, dt=1, dim_names=None,
                      main_legend_label='traj', secondary_legend_label=None, shade_area=False,
                      n_trajs_to_show=5, plot_error=True, col_shift=0, show_legend=True):
    """
    Updated plot_trajectories function with error calculation and limited number of trajectories to display.
    """

    plot_error = plot_error and secondary_trajs is not None

    # Validate the input array shape
    if len(trajs.shape) != 3:
        raise ValueError("Input array should have shape (num_trajs, time, state_dimension).")

    num_trajs, time, state_dimension = trajs.shape
    n_trajs_to_show = min(num_trajs, n_trajs_to_show)
    num_rows = int(np.ceil(np.sqrt(state_dimension + plot_error)))  # Add 1 for the error plot
    num_cols = int(np.ceil((state_dimension + plot_error) / num_rows))

    # Set default dimensions names if not provided. If str is provided set {str}_{dim_idx} as the name
    if dim_names is None:
        dim_names = [f"{i + 1}" for i in range(state_dimension)]
    if isinstance(dim_names, str):
        dim_names = [f"{dim_names}_{i}" for i in range(state_dimension)]

    # Create a new figure if none is provided
    if fig is None:
        fig = make_subplots(rows=num_rows, cols=num_cols,
                            subplot_titles=dim_names + (["MSE"] if plot_error else []))

    # Set default styles if not provided
    if main_style is None:
        main_style = dict(width=3)

    if secondary_style is None:
        secondary_style = dict(width=1)

    # Generate color sequence from the chosen colorscale
    color_seq = colors.qualitative.__dict__[colorscale][:n_trajs_to_show]

    # Loop through each state dimension and plot it
    for dim in range(state_dimension):
        row = dim // num_cols + 1
        col = dim % num_cols + 1 + col_shift

        fig.update_yaxes(title_text=dim_names[dim], row=row, col=col)
        fig.update_xaxes(title_text=f"Time[{'s' if dt != 1 else 'steps'}]", row=row, col=col)

        for traj_idx, color in enumerate(color_seq):
            y_vals = trajs[traj_idx, :, dim]
            x_vals = np.arange(time) * dt

            legend_group = f"{traj_idx}"
            fig.add_trace(
                go.Scatter(x=x_vals, y=y_vals, mode='lines',
                           line=dict(color=color, **main_style),
                           name=f"{main_legend_label}_{traj_idx}",
                           legendgroup=legend_group, showlegend=show_legend and (dim == 0)),
                row=row, col=col
                )

            if secondary_trajs is not None:
                y_vals_secondary = secondary_trajs[traj_idx, :, dim]
                fig.add_trace(
                    go.Scatter(x=x_vals, y=y_vals_secondary, mode='lines', opacity=0.5,
                               line=dict(color=color, **secondary_style),
                               name=f"{secondary_legend_label}_{traj_idx}",
                               legendgroup=legend_group, showlegend=False),
                    row=row, col=col
                    )

                if shade_area:
                    fig.add_trace(
                        go.Scatter(x=np.concatenate([x_vals, x_vals[::-1]]),
                                   y=np.concatenate([y_vals, y_vals_secondary[::-1]]),
                                   fill='toself', fillcolor=color, opacity=0.1,
                                   line=dict(width=0, color='rgba(0,0,0,0)'),
                                   legendgroup=legend_group, showlegend=False),
                        row=row, col=col
                        )

    # Add error plot at the bottom if requested and secondary trajectories are provided
    if plot_error:
        # Calculate the mean squared error across dimensions for each time point and trajectory
        mse_time_traj = np.mean((trajs - secondary_trajs) ** 2, axis=2)
        # Average these values across trajectories to get a single error estimation vs time
        mse_time = np.mean(mse_time_traj, axis=0)

        x_vals = np.arange(mse_time.shape[0]) * dt
        fig.add_trace(
            go.Scatter(x=x_vals, y=mse_time, mode='lines',
                       fill='tozeroy', fillcolor='rgba(255, 0, 0, 0.1)',  # Added fill options here
                       line=dict(color='red', width=2),
                       name=f"MSE",
                       legendgroup=f"MSE", showlegend=False),
            row=num_rows, col=num_rows + col_shift
            )
        fig.update_yaxes(title_text="Mean Square Error", row=num_rows, col=num_cols + col_shift)
        fig.update_xaxes(title_text=f"Time[{'s' if dt != 1 else 'steps'}]", row=num_rows, col=num_cols + col_shift)

    # Update layout
    fig.update_layout(plot_bgcolor='rgba(245, 245, 245, 1)',
                      paper_bgcolor='rgba(245, 245, 245, 1)')
    return fig


from plotly.subplots import make_subplots
import plotly.graph_objects as go


def plot_two_panel_trajectories(state_trajs, obs_state_trajs, pred_obs_state_trajs=None,
                                pred_state_trajs=None,
                                dt=1,
                                main_legend_label='traj', secondary_legend_label=None,
                                n_trajs_to_show=5):
    """
    Plot two-panel trajectories for state space and observable state space with error subplots.

    Parameters:
    - state_trajs, pred_state_trajs, obs_state_trajs, pred_obs_state_trajs (np.ndarray):
      Trajectory data for state and observable state spaces, both actual and predicted.
    - dt (float): Time step for the x-axis.
    - main_legend_label, secondary_legend_label (str): Legend labels for the main and secondary trajectories.
    - n_trajs_to_show (int): Number of trajectories to actually plot.
    - window_size (int or None): Window size for moving average error computation.

    Returns:
    - plotly.graph_objects.Figure: Figure containing the plotted trajectories.
    """

    # Determine the number of dimensions for state and observable state spaces
    state_dimension = state_trajs.shape[2]
    obs_state_dimension = obs_state_trajs.shape[2]

    # Set default dimensions names if not provided. If str is provided set {str}_{dim_idx} as the name
    state_dim_names = ["w_{:d}".format(i + 1) for i in range(state_dimension)]
    obs_state_dim_names = ["x_{:d}(w)".format(i + 1) for i in range(obs_state_dimension)]

    # Calculate the number of rows and columns for the subplots in each panel
    state_num_rows = int(np.ceil(np.sqrt(state_dimension + 1)))  # +1 for the error subplot
    state_num_cols = int(np.ceil((state_dimension + 1) / state_num_rows))

    obs_state_num_rows = int(np.ceil(np.sqrt(obs_state_dimension + 1)))  # +1 for the error subplot
    obs_state_num_cols = int(np.ceil((obs_state_dimension + 1) / obs_state_num_rows))

    # Create an empty figure with subplots
    fig = make_subplots(rows=max(state_num_rows, obs_state_num_rows),
                        cols=state_num_cols + obs_state_num_cols,
                        column_widths=[0.5] * state_num_cols + [0.5] * obs_state_num_cols)

    # Plot the state space trajectories in the left panel
    fig = plot_trajectories(state_trajs, pred_state_trajs, fig=fig, dt=dt,
                            dim_names=state_dim_names, main_legend_label=main_legend_label, shade_area=True,
                            secondary_legend_label=secondary_legend_label, n_trajs_to_show=n_trajs_to_show,
                            plot_error=True)

    # Shift the column indices for the obs_state plots
    shift_idx = state_num_cols

    # Plot the observable state space trajectories in the right panel
    fig = plot_trajectories(obs_state_trajs, pred_obs_state_trajs, fig=fig, dt=dt,
                            dim_names=obs_state_dim_names, main_legend_label=main_legend_label, shade_area=True,
                            secondary_legend_label=secondary_legend_label, n_trajs_to_show=n_trajs_to_show,
                            plot_error=True, col_shift=shift_idx, show_legend=False)

    # Update layout
    fig.update_layout(  # title='Trajectories of Dynamical System',
        # xaxis_title='Time',
        plot_bgcolor='rgba(245, 245, 245, 1)',
        paper_bgcolor='rgba(245, 245, 245, 1)')

    return fig


def combine_side_by_side(*figures):
    """
    Combine multiple Plotly figures side by side into a single figure.

    Parameters:
    - figures (list of plotly.graph_objects.Figure): List of figures to combine.

    Returns:
    - plotly.graph_objects.Figure: Combined figure.
    """
    # Calculate the total number of rows and columns for the combined figure
    total_rows = max(fig._grid_ref.shape[0] for fig in figures)
    total_cols = sum(fig._grid_ref.shape[1] for fig in figures)

    # Create an empty figure with the calculated dimensions
    combined_fig = make_subplots(rows=total_rows, cols=total_cols)

    col_offset = 0
    for fig in figures:
        for trace in fig.data:
            row, col = fig._get_subplot_coordinates(trace)
            combined_fig.add_trace(trace, row=row, col=col + col_offset)

        col_offset += fig._grid_ref.shape[1]

    # Update the layout and return the combined figure
    combined_fig.update_layout(title='Combined Figure')
    return combined_fig