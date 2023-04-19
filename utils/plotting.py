import copy
from collections.abc import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pinocchio
from matplotlib.figure import Figure
from pinocchio import RobotWrapper


def plotOCSolution(xs=None, us=None, xs_des=None, us_des=None, robot_model: pinocchio.Model = None, fig: Figure = None,
                   show=True, plot_area=False, x_labels=None, u_labels=None, dt: float = None, markersize=2,
                   legend=False, color=None):
    # TODO: Detect base confituration and separate euclidean E3 states from joint_space QJ
    import matplotlib.pyplot as plt
    if xs_des is None:
        plot_area = False
    # plt.rcParams['font.family'] = 'sans-serif'
    # plt.rcParams['font.sans-serif'] = ['Verdana']

    nx = xs[0].shape[0]
    nu = us[0].shape[0] if us is not None else 0

    X = np.asarray(xs)
    U = np.asarray(us)

    t = np.arange(xs.shape[0])
    if dt is not None:
        t = t * dt

    if xs_des is not None:
        assert xs_des.shape == xs.shape, f"Real and desired state trajs differ in size {xs_des.shape} != {xs.shape}"

    # Configure labels
    xdes_labels, udes_labels = None, None
    if x_labels is None:
        if robot_model is None:
            x_labels = [r"$x_{%d}$" % i for i in range(nx)]
            xdes_labels = [r"$x_{%d,des}$" % i for i in range(nx)]
        else:
            # TODO: Cope with different joint types, maybe plot limits.
            # for joint in robot.joints:
            #     print(f"{joint} \n {joint.shortname()}")
            x_labels = [r"$q_{%d}$" % i for i in range(robot_model.njoints)] + \
                       [r"$\dot{q}_{%d}$" % i for i in range(robot_model.nv)]
            xdes_labels = [r"$q_{%d,des}$" % i for i in range(robot_model.njoints)] + \
                          [r"$\dot{q}_{%d,des}$" % i for i in range(robot_model.nv)]

    if u_labels is None:
        u_labels = [r"$u_{%d}$" % i for i in range(nu)]
        udes_labels = [r"$u_{%d,des}$" % i for i in range(nu)]

    # Configure or recover fig.
    if fig is None:
        num_rows = nx + nu if robot_model is None else robot_model.nq + robot_model.nv + nu
        fig, axs = plt.subplots(ncols=1, nrows=num_rows, sharex=True, figsize=(7 / 1.5, 10 / 1.5))
    else:
        axs = fig.get_axes()

    # Plotting the state trajectories
    state_traj_style = {"marker": 'o', "markersize": markersize}
    des_state_traj_style = {"markersize": markersize, "alpha": 0.75, "linestyle": '--'}
    if robot_model is None:  # Plot state as a vector, not differentiating between q and dq.
        x_ax, u_ax = axs[:nx], axs[nx:]
        plot_ndim_traj(x_ax, t=t, traj=X, traj_area=xs_des if plot_area else None,
                       dim_labels=x_labels, color=color, legend=legend,
                       y_label="State", linestyle_kwargs=state_traj_style)
        if xs_des is not None:
            plot_ndim_traj(x_ax, t=t, traj=xs_des, dim_labels=xdes_labels, color=color,
                           linestyle_kwargs=des_state_traj_style)
    else:
        q_ax, dq_ax, u_ax = axs[:robot_model.nq], axs[robot_model.nq:robot_model.nq + robot_model.nv], axs[-nu:]
        plot_ndim_traj(q_ax, t=t, traj=X[:, :robot_model.nq], traj_area=xs_des[:, :robot_model.nq] if plot_area else None,
                       dim_labels=x_labels[:robot_model.nq], color=color, legend=legend, linestyle_kwargs=state_traj_style, y_label="State pos")
        if xs_des is not None:
            plot_ndim_traj(q_ax, t=t, traj=xs_des[:, :robot_model.nq], color=color, legend=legend,
                           dim_labels=xdes_labels[:robot_model.nq], linestyle_kwargs=des_state_traj_style)

        plot_ndim_traj(dq_ax, t=t, traj=X[:, robot_model.nq:], traj_area=xs_des[:, robot_model.nq:] if plot_area else None,
                       dim_labels=x_labels[robot_model.nq:], color=color, legend=legend, linestyle_kwargs=state_traj_style, y_label="State vel")
        if xs_des is not None:
            plot_ndim_traj(dq_ax, t=t, traj=xs_des[:, robot_model.nq:], color=color, legend=legend,
                           dim_labels=xdes_labels[robot_model.nq:], linestyle_kwargs=des_state_traj_style)
        if us is not None:
            plot_ndim_traj(u_ax, t=t[:U.shape[0]], traj=U, traj_area=us_des if plot_area else None,
                           dim_labels=u_labels, color=color, legend=legend,
                           x_label="time[s]" if dt is not None else "Nodes", y_label="Control")
            if us_des is not None:
                plot_ndim_traj(u_ax, t=t[:us_des.shape[0]], traj=us_des,
                               dim_labels=udes_labels, color=color,
                               legend=legend, x_label="time[s]" if dt is not None else "Nodes", y_label="Control",
                               linestyle_kwargs=des_state_traj_style)
    fig.tight_layout()
    if show:
        fig.show()

    return fig, axs


def plotNdimTraj(traj=None, traj_des=None, fig: Figure = None, show=True, figTitle="", eigvals=None,
                 var_name='x', dt: float = None, markersize=2, legend=True, color=None, figsize=None,
                 xs_style=None, xs_des_style=None, artists=None,
                 ncols=1, **kwargs):
    # TODO: Detect base confituration and separate euclidean E3 states from joint_space QJ
    import matplotlib.pyplot as plt

    # plt.rcParams["pdf.fonttype"] = 42
    # plt.rcParams["ps.fonttype"] = 42
    xs_line_style = {"marker": 'o', "markersize": markersize}
    densly_dotted = (0, (1, 1))
    xs_des_line_style = {"markersize": markersize, "alpha": 0.75, "linestyle": densly_dotted, "linewidth": 3}
    if xs_style is not None: xs_line_style.update(xs_style)
    if xs_des_style is not None: xs_des_line_style.update(xs_style)

    complex_traj = traj.dtype in (np.csingle, np.cdouble, np.clongdouble)
    nx = traj[0].shape[0]

    if complex_traj and eigvals is not None:
        assert len(eigvals) == nx, "Number of unique eigenvalues (no conj) should be the number of obs"

    if traj_des is not None:
        assert traj_des.shape == traj.shape, f"Real and desired state trajs differ in size {traj_des.shape} != {traj.shape}"

    t = np.arange(traj.shape[0])
    if dt is not None:
        t = t * dt

    # Configure labels
    x_labels, x_des_labels = None, None
    if x_labels is None:
        x_labels = [r"$%s_{%d}$" % (var_name, i) for i in range(nx)]

    # Configure or recover fig.
    if fig is None:
        nrows = nx//ncols
        if figsize is None:
            figsize = (ncols*2.8, nrows*2.8)
        fig, axs = plt.subplots(ncols=ncols, nrows=nrows, sharex=not complex_traj, figsize=figsize)
        if nx == 1: axs = [axs]
    else:
        axs = fig.get_axes()

    if not isinstance(axs, list):
        axs = axs.flatten()

    # Plotting the state trajectories
    for dim, (ax, y_label) in enumerate(zip(axs, x_labels)):
        if complex_traj:
            x = traj[:, dim]
            x_des = traj_des[:, dim ]
            artists = plot_complex_traj(ax, traj=x, artists=artists, color=color, legend=False, y_label=y_label,
                                        linestyle_kwargs=xs_line_style, eigval=eigvals[dim], **kwargs)
            if traj_des is not None:
                plot_complex_traj(ax, traj=x_des, artists=artists, color=color, legend=False, y_label=y_label,
                                  linestyle_kwargs=xs_des_line_style, traj0_markersize=25, traj0_marker="D")
        else:
            x, x2 = traj[:, dim], traj_des[:, dim] if traj_des is not None else None
            plot_traj(ax, t=t, traj=traj[:, dim], traj_area=x2, color=color, legend=False, y_label=y_label,
                      linestyle_kwargs=xs_line_style)
            if traj_des is not None:
                plot_traj(ax, t=t, traj=x2, color=color, legend=False, linestyle_kwargs=xs_des_line_style)

    fig.tight_layout()
    if show:
        fig.show()

    return fig, axs, artists


def plot_traj(ax, t, traj, traj_area=None, linestyle_kwargs: dict = None, label=None, x_label=None, y_label=None,
              color=None, legend=True):
    linestyle_kwargs = {} if linestyle_kwargs is None else linestyle_kwargs

    ax.plot(t, traj, label=label, color=color, **linestyle_kwargs)
    if traj_area is not None:
        ax.fill_between(t, traj, traj_area, color=color, alpha=0.1)

    if x_label is not None: ax.set_xlabel(x_label)
    if y_label is not None: ax.set_ylabel(y_label)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(visible=True, alpha=0.1)
    if legend is not None:
        ax.legend()
    return ax


def plot_complex_traj(ax, traj, linestyle_kwargs: dict = None, label=None, x_label=None, y_label=None,
                      color=None, legend=True, plot_unit_circle=True, traj0_marker='o', traj0_markersize=10,
                      eigval=None, artists=None, plot_grad_field=False):
    from matplotlib.ticker import MaxNLocator
    MAX_STATE, TRAJ, TRAJ0, UNIT_CIRCLE, EIGENVALS, VECT_FIELD = 'r_max', 'x', 'x0', 'circle', 'eigval', 'quiver'
    linestyle_kwargs = {} if linestyle_kwargs is None else linestyle_kwargs
    circle_color = (0.202, 0.416, 0.470)
    eigval_color = (1.00, 0.0700, 0.0700)
    quiver_color = (0.0660, 0.396, 0.440, 0.15)
    if artists is None: artists = {}
    first_draw = False
    if ax not in artists:
        artists[ax] = {TRAJ: [], TRAJ0: [], MAX_STATE: []}
        first_draw = True

    assert traj.dtype in (np.csingle, np.cdouble, np.clongdouble), "Trajectory must be complex"

    # Plot trajectory.
    line = ax.plot(np.real(traj), np.imag(traj), label=label, color=color, **linestyle_kwargs)
    artists[ax][TRAJ].append(line)
    artists[ax][MAX_STATE].append(np.max(np.abs(traj)))
    # Plot start state marker.
    init_state_style = dict(marker=traj0_marker, s=traj0_markersize, linewidths=1.5,
                            edgecolors=color, color=circle_color)
    traj0 = ax.scatter(np.real(traj[0]), np.imag(traj[0]), **init_state_style)
    artists[ax][TRAJ0].append(traj0)

    # Set plot limits for good visualization of trajectories
    ax_lim = np.max(artists[ax][MAX_STATE])
    ax_lim *= 1.1   # Add 10% border
    ax.set_xlim(-ax_lim, ax_lim)
    ax.set_ylim(-ax_lim, ax_lim)
    if ax_lim < 0.5:  # Fix ticks for visualization of collapsed trajs
        ax.xaxis.set_ticks(np.arange(-ax_lim, ax_lim, 3))
        ax.yaxis.set_ticks(np.arange(-ax_lim, ax_lim, 3))
        # ax.xaxis.set_major_locator(plt.MaxNLocator(3))
        # ax.yaxis.set_major_locator(plt.MaxNLocator(3))

    if plot_unit_circle and UNIT_CIRCLE not in artists[ax]:
        theta = np.linspace(0, 2*np.pi, 50)
        x, y = np.cos(theta), np.sin(theta)
        unit_circle = ax.plot(x, y, color=circle_color, alpha=0.3)
        center_point = ax.scatter(0, 0, marker="+", color=circle_color, s=60)
        artists[ax][UNIT_CIRCLE] = [unit_circle, center_point]

    if eigval is not None and EIGENVALS not in artists[ax]:
        # Plot conjugate pair of eigenvalues.
        eig_pair = np.array([[np.real(eigval), np.real(eigval)], [np.imag(eigval), -np.imag(eigval)]])
        eig_points = ax.scatter(eig_pair[0, :], eig_pair[1, :], marker="X", color=eigval_color, s=80,
                                label=r"$\lambda=%.2f \pm %.2fi$" % (np.real(eigval), np.imag(eigval)))
        eig_line = ax.plot(eig_pair[0, :], eig_pair[1, :], color=eigval_color + (0.3,), linewidth=1)
        artists[ax][EIGENVALS] = [eig_points, eig_line]

    if plot_grad_field and eigval is not None and VECT_FIELD not in artists[ax]:
        num_angles, num_rad = 20, 5
        radius = np.linspace(ax_lim/num_rad, ax_lim, num_rad)
        theta = np.linspace(0, 2*np.pi, num_angles)[:-1]
        x, y = [], []
        for r in radius:
            for a in theta:
                x.append(r * np.cos(a))
                y.append(r * np.sin(a))
        x, y = np.asarray(x).flatten(), np.asarray(y).flatten()
        vel = eigval * (x + 1j*y)
        u, v = np.real(vel), np.imag(vel)
        quiver = ax.quiver(x, y, u, v, angles='xy', units='xy', color=quiver_color)
        artists[ax][VECT_FIELD] = quiver

    if first_draw:
        if x_label is not None: ax.set_xlabel(x_label)
        if y_label is not None: ax.set_ylabel(y_label)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.grid(visible=True, alpha=0.1)
        ax.set_aspect('equal')
        ax.set_xlabel("Re")
        ax.set_ylabel("Im")
        ticks_color = (1, 1, 1, 0.4)
        ax.tick_params(axis='both', which='both', color=ticks_color, labelcolor=(0.4, 0.4, 0.4), labelsize='xx-small')
        ax.xaxis.set_major_formatter('{x:.1f}')
        ax.yaxis.set_major_formatter('{x:.1f}')
        # if legend:
        ax.legend(fontsize='xx-small', frameon=True)
    return artists


def plot_ndim_traj(axs, t, traj, dim_labels=None, traj_area=None, linestyle_kwargs: dict = None, x_label=None,
                   y_label=None, color=None, colors=None, legend=True):
    linestyle_kwargs = {} if linestyle_kwargs is None else linestyle_kwargs

    if colors is None: colors = [color] * traj.shape[-1]
    if dim_labels is None: dim_labels = [None] * traj.shape[-1]

    if isinstance(axs, Iterable):
        assert len(axs) == traj.shape[-1], "Number of axes must match number of dimensions"
        multiaxis = True
    else:
        axs = [axs] * traj.shape[-1]
        multiaxis = False

    for i, (ax, label, color) in enumerate(zip(axs, dim_labels, colors)):
        ax.plot(t, traj[:, i], label=label if not multiaxis else None, color=color, **linestyle_kwargs)
        if traj_area is not None:
            ax.fill_between(t, traj[:, i], traj_area[:, i], color=color, alpha=0.1)

    for ax, dim_label in zip(axs, dim_labels):
        if multiaxis:
            ax.set_ylabel(dim_label)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(visible=True, alpha=0.1)
    if legend:
        ax.legend()
    return ax


# Visualization
def get_solver_callbacks(robot: RobotWrapper, log=True, verbose=True, display=False,
                         cameraTF=(1.4, 0., 0.2, 0.5, 0.5, 0.5, 0.5)):
    import crocoddyl
    log_call = crocoddyl.CallbackLogger()
    verbose_call = crocoddyl.CallbackVerbose()

    callbacks = []
    if verbose:
        verbose_call.precision = 3
        verbose_call.level = crocoddyl.VerboseLevel._2
        callbacks += [verbose_call]

    callbacks = callbacks + [log_call] if log else callbacks

    if display:
        try:
            import gepetto
            gepetto.corbaserver.Client()
            display = crocoddyl.GepettoDisplay(robot, 4, 4, cameraTF, floor=False)
            display_call = crocoddyl.CallbackDisplay(display)
            callbacks += [display_call]
        except Exception:
            display = crocoddyl.MeshcatDisplay(robot)
            display_call = crocoddyl.CallbackDisplay(display)
            callbacks += [display_call]

        display.rate = -1
        display.freq = 1

    return callbacks
