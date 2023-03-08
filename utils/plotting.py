import crocoddyl
import numpy as np
from crocoddyl import SolverAbstract
from matplotlib.axis import Axis
from matplotlib.figure import Figure
from pinocchio import RobotWrapper


def plotOCSolution(xs=None, us=None, xs_des=None, us_des=None, robot: RobotWrapper = None, fig: Figure = None,
                   show=True, figTitle="",
                   x_labels=None, u_labels=None, dt: float = None, markersize=2, legend=True, color=None):
    # TODO: Detect base confituration and separate euclidean E3 states from joint_space QJ
    import matplotlib.pyplot as plt

    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42

    nx = xs[0].shape[0]
    nu = us[0].shape[0]

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
        if robot is None:
            x_labels = [r"$x_{%d}$" % i for i in range(nx)]
            xdes_labels = [r"$x_{%d,des}$" % i for i in range(nx)]
        else:
            x_labels = [r"$q_{%d}$" % i for i in range(robot.nq)] + \
                       [r"$\dot{q}_{%d}$" % i for i in range(robot.nv)]
            xdes_labels = [r"$q_{%d,des}$" % i for i in range(robot.nq)] + \
                       [r"$\dot{q}_{%d,des}$" % i for i in range(robot.nv)]

    if u_labels is None:
        u_labels = [r"$u_{%d}$" % i for i in range(nu)]
        udes_labels = [r"$u_{%d,des}$" % i for i in range(nu)]

    # Configure or recover fig.
    if fig is None:
        fig, axs = plt.subplots(ncols=1, nrows=2 if robot is None else 3, sharex=True, figsize=(7/1.5, 10/1.5))
    else:
        axs = fig.get_axes()

    # Plotting the state trajectories
    state_traj_style = {"marker": 'o', "markersize": markersize}
    des_state_traj_style = {"markersize": markersize, "alpha": 0.75, "linestyle": '--'}
    if robot is None:  # Plot state as a vector, not differentiating between q and dq.
        x_ax, u_ax = axs
        plot_ndim_traj(x_ax, t=t, traj=X, traj_area=xs_des, dim_labels=x_labels, color=color, legend=legend, y_label="State",
                       linestyle_kwargs=state_traj_style)
        if xs_des is not None:
            plot_ndim_traj(x_ax, t=t, traj=xs_des, dim_labels=xdes_labels, color=color,
                           linestyle_kwargs=des_state_traj_style)
    else:
        q_ax, dq_ax, u_ax = axs
        plot_ndim_traj(q_ax, t=t, traj=X[:, :robot.nq], traj_area=xs_des[:, :robot.nq], dim_labels=x_labels[:robot.nq],
                       color=color, legend=legend,
                       linestyle_kwargs=state_traj_style, y_label="State pos")
        if xs_des is not None:
            plot_ndim_traj(q_ax, t=t, traj=xs_des[:, :robot.nq], color=color, legend=legend,
                           dim_labels=xdes_labels[:robot.nq],
                           linestyle_kwargs=des_state_traj_style)

        plot_ndim_traj(dq_ax, t=t, traj=X[:, robot.nq:], traj_area=xs_des[:, robot.nq:], dim_labels=x_labels[robot.nq:],
                       color=color, legend=legend,
                       linestyle_kwargs=state_traj_style, y_label="State vel")
        if xs_des is not None:
            plot_ndim_traj(dq_ax, t=t, traj=xs_des[:, robot.nq:], color=color, legend=legend,
                           dim_labels=xdes_labels[robot.nq:],
                           linestyle_kwargs=des_state_traj_style)

        plot_ndim_traj(u_ax, t=t[:U.shape[0]], traj=U, traj_area=us_des, dim_labels=u_labels, color=color, legend=legend,
                   x_label="time[s]" if dt is not None else "Nodes", y_label="Control")
        if xs_des is not None:
            plot_ndim_traj(u_ax, t=t[:us_des.shape[0]], traj=us_des,
                           dim_labels=udes_labels, color=color,
                           legend=legend, x_label="time[s]" if dt is not None else "Nodes", y_label="Control",
                           linestyle_kwargs=des_state_traj_style)
    fig.tight_layout()
    if show:
        fig.show()

    return fig, axs


def plot_ndim_traj(ax, t, traj, dim_labels, traj_area=None, linestyle_kwargs: dict = None, x_label=None,
                   y_label=None, color=None, colors=None, legend=True):
    linestyle_kwargs = {} if linestyle_kwargs is None else linestyle_kwargs

    if colors is None:
        colors = [color] * traj.shape[1]
    for i, (label, color) in enumerate(zip(dim_labels, colors)):
        ax.plot(t, traj[:, i], label=label, color=color, **linestyle_kwargs)
        if traj_area is not None:
            ax.fill_between(t, traj[:, i], traj_area[:, i], color=color, alpha=0.2)

    if x_label is not None: ax.set_xlabel(x_label)
    if y_label is not None: ax.set_ylabel(y_label)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(visible=True, alpha=0.4)
    if legend:
        ax.legend()
    return ax


# Visualization
def get_solver_callbacks(robot: RobotWrapper, log=True, verbose=True, display=False,
                         cameraTF=(1.4, 0., 0.2, 0.5, 0.5, 0.5, 0.5)):
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

        display.rate = -1
        display.freq = 1

    return callbacks
