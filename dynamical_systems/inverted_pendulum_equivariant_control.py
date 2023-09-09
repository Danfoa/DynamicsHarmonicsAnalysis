import sys
import os
import warnings
from typing import Iterable

import pinocchio

import crocoddyl
import numpy as np
import example_robot_data
from pinocchio import RobotWrapper

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


def plotOCSolution(xs=None, us=None, xs_des=None, us_des=None, robot: RobotWrapper = None, fig = None,
                   show=True, plot_area=False, x_labels=None, u_labels=None, dt: float = None, markersize=2,
                   legend=False, color=None):
    # TODO: Detect base confituration and separate euclidean E3 states from joint_space QJ
    import matplotlib.pyplot as plt
    if xs_des is None:
        plot_area = False
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42

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
        if robot is None:
            x_labels = [r"$x_{%d}$" % i for i in range(nx)]
            xdes_labels = [r"$x_{%d,des}$" % i for i in range(nx)]
        else:
            # TODO: Cope with different joint types, maybe plot limits.
            # for joint in robot.model.joints:
            #     print(f"{joint} \n {joint.shortname()}")
            x_labels = [r"$q_{%d}$" % i for i in range(robot.model.njoints)] + \
                       [r"$\dot{q}_{%d}$" % i for i in range(robot.nv)]
            xdes_labels = [r"$q_{%d,des}$" % i for i in range(robot.model.njoints)] + \
                          [r"$\dot{q}_{%d,des}$" % i for i in range(robot.nv)]

    if u_labels is None:
        u_labels = [r"$u_{%d}$" % i for i in range(nu)]
        udes_labels = [r"$u_{%d,des}$" % i for i in range(nu)]

    # Configure or recover fig.
    if fig is None:
        num_rows = nx + nu if robot is None else robot.nq + robot.nv + nu
        fig, axs = plt.subplots(ncols=1, nrows=num_rows, sharex=True, figsize=(7 / 1.5, 10 / 1.5))
    else:
        axs = fig.get_axes()

    # Plotting the state trajectories
    state_traj_style = {"marker": 'o', "markersize": markersize}
    des_state_traj_style = {"markersize": markersize, "alpha": 0.75, "linestyle": '--'}
    if robot is None:  # Plot state as a vector, not differentiating between q and dq.
        x_ax, u_ax = axs[:nx], axs[nx:]
        plot_ndim_traj(x_ax, t=t, traj=X, traj_area=xs_des if plot_area else None,
                       dim_labels=x_labels, color=color, legend=legend,
                       y_label="State", linestyle_kwargs=state_traj_style)
        if xs_des is not None:
            plot_ndim_traj(x_ax, t=t, traj=xs_des, dim_labels=xdes_labels, color=color,
                           linestyle_kwargs=des_state_traj_style)
    else:
        q_ax, dq_ax, u_ax = axs[:robot.nq], axs[robot.nq:robot.nq + robot.nv], axs[-nu:]
        plot_ndim_traj(q_ax, t=t, traj=X[:, :robot.nq], traj_area=xs_des[:, :robot.nq] if plot_area else None,
                       dim_labels=x_labels[:robot.nq], color=color, legend=legend, linestyle_kwargs=state_traj_style, y_label="State pos")
        if xs_des is not None:
            plot_ndim_traj(q_ax, t=t, traj=xs_des[:, :robot.nq], color=color, legend=legend,
                           dim_labels=xdes_labels[:robot.nq], linestyle_kwargs=des_state_traj_style)

        plot_ndim_traj(dq_ax, t=t, traj=X[:, robot.nq:], traj_area=xs_des[:, robot.nq:] if plot_area else None,
                       dim_labels=x_labels[robot.nq:], color=color, legend=legend, linestyle_kwargs=state_traj_style, y_label="State vel")
        if xs_des is not None:
            plot_ndim_traj(dq_ax, t=t, traj=xs_des[:, robot.nq:], color=color, legend=legend,
                           dim_labels=xdes_labels[robot.nq:], linestyle_kwargs=des_state_traj_style)
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


class ActuationModelPendulum(crocoddyl.ActuationModelAbstract):

    def __init__(self, state):
        crocoddyl.ActuationModelAbstract.__init__(self, state, 1)
        self.nv = state.nv

    def calc(self, data, x, u):
        data.tau[:] = data.dtau_du * u

    def calcDiff(self, data, x, u):
        pass

    def commands(self, data, x, tau):
        data.u[:] = tau[0]

    def torqueTransform(self, data, x, tau):
        pass

    def createData(self):
        data = ActuationDataPendulum(self)
        return data


class ActuationDataPendulum(crocoddyl.ActuationDataAbstract):

    def __init__(self, model):
        crocoddyl.ActuationDataAbstract.__init__(self, model)
        self.dtau_du[0] = 1.
        self.tau_set = [True]  # DoF with actuators.
        self.Mtau[0] = 1.


if __name__ == "__main__":

    robot_name = "pendulum"
    # Hyperparameters of dataset
    num_trajs = 2
    # Hyperparameters of trajectory
    dt = 1e-2
    n_traj_nodes = 100
    trajType = "unstable_fix_point"

    state_dim_error_w = np.array([10, .1])  # 0.5 * (||r||_w)^2 => w: weights
    state_error_cost_w = 10
    terminal_state_error_cost_w = state_error_cost_w
    control_error_cost_w = 100

    # Loading the double pendulum model without joint limits.
    double_pendulum_rw = example_robot_data.load('double_pendulum_continuous')
    # Build a single DoF pendulum from the double pendulum model by fixing elbow joint
    pendulum_rw = double_pendulum_rw.buildReducedRobot(list_of_joints_to_lock=[2])
    model = pendulum_rw.dp_net

    # Crocoddyl state and actuation
    state = crocoddyl.StateMultibody(model)
    actuation = ActuationModelPendulum(state)
    nu = actuation.nu

    # Build trajectory optimization node's models.
    trajNodes = []
    #                       [cos(ø), sin(ø), dø/dt]   ø: the pendulum angle (continuous joint)
    targetState = np.array([     1,      0,     0])     # ø = 0   Works
    # targetState = np.array([     -1,      0,     0])  # ø = π   Doest not work

    for node_id in range(n_traj_nodes - 1):
        # Residual cost to target state
        xRegCost = crocoddyl.CostModelResidual(state,
                                               activation=crocoddyl.ActivationModelWeightedQuad(state_dim_error_w),
                                               residual=crocoddyl.ResidualModelState(state, targetState, nu))
        # Regulation of control actions
        uRegCost = crocoddyl.CostModelResidual(state, residual=crocoddyl.ResidualModelControl(state, nu))
        # Node cost as sum of state and control costs
        runningCostModel = crocoddyl.CostModelSum(state, nu)
        runningCostModel.addCost("uReg", uRegCost, control_error_cost_w)
        runningCostModel.addCost("xGoal", xRegCost, state_error_cost_w)

        runningModel = crocoddyl.IntegratedActionModelRK(
            crocoddyl.DifferentialActionModelFreeFwdDynamics(state, actuation, runningCostModel),
            crocoddyl.RKType.two, dt)
        trajNodes.append(runningModel)

    # Add terminal cost
    xRegCost = crocoddyl.CostModelResidual(state,
                                           activation=crocoddyl.ActivationModelWeightedQuad(state_dim_error_w),
                                           residual=crocoddyl.ResidualModelState(state, targetState, nu))
    terminalCostModel = crocoddyl.CostModelSum(state, nu)
    terminalCostModel.addCost("xGoal", xRegCost, terminal_state_error_cost_w)
    terminalModel = crocoddyl.IntegratedActionModelRK(
        crocoddyl.DifferentialActionModelFreeFwdDynamics(state, actuation, terminalCostModel),
        crocoddyl.RKType.two, dt)

    trajs = []

    # Test pendulum model from two symmetric initial conditions at +45º and -45º.
    # The optimal trajectory to reach the unstable fixed point (at 0º) from +45º should be almost
    # symmetric to the one from -45º.
    q0_p45 = np.array([np.cos(np.pi / 4), np.sin(np.pi / 4)])
    q0_n45 = np.array([np.cos(-np.pi / 4), np.sin(-np.pi / 4)])
    dq0 = np.array([0.0])

    # Check pinocchio difference
    diff_p45 = pinocchio.difference(model, q0_p45, targetState[:model.nq])
    diff_n45 = pinocchio.difference(model, q0_n45, targetState[:model.nq])
    assert diff_n45 == -diff_p45, "Pinocchio difference is not symmetric"

    def solve_traj(x0):
        problem = crocoddyl.ShootingProblem(x0, trajNodes, terminalModel)
        solver = crocoddyl.SolverDDP(problem)

        # targetStates = np.asarray(targetStates).T
        callbacks = get_solver_callbacks(robot=pendulum_rw, verbose=False, log=False, display=False)
        solver.setCallbacks(callbacks)
        # Solving the problem with the DDP solver
        xs_init = [x0] * (problem.T + 1)
        solver_converged = solver.solve(init_xs=xs_init, init_us=[])
        if not solver_converged or not solver.isFeasible:
            warnings.warn(f"Invalid trajectory")
        # EXTRACT TRAJECTORY DATA
        return solver, np.asarray(solver.xs), np.asarray(solver.us)
    # +45º traj
    x0_p45 = np.concatenate([q0_p45, dq0])
    solver_p45, xs_p45, us_p45 = solve_traj(x0_p45)
    # -45º traj
    x0_n45 = np.concatenate([q0_n45, dq0])
    solver_n45, xs_n45, us_n45 = solve_traj(x0_n45)

    fig, axs = plotOCSolution(xs=xs_p45, us=us_p45, robot=pendulum_rw)
    plotOCSolution(xs=xs_n45, us=us_n45, robot=pendulum_rw, fig=fig)
    fig.show()

