import pathlib
import sys
import os
import time
import warnings
import pickle

import matplotlib.pyplot as plt
import pinocchio

import crocoddyl
import numpy as np
import example_robot_data

from sklearn.model_selection import train_test_split
from tqdm import tqdm

from utils.mysc import print_dict
from utils.pendulum import ActuationModelPendulum, CostModelPendulum
from utils.plotting import plotOCSolution, get_solver_callbacks


def get_desired_trajectory(traj_type: str, n_nodes: int, dt: float):
    t = np.linspace(0, dt * n_nodes, n_nodes, endpoint=True)
    q_t = np.zeros_like(t)
    dq_t = np.zeros_like(t)
    if traj_type == "unstable_fix_point":
        q_t = np.ones_like(t) * 0
        dq_t = np.zeros_like(t)
    elif traj_type == "periodic":
        A = (2 * np.pi / 8)
        B = (4 * 2 * np.pi / (dt * 200))
        q_t = A * np.sin(B * t)
        dq_t = A * B * np.cos(B * t)
    elif traj_type == "inverted_periodic":
        A = (2 * np.pi / 8)
        B = (3 * 2 * np.pi / (dt * 200))
        q_t = (A * np.sin(B * t)) + np.pi
        dq_t = A * B * np.cos(B * t)

    q_t += np.pi
    # Continuous joints are described by cos(ø) and sin(ø) in pinocchio.
    return np.asarray([np.cos(q_t), np.sin(q_t), dq_t]).T


def construct_oc_nodes(state, actuation, traj_des, dt, state_dim_error_w, state_error_cost_w,
                       terminal_state_error_cost_w, control_error_cost_w):
    # Build trajectory optimization node's models.
    traj_nodes = []
    for node_id, state_des in zip(range(len(traj_des) - 1), traj_des):
        # Residual cost to target state at each node
        xRegCost = crocoddyl.CostModelResidual(state,
                                               activation=crocoddyl.ActivationModelWeightedQuad(state_dim_error_w),
                                               residual=crocoddyl.ResidualModelState(state, state_des, nu))
        # Regulation of control actions
        uRegCost = crocoddyl.CostModelResidual(state, residual=crocoddyl.ResidualModelControl(state, nu))

        runningCostModel = crocoddyl.CostModelSum(state, nu)
        runningCostModel.addCost("uReg", uRegCost, control_error_cost_w)
        runningCostModel.addCost("xGoal", xRegCost, state_error_cost_w)

        runningModel = crocoddyl.IntegratedActionModelRK(
            crocoddyl.DifferentialActionModelFreeFwdDynamics(state, actuation, runningCostModel),
            crocoddyl.RKType.two, dt)
        traj_nodes.append(runningModel)

    # Add terminal cost
    terminal_state = traj_des[-1]
    xRegCost = crocoddyl.CostModelResidual(state,
                                           activation=crocoddyl.ActivationModelWeightedQuad(state_dim_error_w),
                                           residual=crocoddyl.ResidualModelState(state, terminal_state, nu))
    terminal_cost_model = crocoddyl.CostModelSum(state, nu)
    terminal_cost_model.addCost("xGoal", xRegCost, terminal_state_error_cost_w)
    terminal_model = crocoddyl.IntegratedActionModelRK(
        crocoddyl.DifferentialActionModelFreeFwdDynamics(state, actuation, terminal_cost_model),
        crocoddyl.RKType.two, dt)

    return traj_nodes, terminal_model


if __name__ == "__main__":

    from lightning_fabric import seed_everything

    seed_everything(999)
    display = 'display' in sys.argv or 'CROCODDYL_DISPLAY' in os.environ
    plot = 'plot' in sys.argv or 'CROCODDYL_PLOT' in os.environ

    robot_name = "pendulum"
    # Hyperparameters of dataset
    num_trajs = 150
    # Hyperparameters of trajectory
    dt = 1e-2
    n_traj_nodes = 300
    # trajType = "unstable_fix_point"
    # trajType = "inverted_periodic"
    # trajType = "periodic"
    trajType = "unactuated"
    state_dim_error_w = np.array([25, .1])  # 0.5 * (||r||_w)^2 => w: weights
    state_error_cost_w = 1
    terminal_state_error_cost_w = state_error_cost_w
    control_error_cost_w = 5

    # Loading the double pendulum model without joint limits.
    double_pendulum_rw = example_robot_data.load('double_pendulum_continuous')
    assert isinstance(double_pendulum_rw, pinocchio.RobotWrapper)
    # Build a single DoF pendulum from the double pendulum model by fixing elbow joint

    pendulum_rw = double_pendulum_rw.buildReducedRobot(list_of_joints_to_lock=[2])
    model = pendulum_rw.model

    # Crocoddyl state and actuation
    state = crocoddyl.StateMultibody(model)
    actuation = ActuationModelPendulum(state)
    nu = actuation.nu
    g = np.array([1, -1, -1])
    traj_des = get_desired_trajectory(traj_type=trajType, n_nodes=n_traj_nodes, dt=dt)
    g_traj_des = traj_des * g

    oc_nodes, terminal_model = construct_oc_nodes(state, actuation, traj_des, dt, state_dim_error_w, state_error_cost_w,
                                                  terminal_state_error_cost_w, control_error_cost_w)
    g_oc_nodes, g_terminal_model = construct_oc_nodes(state, actuation, g_traj_des, dt, state_dim_error_w,
                                                      state_error_cost_w,
                                                      terminal_state_error_cost_w, control_error_cost_w)

    trajs = []
    fig = None
    Xs, Us, costs = [], [], []
    pbar = tqdm(desc="Closed loop trajs", total=num_trajs)
    for sample in range(num_trajs):
        q0 = np.random.uniform(0, 2 * np.pi, size=1)
        dq0 = np.random.uniform(-2 * np.pi, 2 * np.pi, size=1)

        # Creating the shooting problem and the DDP solver
        x0 = np.array((np.cos(q0), np.sin(q0), dq0)).flatten()
        g_x0 = x0 * g

        problem = crocoddyl.ShootingProblem(x0, oc_nodes, terminal_model)
        solver = crocoddyl.SolverDDP(problem)

        callbacks = get_solver_callbacks(robot=pendulum_rw, verbose=False, log=plot, display=False)
        solver.setCallbacks(callbacks)

        # Solving the problem with the DDP solver
        init_xs = [x for x in traj_des]  # [x0] * (problem.T + 1)
        init_us = [np.zeros(nu)] * problem.T

        if trajType == "unactuated":
            us = init_us
            xs = problem.rollout(us=us)
            costs = [0.] * (problem.T + 1)
        else:
            solver_converged = solver.solve(init_xs=init_xs, init_us=init_us)
            if not solver_converged or not solver.isFeasible:
                warnings.warn(f"Invalid trajectory")
                continue
            # Extract trajectory
            xs, us = np.asarray(solver.xs), np.asarray(solver.us)
            costs = [actionData.cost for actionData in problem.runningDatas] + [problem.terminalData.cost]

        traj_data = {"states": np.asarray(xs), "ctrls": np.asarray(us), "cost": np.asarray(costs)}
        trajs.append(traj_data)

        if sample < num_trajs * 0.15 and not trajType == "unactuated":
            g_problem = crocoddyl.ShootingProblem(g_x0, g_oc_nodes, g_terminal_model)
            g_solver = crocoddyl.SolverDDP(g_problem)
            g_solver.setCallbacks(callbacks)
            g_init_xs = [x for x in g_traj_des]
            g_solver_converged = g_solver.solve(init_xs=g_init_xs, init_us=init_us)
            g_xs, g_us = np.asarray(g_solver.xs), np.asarray(g_solver.us)

            if not np.allclose(g_xs, xs * g, rtol=1e-2, atol=1e-2):
                a = g_xs - (xs * g)
                print("Optimal trajectories appear not to be equivariant.")
                fig, axs = plotOCSolution(xs=xs, us=us, xs_des=traj_des, robot_model=pendulum_rw, show=False, dt=dt,
                                          color='k', markersize=0, plot_area=False)
                plotOCSolution(xs=g_xs, us=g_us, xs_des=g_traj_des, robot_model=pendulum_rw, show=False, dt=dt, color='r',
                               fig=fig, markersize=0, plot_area=False)
                fig.show()
        pbar.update()

    metadata = {"robot_name": robot_name,
                "dynamic_regime": trajType,
                "target_states": traj_des,
                "traj_params": {
                    "dt": dt,
                    "traj_nodes": n_traj_nodes,
                    "num_trajs": num_trajs
                },
                "Pi_params": {
                    "Pi_x_error_w": state_error_cost_w,
                    "Pi_x_dim_error_w": state_dim_error_w,
                    "Pi_u_reg_w": control_error_cost_w,
                }
                }

    # Split recorded trajectories into train test val
    idx = np.arange(len(trajs))
    shuffled_idx = np.random.permutation(idx)
    train_samples, test_samples, val_samples = np.floor(len(trajs) * np.array([.7, .15, .15])).astype(int)

    trajs_np = np.asarray(trajs)
    train_trajs = trajs_np[shuffled_idx[:train_samples]]
    test_trajs = trajs_np[shuffled_idx[train_samples:train_samples + test_samples]]
    val_trajs = trajs_np[shuffled_idx[train_samples + test_samples:]]

    train_color, test_color, val_color = (0, 0.0, 0.0, 1.), (0.218, 0.752, 0.780, 1.), (1.00, 0.784, 0.190, 1.)
    for color, trajs in zip((train_color, test_color, val_color), (train_trajs, val_trajs, test_trajs)):
        for i, traj in enumerate(trajs):
            fig, axs = plotOCSolution(xs=traj["states"], us=traj["ctrls"], xs_des=traj_des, robot_model=pendulum_rw.model,
                                      show=False, dt=dt, fig=fig, color=color, markersize=0, plot_area=False)
            fig.suptitle(f"{robot_name} {trajType} \n"
                         f"Nodes: {n_traj_nodes}, dt: {dt}, train:{len(train_trajs)}-test:{len(test_trajs)}-"
                         f"val:{len(val_trajs)}")
    fig.show()

    path = pathlib.Path("data") / robot_name / trajType / print_dict(metadata['traj_params'])
    path.mkdir(parents=True, exist_ok=True)
    print(f"Saving data to {path}")
    fig.savefig(path / "trajectories.png", dpi=120)

    def save_pickle(file_path: pathlib.Path, object):
        with open(file_path, 'wb') as file_handle:
            pickle.dump(object, file_handle, protocol=pickle.HIGHEST_PROTOCOL)

    dataset = metadata

    for type, trajs in zip(("train", "test", "val"), (train_trajs, test_trajs, val_trajs)):
        dataset['type'] = type
        dataset['trajs'] = trajs
        file_path = path / f'{type}.pickle'
        print(f"Saving dataset {type} with {len(trajs)} trajectories")
        save_pickle(file_path, dataset)
