import pathlib
import sys
import os
import time
import warnings
import pickle

import matplotlib.pyplot as plt
from datasets import Dataset
import pinocchio as pin
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
        pass
    elif traj_type == "inverted_periodic":
        q_t = (2 * np.pi / 6) * np.sin((3 * 2 * np.pi / (dt * n_nodes)) * t)
        dq_t = (2 * np.pi / 6) * np.cos((3 * 2 * np.pi / (dt * n_nodes)) * t)
    elif traj_type == "periodic":
        q_t = (2 * np.pi / 6) * np.sin((3 * 2 * np.pi / (dt * n_nodes)) * t) + np.pi
        dq_t = (2 * np.pi / 6) * np.cos((3 * 2 * np.pi / (dt * n_nodes)) * t)
    return q_t, dq_t


if __name__ == "__main__":

    from lightning_fabric import seed_everything
    seed_everything(999)
    display = 'display' in sys.argv or 'CROCODDYL_DISPLAY' in os.environ
    plot = 'plot' in sys.argv or 'CROCODDYL_PLOT' in os.environ

    robot_name = "pendulum"
    # Hyperparameters of dataset
    n_tiles = 15
    # Hyperparameters of trajectory
    dt = 1e-2
    n_traj_nodes = 100
    trajType = "unstable_fix_point"
    # Hyperparameters of control policy
    state_dim_error_w = np.array([10., .01])  # 0.5 * (||r||_w)^2 => w: weights
    state_error_cost_w = 10
    terminal_state_error_cost_w = 10
    control_error_cost_w = 1000

    # Loading the double pendulum model
    double_pendulum_rw = example_robot_data.load('double_pendulum')
    # Build a single DoF pendulum from the double pendulum model by fixing elbow joint
    pendulum_rw = double_pendulum_rw.buildReducedRobot(list_of_joints_to_lock=[2])
    model = pendulum_rw.model

    # Crocoddyl state and actuation
    state = crocoddyl.StateMultibody(model)
    actuation = ActuationModelPendulum(state)
    nu = actuation.nu

    q_des_t, dq_des_t = get_desired_trajectory(traj_type=trajType, n_nodes=n_traj_nodes, dt=dt)

    # Build trajectory optimization node's models.
    trajNodes = []
    targetStates = []
    for node_id, q_des, dq_des in zip(range(n_traj_nodes - 1), q_des_t, dq_des_t):
        runningCostModel = crocoddyl.CostModelSum(state, nu)

        # Residual cost to target state at 0
        targetState = np.stack((q_des, dq_des))
        xRegCost = crocoddyl.CostModelResidual(state,
                                               activation=crocoddyl.ActivationModelWeightedQuad(state_dim_error_w),
                                               residual=crocoddyl.ResidualModelState(state, targetState, nu))
        # Regulation of control actions
        uRegCost = crocoddyl.CostModelResidual(state, residual=crocoddyl.ResidualModelControl(state, nu))
        # xPendCostAct = crocoddyl.ActivationModelWeightedQuad(np.array([1.] * 1 + [0.1] * 1))
        # xPendCost = CostModelPendulum(state, activation=xPendCostAct, nu=nu)
        runningCostModel.addCost("uReg", uRegCost, control_error_cost_w / dt)
        runningCostModel.addCost("xGoal", xRegCost, state_error_cost_w / dt)
        runningModel = crocoddyl.IntegratedActionModelRK(
            crocoddyl.DifferentialActionModelFreeFwdDynamics(state, actuation, runningCostModel),
            crocoddyl.RKType.two, dt)
        trajNodes.append(runningModel)
        targetStates.append(targetState)

    # Add terminal cost
    terminalState = np.stack((q_des_t[-1], dq_des_t[-1]))
    targetStates.append(terminalState)
    targetStates = np.array(targetStates)
    xRegCost = crocoddyl.CostModelResidual(state,
                                           activation=crocoddyl.ActivationModelWeightedQuad(state_dim_error_w),
                                           residual=crocoddyl.ResidualModelState(state, terminalState, nu))
    terminalCostModel = crocoddyl.CostModelSum(state, nu)
    terminalCostModel.addCost("xGoal", xRegCost, terminal_state_error_cost_w / dt)
    terminalModel = crocoddyl.IntegratedActionModelRK(
        crocoddyl.DifferentialActionModelFreeFwdDynamics(state, actuation, terminalCostModel),
        crocoddyl.RKType.two, dt)

    trajs = []
    q0s = np.linspace(-2 * np.pi, 2 * np.pi, num=n_tiles)
    dq0s = np.linspace(-2 * np.pi, 2 * np.pi, num=n_tiles)

    fig = None
    Xs, Us, costs = [], [], []
    pbar = tqdm(desc="Closed loop trajs", total=n_tiles ** 2)
    for q0 in q0s:
        for dq0 in dq0s:
            pbar.update()
            # Creating the shooting problem and the DDP solver
            x0 = np.stack((q0, dq0))
            problem = crocoddyl.ShootingProblem(x0, trajNodes, terminalModel)
            solver = crocoddyl.SolverDDP(problem)
            log = crocoddyl.CallbackLogger()

            # targetStates = np.asarray(targetStates).T
            callbacks = get_solver_callbacks(robot=pendulum_rw, verbose=False, log=plot, display=display)
            solver.setCallbacks(callbacks)
            # Solving the problem with the DDP solver
            solver_converged = solver.solve()
            if not solver_converged or not solver.isFeasible:
                warnings.warn(f"Invalid trajectory")
                continue

            # EXTRACT TRAJECTORY DATA
            xs, us = np.asarray(solver.xs), np.asarray(solver.us)
            costs = [actionData.cost for actionData in problem.runningDatas] + [problem.terminalData.cost]

            traj_data = {"states": np.asarray(xs), "ctrls": np.asarray(us), "cost": np.asarray(costs)}
            trajs.append(traj_data)

    #         fig, axs = plotOCSolution(xs=traj_data["states"], us=traj_data["ctrls"], X_des=targetStates, robot=pendulum_rw,
    #                                   show=False, dt=dt, fig=fig, legend=fig is None)
    # fig.show()

    metadata = {"robot_name": robot_name,
                "dynamic_regime": trajType,
                "target_states": targetStates,
                "traj_params": {
                    "dt": dt,
                    "traj_nodes": n_traj_nodes,
                    "n_tiles": n_tiles
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

    trajs = np.asarray(trajs)
    train_trajs = trajs[shuffled_idx[:train_samples]]
    test_trajs = trajs[shuffled_idx[train_samples:train_samples + test_samples]]
    val_trajs = trajs[shuffled_idx[train_samples + test_samples:]]
    
    for trajs, color in zip((train_trajs, val_trajs, test_trajs), ('tab20c', 'gnuplot', 'prism')):
        for traj in trajs:
            fig, axs = plotOCSolution(xs=traj["states"], us=traj["ctrls"], xs_des=targetStates, robot=pendulum_rw,
                                      show=False, dt=dt, fig=fig, legend=fig is None, color=color)
    fig.show()

    path = pathlib.Path("data") / robot_name / trajType / print_dict(metadata['traj_params'])
    path.mkdir(parents=True, exist_ok=True)
    print(f"Saving data to {path}")
    fig.savefig(path/"trajectories.png", dpi=120)

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

    # # /    a = load_dataset("json", data_files=)
    #
    # # Plotting the entire motion
    # if log:
    #     log = solver.getCallbacks()[1]
    #     plotOCSolution(log.xs, log.us, xs_des=targetStates, robot=pendulum_rw, figIndex=1, show=True)
    #     # crocoddyl.plotConvergence(log.costs, log.u_regs, log.x_regs, log.grads, log.stops, log.steps, figIndex=2)
    #
    # # Display the entire motion
    # if display:
    #     while True:
    #         display_call = callbacks[-1]
    #         display_call(solver)
    #         time.sleep(.5)
