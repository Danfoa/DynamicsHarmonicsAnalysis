import cProfile
import logging
import math
import os
import pstats
import time
from pathlib import Path

import hydra
import numpy as np
import torch
from escnn.nn import FieldType
from morpho_symm.data.DynamicsRecording import DynamicsRecording, get_dynamics_dataset, get_train_test_val_file_paths

from hydra.utils import get_original_cwd
from lightning import Trainer, seed_everything
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig, OmegaConf
from scipy.spatial.transform import Rotation
from torch.utils.data import DataLoader

from data.DynamicsDataModule import DynamicsDataModule
from nn.DynamicsAE2 import DynamicAE
from nn.LightningLatentMarkovDynamics import LightLatentMarkovDynamics
from utils.mysc import check_if_resume_experiment, class_from_name, format_scientific

log = logging.getLogger(__name__)


@hydra.main(config_path='cfg', config_name='config', version_base='1.1')
def main(cfg: DictConfig):
    torch.set_float32_matmul_precision('medium')
    log.info("\n\n NEW RUN \n\n")
    device = torch.device("cuda" if torch.cuda.is_available() and cfg.device != "cpu" else "cpu")

    cfg.seed = cfg.seed if cfg.seed >= 0 else np.random.randint(0, 1000)
    cfg['debug'] = cfg.get('debug', False)
    cfg['debug_loops'] = cfg.get('debug_loops', False)
    seed_everything(seed=cfg.seed)

    root_path = Path(get_original_cwd()).resolve()
    run_path = Path(os.getcwd())
    # Create seed folder
    seed_path = run_path / f"seed={cfg.seed:03d}"
    seed_path.mkdir(exist_ok=True)

    # Check if experiment already run
    ckpt_folder_path = seed_path
    ckpt_call = ModelCheckpoint(dirpath=ckpt_folder_path, filename='best', monitor="loss/val", save_last=True)
    training_done, ckpt_path, best_path = check_if_resume_experiment(ckpt_call)

    if not training_done:
        # Load the dynamics dataset.
        data_path = root_path / "data" / cfg.system.data_path
        device = torch.device(f"cuda:{cfg.device}" if torch.cuda.is_available() and cfg.device != "cpu" else "cpu")
        log.info(f"Configuring to use device {device}")

        datamodule = DynamicsDataModule(data_path,
                                        batch_size=cfg.model.batch_size,
                                        lookback_len=cfg.system.frames_per_state,
                                        pred_horizon=cfg.system.pred_horizon,
                                        val_pred_horizon=cfg.system.eval_pred_horizon,
                                        test_pred_horizon=cfg.system.test_pred_horizon,
                                        system_cfg=cfg.system,
                                        num_workers=cfg.num_workers,
                                        device=device,
                                        split_ratios=(cfg.system.split_ratios.train,
                                                      cfg.system.split_ratios.val,
                                                      cfg.system.split_ratios.test),
                                        train_ratio=cfg.system.train_ratio,
                                        data_augmentation=cfg.model.augment,
                                        state_obs=cfg.system.get('state_obs', None),
                                        standardize=cfg.system.standardize)

        datamodule.prepare_data()
        # datamodule.plot_sample_trajs()
        state_dim = datamodule.test_dataset.shape[-1]

        # Get the LatentModel to train _________________________________________________________________
        latent_dyn_model = get_model(cfg, state_dim=state_dim, state_type=datamodule.state_type)
        latent_dyn_model.to(device)

        pl_trainer, wandb_logger = configure_experiment_trainer(cfg, ckpt_call, run_path, seed_path)

        if cfg.debug_loops:
            profiler = cProfile.Profile()
            profiler.enable()

        pl_latent_dyn_model = latent_dyn_model.fit(trainer=pl_trainer, datamodule=datamodule)
        training_successful = pl_trainer.state.status == "finished"

        # Temporary ==================================================================================================

        import plotly.io as pio
        pio.renderers.default = "browser"

        # Compute the spectral decomposition of the learned evolution operator =========================================
        sample_batch = next(iter(datamodule.test_dataloader()))
        sample_batch.to(device=latent_dyn_model.evolution_operator.device,
                        dtype=latent_dyn_model.evolution_operator.dtype)
        out = latent_dyn_model.evolve_forward(sample_batch)
        pred_state = out['pred_state']
        latent_obs = out['latent_obs']
        pred_latent_obs = out['pred_latent_obs']
        state_lin_rec = torch.einsum('...ol,...l->...o', latent_dyn_model.lin_decoder, pred_latent_obs.data)
        state_lin_rec = state_lin_rec.detach().cpu().numpy()

        from utils.plotting import plot_trajectories
        batch_idx = 0
        x_traj = sample_batch.data.cpu().detach().numpy()[[batch_idx]]
        x_traj_pred = pred_state.data.cpu().detach().numpy()[[batch_idx]]
        x_traj_pred_lin_decoder = state_lin_rec[[batch_idx]]

        # Compute the modes _________________________________________________________________
        mode_info = latent_dyn_model.modes(sample_batch[batch_idx])
        mode_info.dt = 0.003
        mode_info.plot_eigfn_dynamics()
        # mode_info.visual_mode_selection()

        n_modes = mode_info.n_modes
        state_modes = mode_info.modes  # (batch, time, mode_idx, state_dim)
        x_traj_modes = state_modes[[batch_idx]]
        eigvals = mode_info.eigvals

        diff_mode_trajs = [x_traj_pred_lin_decoder]
        fig = plot_trajectories(x_traj, secondary_trajs=x_traj_pred_lin_decoder, dt=mode_info.dt, colorscale='Dark2',
                                main_legend_label="all_modes", main_style=dict(width=1), secondary_style=dict(width=2))
        for n_modes_to_show, color_scale in zip([1, 5, 9, 12],
                                                ['Viridis', 'Cividis', 'Plasma', 'Inferno', 'Magma']):
            x_traj_modes_n = x_traj_modes[:, :, :n_modes_to_show]
            x_traj_rec = np.sum(x_traj_modes_n, axis=-2)
            diff_mode_trajs.append(x_traj_rec)
            plot_trajectories(x_traj, secondary_trajs=x_traj_rec, fig=fig, dt=mode_info.dt, colorscale=color_scale,
                              main_legend_label=f"n_modes={n_modes_to_show}", main_style=dict(width=1),
                              secondary_style=dict(width=2))
        plot_trajectories(x_traj, secondary_trajs=x_traj_pred, fig=fig, dt=mode_info.dt, shade_area=True,
                          main_style=dict(width=2))

        fig.show()

        # Visualize modes in the real robot. _______________________________________________________________________
        import morpho_symm
        from morpho_symm.data.DynamicsRecording import DynamicsRecording
        from morpho_symm.utils.algebra_utils import matrix_to_quat_xyzw, permutation_matrix
        from morpho_symm.utils.rep_theory_utils import group_rep_from_gens
        from morpho_symm.utils.pybullet_visual_utils import configure_bullet_simulation
        from morpho_symm.utils.pybullet_visual_utils import (draw_vector, draw_plane, render_camera_trajectory,
                                                             spawn_robot_instances)
        from morpho_symm.utils.robot_utils import load_symmetric_system
        from hydra.core.global_hydra import GlobalHydra
        GlobalHydra.instance().clear()
        state_mean, state_var = datamodule.state_mean, datamodule.state_var
        robot, G = load_symmetric_system(robot_name="mini_cheetah")
        pb = configure_bullet_simulation(gui=True)

        def get_obs_from_state(state: np.array):
            """Auxiliary function to extract the different observations of the state."""
            q_js = state[..., :12]                 #  (12,)  -> [0, 12)
            v_js = state[..., 12:24]               #  (12,)  -> [12, 24)
            base_z = state[..., [24]]              #  (1,)   -> [24, 25)
            base_vel = state[..., 25:28]           #  (3,)   -> [25, 28)
            base_ori = state[..., 28:31]           #  (3,)   -> [28, 31)
            base_ang_vel = state[..., 31:]         #  (3,)   -> [31, 34)
            return q_js, v_js, base_z, base_vel, base_ori, base_ang_vel

        def get_state_from_obs(q_js, v_js, base_z, base_vel, base_ori, base_ang_vel, base_pos):
            R = Rotation.from_euler("xyz", base_ori).as_matrix()
            q_ori = matrix_to_quat_xyzw(R)
            base_pos[2] = base_z[0]
            base_pos = np.array(base_pos)

            cos_q_js, sin_q_js = np.cos(q_js), np.sin(q_js)  # convert from angle to unit circle parametrization
            # Define joint positions [q1, q2, ..., qn] -> [cos(q1), sin(q1), ..., cos(qn), sin(qn)] format.
            q_js_unit_circle_t = np.stack([cos_q_js, sin_q_js], axis=1)
            q_js_unit_circle_t = q_js_unit_circle_t.reshape(-1)

            q = np.concatenate([base_pos, q_ori, q_js_unit_circle_t], axis=-1)
            v = np.concatenate([np.zeros(6,), v_js], axis=-1)
            return q, v


        offset = max(0.2, 1.8 * robot.hip_height)
        n_mode_trajs = len(diff_mode_trajs)
        base_positions = [[0, 0, 0]] +  [[0, 2 * i * robot.hip_height, 0] for i in range(1, n_mode_trajs + 2)]
        robots = spawn_robot_instances(
            robot, bullet_client=pb, base_positions=base_positions, tint=True, alpha=1.0)

        # Standardize the state trajectory
        state_traj = (x_traj[0] * np.sqrt(state_var)) + state_mean
        state_traj_pred = (x_traj_pred[0] * np.sqrt(state_var)) + state_mean
        state_traj_modes = [(mode_traj[0] * np.sqrt(state_var)) + state_mean for mode_traj in diff_mode_trajs]

        n_repeat = 100
        for _ in range(n_repeat):
            for t in range(len(state_traj)):
                s_t = state_traj[t]
                s_t_pred = state_traj_pred[t]

                q_t, v_t = get_state_from_obs(*get_obs_from_state(s_t), base_pos=base_positions[0])
                q_t_pred, v_t_pred = get_state_from_obs(*get_obs_from_state(s_t_pred), base_pos=base_positions[1])
                robots[0].reset_state(q_t, v_t)
                robots[1].reset_state(q_t_pred, v_t_pred)

                for k, mode_traj in enumerate(state_traj_modes):
                    s_t_mode = mode_traj[t]
                    q_t_mode, v_t_mode = get_state_from_obs(*get_obs_from_state(s_t_mode), base_pos=base_positions[k + 1])
                    robots[k + 2].reset_state(q_t_mode, v_t_mode)
                time.sleep(0.01)

        time.sleep(50)
        print("")
        # import plotly.graph_objects as go
        #
        # x_traj_modes_freq = mode_info.modes_frequency
        # x_traj_modes_modulus = mode_info.modes_modulus
        # x_traj_modes_amp = mode_info.modes_amplitude[[batch_idx]]
        #
        # modes_amplitude = x_traj_modes_amp[0].T  # (num_modes, time,)
        # freqs = x_traj_modes_freq                # (num_modes,)
        # time_steps, n_modes = modes_amplitude.shape
        # time = np.arange(time_steps) * mode_info.dt              # (time,)
        #
        # fig2 = go.Figure()

    #     for i, freq in enumerate(freqs):
    #         # Extract the amplitude trajectory for the current mode
    #         mode_amplitude = modes_amplitude[i, :]
    #         # All points along this trajectory have the same frequency (x value)
    #         fig2.add_trace(go.Scatter3d(
    #             x=[freq] * len(time),  # Repeat the frequency value to match the length of time
    #             y=time,  # Y-axis represents time
    #             z=mode_amplitude,  # Z-axis represents amplitude
    #             mode='lines',  # Connect points with lines
    #             name=f'Mode {i + 1} @ {freq:.2f} Hz'  # Label for the mode
    #             ))
    #
    #     fig2.show()
    #     raise NotImplementedError("Testing the output")
    #     #
    #     # fig = plot_trajectories(trajs=x_traj, dt=1.0, main_legend_label="gt")
    #     # fig = plot_trajectories(trajs=np.permx_traj_modes, fig=fig, dt=1.0, main_legend_label="lin_rec",
    #     #                         colorscale='Tealgrn', n_trajs_to_show=len(x_traj_modes), main_style=dict(width=1))
    #     # fig.show()
    #     # #
    #     #
    #     =============================================================================================================
    #
    #     # Create a pstats object
    #     if cfg.debug_loops:
    #         stats = pstats.Stats(profiler)
    #         stats.sort_stats('cumulative')  # Sort stats by the cumulative time spent in the function
    #         stats.print_stats('koopman_robotics')
    #
    #     if training_successful:
    #         if not cfg.debug:  # Loading best model and test it
    #             if best_path.exists():
    #                 best_ckpt = torch.load(best_path)
    #                 pl_latent_dyn_model.eval()
    #                 pl_latent_dyn_model.load_state_dict(best_ckpt['state_dict'], strict=False)
    #                 pass
    #             else:
    #                 log.warning(f"Best model not found, testing with latest model")
    #         # Test best model. Selected as the model with lowest evaluation loss during training.
    #         results = pl_trainer.test(model=pl_latent_dyn_model, datamodule=datamodule)
    #         test_loss = results[0]['loss/test']
    #         # wandb_logger.experiment.unwatch(model)
    #         wandb_logger.experiment.finish()
    #         return test_loss
    #     else:
    #         raise RuntimeError("Training failed. Check logs for details.")
    # else:
    #     log.warning(f"Training run done. Check {run_path} for results.")


def configure_experiment_trainer(cfg, ckpt_call, run_path, seed_path, wb_logger=True):
    stop_call = EarlyStopping(monitor='loss/val',
                              mode='min',
                              patience=max(cfg.system.early_stop_epochs, int(cfg.system.max_epochs * 0.1)))
    # Get the Hyperparameters for the run
    run_hps = OmegaConf.to_container(cfg, resolve=True)
    run_name = run_path.name
    if wb_logger:
        wandb_logger = WandbLogger(project=f'{cfg.system.name}',
                                   save_dir=seed_path.absolute(),
                                   config=run_hps,
                                   name=run_name,
                                   group=f'{cfg.exp_name}',
                                   job_type='debug' if (cfg.debug or cfg.debug_loops) else None)
    else:
        wandb_logger = None
    # Configure Lightning trainer
    trainer = Trainer(accelerator='gpu' if torch.cuda.is_available() and cfg.device != 'cpu' else 'cpu',
                      devices=[cfg.device] if torch.cuda.is_available() and cfg.device != 'cpu' else 'auto',
                      logger=wandb_logger,
                      log_every_n_steps=1,
                      max_epochs=cfg.system.max_epochs if not cfg.debug_loops else 2,
                      check_val_every_n_epoch=2,
                      callbacks=[ckpt_call, stop_call],
                      fast_dev_run=10 if cfg.debug else False,
                      enable_progress_bar=True,  # cfg.debug_loops or cfg.debug,
                      limit_train_batches=2 if cfg.debug_loops else 1.0,
                      limit_test_batches=2 if cfg.debug_loops else 1.0,
                      limit_val_batches=2 if cfg.debug_loops else 1.0,
                      )
    return trainer, wandb_logger


def get_model(cfg, state_dim: int, state_type: FieldType):
    # state_dim = cfg.system.state_dim
    latent_state_dim = math.ceil(cfg.system.obs_state_ratio * state_dim)
    num_hidden_neurons = cfg.model.num_hidden_units

    if latent_state_dim > num_hidden_neurons:
        # Set num_hidden_neurons to be the closest power of 2 to obs_state_dim from above
        # For obs_state_dim=210 -> num_hidden_neurons=256
        num_hidden_neurons = 2 ** math.ceil(math.log2(latent_state_dim))

    # Get the selected model for observation learning _____________________________________________________________
    if cfg.model.equivariant:
        activation = cfg.model.activation
    else:
        activation = class_from_name('torch.nn', cfg.model.activation)

    obs_fn_params = dict(num_layers=cfg.model.num_layers,
                         num_hidden_units=num_hidden_neurons,
                         activation=activation,
                         bias=cfg.model.bias,
                         batch_norm=cfg.model.batch_norm)

    assert cfg.system.pred_horizon >= 1

    if cfg.model.name.lower() in ["dae", "dae-aug"]:
        from kooplearn.models.ae.dynamic import DynamicAE
        from morpho_symm.nn.MLP import MLP

        encoder_kwargs = dict(in_dim=state_dim, out_dim=latent_state_dim, **obs_fn_params)
        decoder_kwargs = dict(in_dim=latent_state_dim, out_dim=state_dim, **obs_fn_params)
        model = DynamicAE(encoder=MLP, encoder_kwargs=encoder_kwargs,
                          decoder=MLP, decoder_kwargs=decoder_kwargs,
                          latent_dim=latent_state_dim,
                          loss_weights=None,
                          evolution_op_init_mode=cfg.model.evolution_op_init_mode, )

    elif cfg.model.name.lower() == "e-dae" or cfg.model.name.lower() == "edae":
        from morpho_symm.utils.abstract_harmonics_analysis import isotypic_basis
        from kooplearn.models.ae.equiv_dynamic import EquivDynamicAE
        from morpho_symm.nn.EMLP import EMLP

        multiplicity = math.ceil(latent_state_dim / state_type.size)
        # Define the observation space representation in the isotypic basis.
        obs_iso_reps, obs_iso_dims = isotypic_basis(representation=state_type.representation,
                                                    multiplicity=multiplicity,
                                                    prefix='LatentState')
        latent_state_type = FieldType(state_type.gspace, [rep_iso for rep_iso in obs_iso_reps.values()])

        encoder_kwargs = dict(in_type=state_type, out_type=latent_state_type, **obs_fn_params)
        decoder_kwargs = dict(in_type=latent_state_type, out_type=state_type, **obs_fn_params)

        model = EquivDynamicAE(encoder=EMLP, encoder_kwargs=encoder_kwargs,
                               decoder=EMLP, decoder_kwargs=decoder_kwargs,
                               loss_weights=None,
                               evolution_op_init_mode=cfg.model.evolution_op_init_mode)

    elif cfg.model.name.lower() == "e-dpnet":
        assert cfg.model.max_ck_window_length <= cfg.system.pred_horizon, "max_ck_window_length <= pred_horizon"
        from nn.EquivDeepPojections import EquivDPNet
        model = EquivDPNet(state_rep=datamodule.state_type.representation,
                           obs_state_dim=latent_state_dim,
                           max_ck_window_length=cfg.model.max_ck_window_length,
                           dt=datamodule.dt,
                           ck_w=cfg.model.ck_w,
                           orth_w=cfg.model.orth_w,
                           use_spectral_score=cfg.model.use_spectral_score,
                           enforce_constant_fn=cfg.model.constant_function,
                           explicit_transfer_op=cfg.model.explicit_transfer_op,
                           obs_fn_params=obs_fn_params,
                           group_avg_trick=cfg.model.group_avg_trick)
    elif cfg.model.name.lower() == "dpnet":
        assert cfg.model.max_ck_window_length <= cfg.system.pred_horizon, "max_ck_window_length <= pred_horizon"
        from nn.DeepProjections import DPNet
        model = DPNet(state_dim=datamodule.state_type.size,
                      obs_state_dim=latent_state_dim,
                      max_ck_window_length=cfg.model.max_ck_window_length,
                      dt=datamodule.dt,
                      ck_w=cfg.model.ck_w,
                      orth_w=cfg.model.orth_w,
                      use_spectral_score=cfg.model.use_spectral_score,
                      enforce_constant_fn=cfg.model.constant_function,
                      explicit_transfer_op=cfg.model.explicit_transfer_op,
                      obs_fn_params=obs_fn_params)
    else:
        raise NotImplementedError(f"Model {cfg.model.name} not implemented")
    log.info(f"Model \n {model}")
    # raise NotImplementedError("Testing the output")
    return model


if __name__ == '__main__':
    main()
    # return r
