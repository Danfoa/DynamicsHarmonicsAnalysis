import cProfile
import logging
import math
import os
import pstats
from pathlib import Path

import hydra
import numpy as np
import torch
from hydra.utils import get_original_cwd
from lightning import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from lightning_fabric import seed_everything
from omegaconf import DictConfig, OmegaConf

from data.DynamicsDataModule import DynamicsDataModule
from nn.DeepProjections import DPNet
from nn.DynamicsAutoEncoder import DAE
from nn.EquivDeepPojections import EquivDPNet
from nn.EquivDynamicsAutoencoder import EquivDAE
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
        device = torch.device(f"cuda:{cfg.device}" if torch.cuda.is_available() and cfg.device != "cpu" else None)
        log.info(f"Configuring to use device {device}")
        datamodule = DynamicsDataModule(data_path,
                                        batch_size=cfg.model.batch_size,
                                        frames_per_step=cfg.system.frames_per_state,
                                        pred_horizon=cfg.system.pred_horizon,
                                        eval_pred_horizon=cfg.system.eval_pred_horizon,
                                        system_cfg=cfg.system,
                                        num_workers=cfg.num_workers,
                                        device=device,
                                        augment=cfg.model.augment)
        datamodule.prepare_data()
        obs_state_dim = math.ceil(
            cfg.system.obs_state_dim / datamodule.state_field_type.size) * datamodule.state_field_type.size
        num_hidden_neurons = max(cfg.model.num_hidden_units, 2 * obs_state_dim)

        # Get the selected model for observation learning _____________________________________________________________
        if cfg.model.equivariant:
            activation = cfg.model.activation
        else:
            activation = class_from_name('torch.nn', cfg.model.activation)

        obs_fn_params = dict(num_layers=cfg.model.num_layers,
                             num_hidden_units=num_hidden_neurons,
                             activation=activation,
                             bias=cfg.model.bias,
                             batch_norm=cfg.model.batch_norm,
                             )

        if cfg.model.name.lower() == "dae":
            assert cfg.system.pred_horizon >= 1
            model = DAE(state_dim=datamodule.state_field_type.size,
                        obs_state_dim=obs_state_dim,
                        dt=datamodule.dt,
                        obs_pred_w=cfg.model.obs_pred_w,
                        orth_w=cfg.model.orth_w,
                        corr_w=cfg.model.corr_w,
                        obs_fn_params=obs_fn_params,
                        enforce_constant_fn=cfg.model.constant_function,
                        )
        elif cfg.model.name.lower() == "e-dae":
            assert cfg.system.pred_horizon >= 1
            model = EquivDAE(state_rep=datamodule.state_field_type.representation,
                             obs_state_dim=obs_state_dim,
                             dt=datamodule.dt,
                             orth_w=cfg.model.orth_w,
                             obs_fn_params=obs_fn_params,
                             group_avg_trick=cfg.model.group_avg_trick,
                             state_dependent_obs_dyn=cfg.model.state_dependent_obs_dyn,
                             enforce_constant_fn=cfg.model.constant_function,
                             )


        elif cfg.model.name.lower() == "e-dpnet":
            assert cfg.model.max_ck_window_length <= cfg.system.pred_horizon, "max_ck_window_length <= pred_horizon"
            model = EquivDPNet(state_rep=datamodule.state_field_type.representation,
                               obs_state_dim=obs_state_dim,
                               max_ck_window_length=cfg.model.max_ck_window_length,
                               dt=datamodule.dt,
                               ck_w=cfg.model.ck_w,
                               orth_w=cfg.model.orth_w,
                               use_spectral_score=cfg.model.use_spectral_score,
                               enforce_constant_fn=cfg.model.constant_function,
                               aux_obs_space=cfg.model.aux_obs_space,
                               obs_fn_params=obs_fn_params,
                               group_avg_trick=cfg.model.group_avg_trick)

        elif cfg.model.name.lower() == "dpnet":
            assert cfg.model.max_ck_window_length <= cfg.system.pred_horizon, "max_ck_window_length <= pred_horizon"
            model = DPNet(state_dim=datamodule.state_field_type.size,
                          obs_state_dim=obs_state_dim,
                          max_ck_window_length=cfg.model.max_ck_window_length,
                          dt=datamodule.dt,
                          ck_w=cfg.model.ck_w,
                          orth_w=cfg.model.orth_w,
                          use_spectral_score=cfg.model.use_spectral_score,
                          enforce_constant_fn=cfg.model.constant_function,
                          aux_obs_space=cfg.model.aux_obs_space,
                          obs_fn_params=obs_fn_params)
        else:
            raise NotImplementedError(f"Model {cfg.model.name} not implemented")

        log.info(f"Model \n {model}")

        stop_call = EarlyStopping(monitor='loss/val', mode='min', patience=max(10, int(cfg.max_epochs * 0.1)))
        # Get the Hyperparameters for the run
        run_hps = OmegaConf.to_container(cfg, resolve=True)
        run_hps['dynamics_parameters'] = datamodule.metadata.dynamics_parameters
        run_hps['model']['num_hidden_neurons'] = num_hidden_neurons

        run_name = run_path.name
        wandb_logger = WandbLogger(project=f'{cfg.system.name}',
                                   save_dir=seed_path.absolute(),
                                   config=run_hps,
                                   name=run_name,
                                   group=f'{cfg.exp_name}',
                                   job_type='debug' if (cfg.debug or cfg.debug_loops) else None)

        # Configure Lightning trainer
        trainer = Trainer(accelerator='cuda' if torch.cuda.is_available() and device != 'cpu' else 'cpu',
                          devices=[cfg.device] if torch.cuda.is_available() and device != 'cpu' else 1,
                          logger=wandb_logger,
                          log_every_n_steps=1,
                          max_epochs=cfg.max_epochs if not cfg.debug_loops else 2,
                          check_val_every_n_epoch=1,
                          callbacks=[ckpt_call, stop_call],
                          fast_dev_run=10 if cfg.debug else False,
                          enable_progress_bar=True,  # cfg.debug_loops or cfg.debug,
                          limit_train_batches=5 if cfg.debug_loops else 1.0,
                          limit_test_batches=10 if cfg.debug_loops else 1.0,
                          limit_val_batches=10 if cfg.debug_loops else 1.0,
                          )

        # Load lightning module handling the operations of all model variants
        epoch_metrics_fn = model.evaluate_observation_space if hasattr(model, "evaluate_observation_space") else None

        pl_model = LightLatentMarkovDynamics(lr=cfg.model.lr,
                                             batch_size=cfg.model.batch_size,
                                             run_hps=cfg.model,
                                             test_epoch_metrics_fn=epoch_metrics_fn,
                                             val_epoch_metrics_fn=epoch_metrics_fn,
                                             log_figs_every_n_epochs=10)
        pl_model.set_model(model)
        # pl_model.to(device)
        # wandb_logger.watch(model, log_graph=False, log='all', log_freq=10)

        # trainer.test(model=pl_model, datamodule=datamodule)

        # profiler = cProfile.Profile()
        # profiler.enable()
        log.info("\nTraining Started\n")
        trainer.fit(model=pl_model, datamodule=datamodule)
        log.info("\nTraining Done\n")

        if cfg.model == "dpnet":
            # Train non-linear inverse observable function
            log.info("\nTraining Inverse Observable\n")
            pl_model.model.train_mode = DPNet.INV_PROJECTION
            trainer.fit(model=pl_model, datamodule=datamodule)
            log.info("\nTraining Done\n")

        # # Create a pstats object
        # stats = pstats.Stats(profiler)
        # # Sort stats by the cumulative time spent in the function
        # stats.sort_stats('cumulative')
        # # Print only the info for the functions defined in your script
        # # Assuming your script's name is 'your_script.py'
        # stats.print_stats('koopman_robotics')
        # Plot performance of best model
        if not cfg.debug:
            log.info("Loading best model and testing")
            best_ckpt = torch.load(best_path)
            pl_model.eval()
            pl_model.model.eval()
            pl_model.load_state_dict(best_ckpt['state_dict'], strict=False)

        results = trainer.test(model=pl_model, datamodule=datamodule)
        test_pred_loss = results[0]['obs_pred_loss/test']
        # wandb_logger.experiment.unwatch(model)
        wandb_logger.experiment.finish()
        return test_pred_loss
    else:
        log.warning(f"Training run done. Check {run_path} for results.")


if __name__ == '__main__':
    main()
    # return r
