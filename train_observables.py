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
from nn.DeepProjectionNetworks import DeepProjectionNet, EquivDeepProjectionNet
from nn.EquivDynamicsAutoencoder import EquivDynamicsAutoEncoder
from nn.LightningModel import LightningModel
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
    ckpt_call = ModelCheckpoint(dirpath=ckpt_folder_path, filename='best', monitor="val/loss", save_last=True)
    training_done, ckpt_path, best_path = check_if_resume_experiment(ckpt_call)

    if not training_done:
        # Load the dynamics dataset.
        data_path = root_path / "data" / cfg.system.data_path
        device = torch.device("cuda" if torch.cuda.is_available() and cfg.device != "cpu" else "cpu")
        datamodule = DynamicsDataModule(data_path,
                                        batch_size=cfg.model.batch_size,
                                        frames_per_step=cfg.system.frames_per_state,
                                        pred_horizon=cfg.system.pred_horizon,
                                        eval_pred_horizon=cfg.system.eval_pred_horizon,
                                        num_workers=cfg.num_workers,
                                        device=device,
                                        augment=cfg.model.augment)
        datamodule.prepare_data()
        group = datamodule.symm_group
        obs_state_dim = math.ceil(cfg.system.obs_state_dim / group.order())
        num_hidden_neurons = max(32, 2 * obs_state_dim)

        # Get the selected model for observation learning _____________________________________________________________
        activation = class_from_name('escnn.nn' if cfg.model.equivariant else 'torch.nn', cfg.model.activation)
        model_agnostic_params = dict(obs_state_dimension=obs_state_dim,
                                     num_encoder_layers=cfg.model.n_layers,
                                     num_encoder_hidden_neurons=num_hidden_neurons,
                                     activation=activation)
        if cfg.model.name.lower() == "dae":
            if cfg.model.equivariant:
                model = EquivDynamicsAutoEncoder(**model_agnostic_params,
                                                 state_type=datamodule.state_field_type,
                                                 dt=datamodule.dt)
                assert cfg.system.pred_horizon >= 2, "DAE requires at least 2 steps prediction horizon"
        elif cfg.model.name.lower() == "dpnet":
            assert cfg.model.max_ck_window_length <= cfg.system.pred_horizon, "max_ck_window_length <= pred_horizon"
            if cfg.model.equivariant:
                model = EquivDeepProjectionNet(**model_agnostic_params,
                                               state_type=datamodule.state_field_type,
                                               max_ck_window_length=cfg.model.max_ck_window_length,
                                               ck_w=cfg.model.ck_w,
                                               orthonormal_w=cfg.model.orthonormal_w)
            else:
                model = DeepProjectionNet(**model_agnostic_params,
                                          state_dim=datamodule.state_field_type.size,
                                          max_ck_window_length=cfg.model.max_ck_window_length,
                                          ck_w=cfg.model.ck_w,
                                          orthonormal_w=cfg.model.orthonormal_w)
        else:
            raise NotImplementedError(f"Model {cfg.model.name} not implemented")

        stop_call = EarlyStopping(monitor='val/loss', patience=max(5, int(cfg.max_epochs * 0.1)), mode='min')
        # Get the Hyperparameters for the run
        run_hps = OmegaConf.to_container(cfg, resolve=True)
        run_hps['dynamics_parameters'] = datamodule.metadata.dynamics_parameters
        run_hps['model']['num_hidden_neurons'] = num_hidden_neurons

        run_name = run_path.name
        wandb_logger = WandbLogger(project=f'{cfg.exp_name}',
                                   save_dir=seed_path.absolute(),
                                   config=run_hps,
                                   group=format_scientific(run_name),
                                   # offline=True,
                                   job_type='debug' if cfg.debug else None)

        # Configure Lightning trainer
        trainer = Trainer(accelerator='cuda' if torch.cuda.is_available() and device != 'cpu' else 'cpu',
                          devices='auto',  # 1 if torch.cuda.is_available() and device != 'cpu' else 1,
                          logger=wandb_logger,
                          log_every_n_steps=25,
                          max_epochs=cfg.max_epochs if not cfg.debug_loops else 2,
                          check_val_every_n_epoch=1,
                          # benchmark=True,
                          callbacks=[ckpt_call, stop_call],
                          fast_dev_run=10 if cfg.debug else False,
                          # detect_anomaly=cfg.debug, # This shit slows down to the point of gen existential dread.
                          enable_progress_bar=cfg.debug_loops or cfg.debug,
                          limit_train_batches=5 if cfg.debug_loops else 1.0,
                          limit_test_batches=10 if cfg.debug_loops else 1.0,
                          limit_val_batches=10 if cfg.debug_loops else 1.0,
                          )

        # Load lightning module handling the operations of all model variants
        epoch_metrics_fn = model.evaluate_observation_space if hasattr(model, "evaluate_observation_space") else None

        pl_model = LightningModel(lr=cfg.model.lr,
                                  batch_size=cfg.model.batch_size,
                                  run_hps=cfg.model,
                                  test_epoch_metrics_fn=epoch_metrics_fn,
                                  val_epoch_metrics_fn=epoch_metrics_fn)
        pl_model.set_model(model)
        # pl_model.to(device)
        wandb_logger.watch(model, log_graph=False, log='all', log_freq=10)

        # trainer.test(model=pl_model, datamodule=datamodule)

        # profiler = cProfile.Profile()
        # profiler.enable()
        log.info("\nTraining Started\n")
        trainer.fit(model=pl_model, datamodule=datamodule)
        log.info("\nTraining Done\n")

        # profiler.disable()

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
        try:
            test_pred_loss = results[0]['test/pred_loss_avg']
        except:
            test_pred_loss = results[0]['test/pred_loss']
        wandb_logger.experiment.unwatch(model)
        wandb_logger.experiment.finish()
        return test_pred_loss
    else:
        log.warning(f"Training run done. Check {run_path} for results.")


if __name__ == '__main__':
    main()
    # return r
