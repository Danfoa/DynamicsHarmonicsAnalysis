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
        device = torch.device(f"cuda:{cfg.device}" if torch.cuda.is_available() and cfg.device != "cpu" else "cpu")
        log.info(f"Configuring to use device {device}")
        log.info(f"Loading data from {data_path}")
        # Get the Lightning data module handling training/test/val data loaders
        datamodule = DynamicsDataModule(data_path,
                                        batch_size=cfg.model.batch_size,
                                        frames_per_step=cfg.system.frames_per_state,
                                        pred_horizon=cfg.system.pred_horizon,
                                        eval_pred_horizon=cfg.system.eval_pred_horizon,
                                        test_pred_horizon=cfg.system.test_pred_horizon,
                                        system_cfg=cfg.system,
                                        num_workers=cfg.num_workers,
                                        device=device,
                                        train_ratio=cfg.system.train_ratio,
                                        augment=cfg.model.augment,
                                        state_obs=cfg.system.get('state_obs', None),
                                        action_obs=cfg.system.get('action_obs', None),
                                        standardize=cfg.system.standardize)
        datamodule.prepare_data()
        if cfg.system.state_dim != '??':
            assert datamodule.state_type.size == cfg.system.state_dim, \
                f"State dim mismatch {datamodule.state_type.size} != {cfg.system.state_dim}"
        # Get the MarkovDynamics model to train _________________________________________________________________
        model = get_model(cfg, datamodule)

        stop_call = EarlyStopping(monitor='loss/val',
                                  mode='min',
                                  patience=max(cfg.system.early_stop_epochs, int(cfg.system.max_epochs * 0.1)))
        # Get the Hyperparameters for the run
        run_hps = OmegaConf.to_container(cfg, resolve=True)
        # run_hps['dynamics_parameters'] = datamodule.metadata.dynamics_parameters

        run_name = run_path.name
        wandb_logger = WandbLogger(project=f'{cfg.system.name}',
                                   save_dir=seed_path.absolute(),
                                   config=run_hps,
                                   name=run_name,
                                   group=f'{cfg.exp_name}',
                                   job_type='debug' if (cfg.debug or cfg.debug_loops) else None)

        # Configure Lightning trainer
        trainer = Trainer(accelerator='cuda' if torch.cuda.is_available() and cfg.device != 'cpu' else 'cpu',
                          devices=[cfg.device] if torch.cuda.is_available() and cfg.device != 'cpu' else 'auto',
                          logger=wandb_logger,
                          log_every_n_steps=1,
                          max_epochs=cfg.system.max_epochs if not cfg.debug_loops else 2,
                          check_val_every_n_epoch=2,
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

        if cfg.debug_loops:
            profiler = cProfile.Profile()
            profiler.enable()

        # Flag to track training success
        training_successful = False
        try:
            # Train the model
            trainer.fit(pl_model, datamodule=datamodule)
            # If training is successful, update the flag
            training_successful = True
        except Exception as e:
            # Handle the exception (log it, etc.)
            raise e

        # Create a pstats object
        if cfg.debug_loops:
            stats = pstats.Stats(profiler)
            stats.sort_stats('cumulative')  # Sort stats by the cumulative time spent in the function
            stats.print_stats('koopman_robotics')

        if training_successful:
            if not cfg.debug :  # Loading best model and test it
                if best_path.exists():
                    best_ckpt = torch.load(best_path)
                    pl_model.eval()
                    pl_model.model.eval()
                    pl_model.load_state_dict(best_ckpt['state_dict'], strict=False)
                else:
                    log.warning(f"Best model not found, testing with latest model")
            # Test best model. Selected as the model with lowest evaluation loss during training.
            results = trainer.test(model=pl_model, datamodule=datamodule)
            test_pred_loss = results[0]['obs_pred_loss/test']
            # wandb_logger.experiment.unwatch(model)
            wandb_logger.experiment.finish()
            return test_pred_loss
        else:
            raise RuntimeError("Training failed. Check logs for details.")
    else:
        log.warning(f"Training run done. Check {run_path} for results.")





def get_model(cfg, datamodule):
    state_dim = datamodule.state_type.size
    obs_state_dim = math.ceil(cfg.system.obs_state_ratio * state_dim)
    num_hidden_neurons = cfg.model.num_hidden_units

    if obs_state_dim > num_hidden_neurons:
        # Set num_hidden_neurons to be the closest power of 2 to obs_state_dim from above
        # For obs_state_dim=210 -> num_hidden_neurons=256
        num_hidden_neurons = 2 ** math.ceil(math.log2(obs_state_dim))
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

    if cfg.model.name.lower() in ["dae", "dae-aug"]:
        assert cfg.system.pred_horizon >= 1
        model = DAE(state_dim=state_dim,
                    obs_state_dim=obs_state_dim,
                    dt=datamodule.dt,
                    obs_pred_w=cfg.model.obs_pred_w,
                    orth_w=cfg.model.orth_w,
                    corr_w=cfg.model.corr_w,
                    obs_fn_params=obs_fn_params,
                    enforce_constant_fn=cfg.model.constant_function,
                    reuse_input_observable=cfg.model.reuse_input_observable,
                    )
    elif cfg.model.name.lower() == "e-dae":
        assert cfg.system.pred_horizon >= 1
        model = EquivDAE(state_rep=datamodule.state_type.representation,
                         obs_state_dim=obs_state_dim,
                         dt=datamodule.dt,
                         orth_w=cfg.model.orth_w,
                         obs_fn_params=obs_fn_params,
                         group_avg_trick=cfg.model.group_avg_trick,
                         state_dependent_obs_dyn=cfg.model.state_dependent_obs_dyn,
                         enforce_constant_fn=cfg.model.constant_function,
                         reuse_input_observable=cfg.model.reuse_input_observable,
                         )
    elif cfg.model.name.lower() == "e-dpnet":
        assert cfg.model.max_ck_window_length <= cfg.system.pred_horizon, "max_ck_window_length <= pred_horizon"
        model = EquivDPNet(state_rep=datamodule.state_type.representation,
                           obs_state_dim=obs_state_dim,
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
        model = DPNet(state_dim=datamodule.state_type.size,
                      obs_state_dim=obs_state_dim,
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
