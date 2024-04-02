import cProfile
import logging
import math
import os
import pstats
from pathlib import Path

import hydra
import numpy as np
import torch
from escnn.nn import FieldType
from morpho_symm.data.DynamicsRecording import DynamicsRecording, get_dynamics_dataset, get_train_test_val_file_paths

from hydra.utils import get_original_cwd
from lightning import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from lightning_fabric import seed_everything
from morpho_symm.nn.MLP import MLP
from omegaconf import DictConfig, OmegaConf
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
                                        data_augmentation=cfg.model.augment,
                                        state_obs=cfg.system.get('state_obs', None),
                                        standardize=cfg.system.standardize)

        datamodule.prepare_data()
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

        # Create a pstats object
        if cfg.debug_loops:
            stats = pstats.Stats(profiler)
            stats.sort_stats('cumulative')  # Sort stats by the cumulative time spent in the function
            stats.print_stats('koopman_robotics')

        if training_successful:
            if not cfg.debug:  # Loading best model and test it
                if best_path.exists():
                    best_ckpt = torch.load(best_path)
                    pl_latent_dyn_model.eval()
                    pl_latent_dyn_model.load_state_dict(best_ckpt['state_dict'], strict=False)
                    pass
                else:
                    log.warning(f"Best model not found, testing with latest model")
            # Test best model. Selected as the model with lowest evaluation loss during training.
            results = pl_trainer.test(model=pl_latent_dyn_model, datamodule=datamodule)
            test_loss = results[0]['loss/test']
            # wandb_logger.experiment.unwatch(model)
            wandb_logger.experiment.finish()
            return test_loss
        else:
            raise RuntimeError("Training failed. Check logs for details.")
    else:
        log.warning(f"Training run done. Check {run_path} for results.")


def configure_experiment_trainer(cfg, ckpt_call, run_path, seed_path):
    stop_call = EarlyStopping(monitor='loss/val',
                              mode='min',
                              patience=max(cfg.system.early_stop_epochs, int(cfg.system.max_epochs * 0.1)))
    # Get the Hyperparameters for the run
    run_hps = OmegaConf.to_container(cfg, resolve=True)
    run_name = run_path.name
    wandb_logger = WandbLogger(project=f'{cfg.system.name}',
                               save_dir=seed_path.absolute(),
                               config=run_hps,
                               name=run_name,
                               group=f'{cfg.exp_name}',
                               job_type='debug' if (cfg.debug or cfg.debug_loops) else None)
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
        from kooplearn.models.ae.dynamic import DynamicAE

        encoder_kwargs = dict(in_dim=state_dim, out_dim=obs_state_dim, **obs_fn_params)
        decoder_kwargs = dict(in_dim=obs_state_dim, out_dim=state_dim, **obs_fn_params)
        model = DynamicAE(encoder=MLP, encoder_kwargs=encoder_kwargs,
                          decoder=MLP, decoder_kwargs=decoder_kwargs,
                          latent_dim=obs_state_dim,
                          loss_weights=None)

    elif cfg.model.name.lower() == "e-dae":
        assert cfg.system.pred_horizon >= 1
        from nn.EquivDynamicsAutoencoder import EquivDAE
        model = EquivDAE(state_rep=datamodule.state_type.representation,
                         obs_state_dim=obs_state_dim,
                         dt=datamodule.dt,
                         orth_w=cfg.model.orth_w,
                         obs_fn_params=obs_fn_params,
                         group_avg_trick=cfg.model.group_avg_trick,
                         state_dependent_obs_dyn=cfg.model.state_dependent_obs_dyn,
                         enforce_constant_fn=cfg.model.constant_function,
                         # reuse_input_observable=cfg.model.reuse_input_observable,
                         )
    elif cfg.model.name.lower() == "e-dpnet":
        assert cfg.model.max_ck_window_length <= cfg.system.pred_horizon, "max_ck_window_length <= pred_horizon"
        from nn.EquivDeepPojections import EquivDPNet
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
        from nn.DeepProjections import DPNet
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
