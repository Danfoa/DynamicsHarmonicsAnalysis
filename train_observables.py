import logging
import os
from pathlib import Path

import example_robot_data
import hydra
import numpy as np
import torch
from hydra.utils import get_original_cwd
from lightning.pytorch.loggers import WandbLogger
from lightning_fabric import seed_everything
from omegaconf import DictConfig
from pytorch_lightning import Trainer

log = logging.getLogger(__name__)

from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from data.ClosedLoopDynamicsDataModule import ClosedLoopDynDataModule
from nn.DynamicsAutoencoder import DynamicsAutoEncoder
from nn.LightningModel import LightningModel
from nn.VAMP import VAMP

try:
    from src.RobotEquivariantNN.groups.SparseRepresentation import SparseRep
    from src.RobotEquivariantNN.groups.SymmetryGroups import C2
except ImportError:
    raise Exception("run `git submodule update --init")

from utils.mysc import check_if_resume_experiment, class_from_name


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
    run_name = run_path.name
    wandb_logger = WandbLogger(project=f'{cfg.system}-{cfg.exp_name}',
                               save_dir=seed_path, config=cfg, group=run_name,
                               job_type='debug' if cfg.debug else None)

    ckpt_folder_path = seed_path
    ckpt_call = ModelCheckpoint(dirpath=ckpt_folder_path, filename='best', monitor="val/loss", save_last=True)
    training_done, ckpt_path, best_path = check_if_resume_experiment(ckpt_call)
    # test_metrics_path = Path(tb_logger.log_dir) / 'test_metrics.csv'

    if not training_done:
        stop_call = EarlyStopping(monitor='val/loss', patience=max(10, int(cfg.max_epochs * 0.1)), mode='min')

        log.info("Initiating Training\n")
        # Configure Lightning trainer
        trainer = Trainer(gpus=1 if torch.cuda.is_available() and device != 'cpu' else 0,
                          logger=wandb_logger,
                          accelerator="auto",
                          log_every_n_steps=50,
                          max_epochs=cfg.max_epochs if not cfg.debug_loops else 2,
                          check_val_every_n_epoch=1,
                          benchmark=True,
                          callbacks=[ckpt_call, stop_call],
                          fast_dev_run=10 if cfg.debug else False,
                          detect_anomaly=cfg.debug,
                          enable_progress_bar=cfg.debug_loops or cfg.debug,
                          limit_train_batches=10 if cfg.debug_loops else 1.0,
                          limit_test_batches=10 if cfg.debug_loops else 1.0,
                          limit_val_batches=10 if cfg.debug_loops else 1.0,
                          resume_from_checkpoint=ckpt_path if ckpt_path.exists() else None,
                          )

        data_path = root_path / "data" / cfg.robot
        datamodule = ClosedLoopDynDataModule(data_path, batch_size=cfg.model.batch_size,
                                             pred_horizon=cfg.model.pred_horizon,
                                             num_workers=cfg.num_workers, device=device, augment=cfg.model.augment,
                                             rep_state=rep_state, rep_ctrl=rep_crtl,
                                             robot=robot.model,
                                             dynamic_regime=cfg.dynamic_regime)
        datamodule.prepare_data()

        # Get the selected model for observation learning _____________________________________________________________
        activation = class_from_name('torch.nn', cfg.model.activation)
        dt = 1e-2  # TODO: Get from trajectory meta-data
        shared_model_params = dict(obs_dim=cfg.model.obs_dim, dt=dt, activation=activation, robot=robot.model,
                                   n_hidden_neurons=cfg.model.hidden_neurons, n_layers=cfg.model.n_layers)
        if cfg.model.name == "VAMP":
            if cfg.model.equivariance:
                raise NotImplementedError("VAMP is not implemented with equivariance yet")
            else:
                # Current implementation of this variant needs to know in advance how many timesteps we will use
                # since a NN head needs to be created for each timestep (This is what I want to check if we can avoid)
                model = VAMP(state_dim=repX.symm_group.d, pred_horizon=cfg.model.pred_horizon,
                             reg_lambda=cfg.model.reg_w,
                             **shared_model_params)

        elif cfg.model.name == "DAE":
            # Model output in state coordinates is unstandarized to get original coordinates
            nn_input_mean = torch.cat((datamodule.train_dataset._state_mean, datamodule.train_dataset._ctrl_mean)).to(
                torch.float32)
            nn_inout_std = torch.cat((datamodule.train_dataset._state_scale, datamodule.train_dataset._ctrl_scale)).to(
                torch.float32)
            dae_params = dict(respect_state_topology=cfg.model.state_topology, pred_w=cfg.model.loss_pred_w,
                              eigval_init=cfg.model.eigval_init, eigval_constraint=cfg.model.eigval_constraint,
                              input_mean=nn_input_mean, input_std=nn_inout_std, )
            if cfg.model.equivariance:
                model = EDynamicsAutoEncoder(repX=repX, **dae_params, **shared_model_params)
            else:
                model = DynamicsAutoEncoder(state_dim=repX.symm_group.d, **dae_params, **shared_model_params)
        else:
            raise NotImplementedError(f"Model {cfg.model.name} not implemented")
        model.to(device)

        # Load lightning module handling the operations of all model variants
        epoch_metrics_fn = model.evaluate_observation_space if hasattr(model, "evaluate_observation_space") else None
        pl_model = LightningModel(lr=cfg.model.lr, batch_size=cfg.model.batch_size, run_hps=cfg.model,
                                  test_epoch_metrics_fn=epoch_metrics_fn, val_epoch_metrics_fn=epoch_metrics_fn)
        pl_model.set_model(model)
        pl_model.to(device)
        wandb_logger.watch(pl_model, log_graph=False, log='all')

        # ____________________________________________________________________________________________________________

        #
        trainer.fit(model=pl_model, datamodule=datamodule)

        # Plot performance of best model
        if not cfg.debug:
            log.info("Loading best model and testing")
            best_ckpt = torch.load(best_path)
            pl_model.load_state_dict(best_ckpt['state_dict'])  #

        trainer.test(model=pl_model, datamodule=datamodule)
        pl_model.to(device)

        datamodule.plot_test_performance(pl_model, dataset=datamodule.test_dataset, log_prefix="test/",
                                         log_fig=True, show=cfg.debug_loops or cfg.debug,
                                         eigvals=pl_model.model.obs_state_dynamics.eigvals.cpu().detach().numpy())
        datamodule.plot_test_performance(pl_model, dataset=datamodule.val_dataset, log_prefix="val/",
                                         log_fig=True, show=cfg.debug_loops or cfg.debug,
                                         eigvals=pl_model.model.obs_state_dynamics.eigvals.cpu().detach().numpy())
        # if cfg.debug:
        #     datamodule.plot_test_performance(pl_model, dataset=datamodule.train_dataset, show=True,
        #                                      eigvals=pl_model.model.observation_dynamics.eigvals.cpu().detach(
        #                                      ).numpy())

    else:
        log.warning(f"Training run done. Check {run_path} for results.")

    wandb_logger.experiment.unwatch(pl_model)


if __name__ == '__main__':
    main()
    # Dan5491355
