import example_robot_data
import numpy as np
import torch
import hydra
from pathlib import Path

from omegaconf import DictConfig
from hydra.utils import get_original_cwd
from lightning_fabric import seed_everything
from pinocchio import RobotWrapper
from pytorch_lightning import loggers as pl_loggers, Trainer

import logging as log

from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch.utils.data import DataLoader

from data.ClosedLoopDynamics import ClosedLoopDynDataset, STATES, CTRLS
from data.ClosedLoopDynamicsDataModule import ClosedLoopDynDataModule
from nn.LightningModel import LightningModel
from nn.ObservableModules import EDynamicsAutoEncoder, DynamicsAutoEncoder

try:
    from src.RobotEquivariantNN.groups.SparseRepresentation import SparseRep
    from src.RobotEquivariantNN.groups.SymmetryGroups import C2
except ImportError as e:
    raise Exception("run `git submodule update --init")

from utils.mysc import check_if_resume_experiment


@hydra.main(config_path='cfg', config_name='config')
def main(cfg: DictConfig):
    torch.set_float32_matmul_precision('medium')
    log.info("\n\n NEW RUN \n\n")
    device = torch.device("cuda" if torch.cuda.is_available() and cfg.device != "cpu" else "cpu")
    cfg.seed = cfg.seed if cfg.seed >= 0 else np.random.randint(0, 1000)
    cfg['debug'] = cfg.get('debug', False)
    cfg['debug_loops'] = cfg.get('debug_loops', False)
    seed_everything(seed=cfg.seed)

    root_path = Path(get_original_cwd()).resolve()

    # Check CLI arguments/params

    # Check if experiment already run
    tb_logger = pl_loggers.TensorBoardLogger(".", name=f'seed={cfg.seed}', version=cfg.seed, default_hp_metric=False)
    ckpt_folder_path = Path(tb_logger.log_dir) / "ckpt"
    ckpt_call = ModelCheckpoint(dirpath=ckpt_folder_path, filename='best', monitor="val_loss", save_last=True)
    training_done, ckpt_path, best_path = check_if_resume_experiment(ckpt_call)
    # test_metrics_path = Path(tb_logger.log_dir) / 'test_metrics.csv'

    if not training_done:
        stop_call = EarlyStopping(monitor='val_loss', patience=max(10, int(cfg.model.max_epochs * 0.1)), mode='min')

        log.info("\nInitiating Training\n")
        # Configure Lightning trainer
        trainer = Trainer(gpus=1 if torch.cuda.is_available() and device != 'cpu' else 0,
                          logger=tb_logger,
                          accelerator="auto",
                          log_every_n_steps=50,
                          max_epochs=cfg.model.max_epochs if not cfg.debug_loops else 2,
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
        # b = trainer.device_ids
        # assert b[0] == 0, "Bad GPU allocation"
        # Loading the double pendulum model
        # TODO: Centralize robot loading in a single place
        double_pendulum_rw = example_robot_data.load('double_pendulum_continuous')
        # Build a single DoF pendulum from the double pendulum model by fixing elbow joint
        robot = double_pendulum_rw.buildReducedRobot(list_of_joints_to_lock=[2])
        # robot_model = pe/ndulum_rw.model
        G = C2(generators=[C2.oneline2matrix([0, 1, 2, 3], reflexions=[1, -1, -1, -1])])
        repX = SparseRep(G)
        assert G.d == robot.nq + (2 * robot.nv)

        data_path = root_path / "data" / cfg.model.robot
        datamodule = ClosedLoopDynDataModule(data_path, batch_size=cfg.model.batch_size,
                                             window_size=cfg.model.window_size,
                                             num_workers=cfg.num_workers, device=device, augment=cfg.model.augment,
                                             robot=robot.model, pred_w=cfg.model.loss_pred_w,
                                             dynamic_regime=cfg.model.dynamic_regime)

        # Get DynAE model.
        obs_dim = cfg.model.obs_dim
        if cfg.model.equivariance:
            model = EDynamicsAutoEncoder(repX=repX, obs_dim=obs_dim, dt=1e-2)
        else:
            model = DynamicsAutoEncoder(in_dim=G.d, obs_dim=obs_dim, dt=1e-2)
        model.to(device)
        # edae_model = EDynamicsAutoEncoder(repX=repX, obs_dim=obs_dim)

        pl_model = LightningModel(lr=cfg.model.lr, batch_size=cfg.model.batch_size, run_hps=cfg.model)
        pl_model.set_model(model)
        pl_model.to(device)

        trainer.fit(model=pl_model, datamodule=datamodule)

        # Plot performance of best model
        if not cfg.debug:
            log.info(f"Loading best model and testing")
            best_ckpt = torch.load(best_path)
            pl_model.load_state_dict(best_ckpt['state_dict'])  #

        trainer.test(model=pl_model, datamodule=datamodule)
        pl_model.to(device)
        datamodule.plot_test_performance(pl_model, log_fig=True, show=cfg.debug,
                                         eigvals=pl_model.model.observation_dynamics.eigvals.cpu().detach().numpy())
    else:
        log.warning(f"Training run done. Check {Path(tb_logger.log_dir).absolute()} for results.")


if __name__ == '__main__':
    main()
    # Dan5491355
