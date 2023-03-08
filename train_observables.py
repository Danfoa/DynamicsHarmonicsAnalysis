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
from src.RobotEquivariantNN.groups.SparseRepresentation import SparseRep
from src.RobotEquivariantNN.groups.SymmetryGroups import C2


def get_robot_data(robot_name, work_path: Path):
    data_path = work_path / "data" / robot_name
    assert data_path.exists(), f"No dataset for {robot_name} in {data_path.absolute()}"

    data = {}
    train_list = list(data_path.rglob("*train.pickle"))
    for file in train_list:
        dynamic_mode = file.parent.parent.stem
        data = {dynamic_mode: {
            "train": file,
            "test": file.with_stem("test"),
            "val": file.with_stem("val")
        }
        }
    return data


def get_datasets(cfg: DictConfig, device: str, work_path: Path, robot: RobotWrapper=None):
    robot_data = get_robot_data(cfg.robot, work_path)
    dynamic_regime = "unstable_fix_point"
    train_dataset = ClosedLoopDynDataset(path=robot_data[dynamic_regime]["train"], normalize=True, robot=robot,
                                         device=device, augment=cfg.augment, window_size=cfg.window_size)
    test_dataset = ClosedLoopDynDataset(path=robot_data[dynamic_regime]["test"], normalize=True, robot=robot,
                                        device=device, augment=cfg.augment, window_size=cfg.window_size,
                                        state_scaler=train_dataset.standard_scaler[STATES],
                                        ctrl_scaler=train_dataset.standard_scaler[CTRLS])
    val_dataset = ClosedLoopDynDataset(path=robot_data[dynamic_regime]["val"], normalize=True, robot=robot,
                                       device=device, augment=cfg.augment, window_size=cfg.window_size,
                                       state_scaler=train_dataset.standard_scaler[STATES],
                                       ctrl_scaler=train_dataset.standard_scaler[CTRLS])

    train_loader = DataLoader(dataset=train_dataset, batch_size=cfg.batch_size, shuffle=True, sampler=None,
                              collate_fn=lambda x: train_dataset.collate_fn(x), num_workers=cfg.num_workers)
    test_loader = DataLoader(dataset=test_dataset, batch_size=cfg.batch_size, shuffle=False, sampler=None,
                             collate_fn=lambda x: test_dataset.collate_fn(x), num_workers=cfg.num_workers)
    val_loader = DataLoader(dataset=val_dataset, batch_size=cfg.batch_size, shuffle=False, sampler=None,
                            collate_fn=lambda x: val_dataset.collate_fn(x), num_workers=cfg.num_workers)

    return (train_dataset, test_dataset, val_dataset), (train_loader, test_loader, val_loader)


@hydra.main(config_path='cfg', config_name='config')
def main(cfg: DictConfig):
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
    ckpt_path = Path(ckpt_call.dirpath).joinpath(ckpt_call.CHECKPOINT_NAME_LAST + ckpt_call.FILE_EXTENSION)
    best_path = Path(ckpt_call.dirpath).joinpath(ckpt_call.filename + ckpt_call.FILE_EXTENSION)
    test_metrics_path = Path(tb_logger.log_dir) / 'test_metrics.csv'

    stop_call = EarlyStopping(monitor='val_loss', patience=max(10, int(cfg.max_epochs * 0.1)), mode='min')

    log.info("\n\nInitiating Training\n\n")
    # Configure Lightning trainer
    trainer = Trainer(gpus=1 if torch.cuda.is_available() and device != 'cpu' else 0,
                      logger=tb_logger,
                      accelerator="auto",
                      log_every_n_steps=50,
                      max_epochs=cfg.max_epochs if not cfg.debug_loops else 3,
                      check_val_every_n_epoch=1,
                      benchmark=True,
                      callbacks=[ckpt_call, stop_call],
                      fast_dev_run=10 if cfg.debug else False,
                      detect_anomaly=cfg.debug,
                      enable_progress_bar=True,  # cfg.debug_loops or cfg.debug,
                      limit_train_batches=10 if cfg.debug_loops else False,
                      limit_test_batches=10 if cfg.debug_loops else False,
                      limit_val_batches=10 if cfg.debug_loops else False,
                      resume_from_checkpoint=ckpt_path if ckpt_path.exists() else None,
                      )

    # Loading the double pendulum model
    double_pendulum_rw = example_robot_data.load('double_pendulum')
    # Build a single DoF pendulum from the double pendulum model by fixing elbow joint
    pendulum_rw = double_pendulum_rw.buildReducedRobot(list_of_joints_to_lock=[2])
    robot_model = pendulum_rw.model
    G = C2(generators=[C2.oneline2matrix([0, 1, 2], reflexions=[-1, -1, -1])])
    repX = SparseRep(G)

    data_path = root_path / "data" / cfg.robot
    datamodule = ClosedLoopDynDataModule(data_path, batch_size=cfg.batch_size, window_size=cfg.window_size,
                                         num_workers=cfg.num_workers, device=device, augment=cfg.augment,
                                         robot=robot_model)
    # Hack for dt
    datamodule.prepare_data()

    # Get DynAE model.
    obs_dim = cfg.obs_dim
    if cfg.equivariance:
        model = EDynamicsAutoEncoder(repX=repX, obs_dim=obs_dim, dt=datamodule.dt)
    else:
        model = DynamicsAutoEncoder(in_dim=G.d, obs_dim=obs_dim, dt=datamodule.dt)

    # edae_model = EDynamicsAutoEncoder(repX=repX, obs_dim=obs_dim)


    hparams = model.get_hparams()
    pl_model = LightningModel(model=model, lr=cfg.lr, batch_size=cfg.batch_size,
                              loss_metrics_fn=datamodule.losses_and_metrics,
                              batch_unpack_fn=datamodule.state_ctrl_to_x)

    trainer.fit(model=pl_model, datamodule=datamodule)

    datamodule.plot_test_performance(pl_model, log_fig=True, show=cfg.debug)
    print("E")


if __name__ == '__main__':
    main()
