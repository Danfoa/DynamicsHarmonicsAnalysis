import warnings
from pathlib import Path
from typing import Union

import matplotlib.pyplot as plt
import pinocchio
import torch
from pinocchio import RobotWrapper
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from utils.complex import view_as_complex, interleave_with_conjugate
from .ClosedLoopDynamics import ClosedLoopDynDataset, STATES, CTRLS, FULL_TRAJ

import logging

log = logging.getLogger(__name__)


class ClosedLoopDynDataModule(pl.LightningDataModule):

    def __init__(self, data_path: Path, pred_horizon: Union[int, float] = 0.5, batch_size: int = 256,
                 num_workers: int = 0, device=None, robot: pinocchio.Model = None, augment: bool = False,
                 rep_state=None, rep_ctrl=None, dynamic_regime='all'):
        super().__init__()
        assert data_path.exists(), f"Data folder not found {data_path.absolute()}"
        self._data_path = data_path
        self.robot = robot
        self.augment = augment
        if isinstance(pred_horizon, float):
            assert 0 < pred_horizon <= 1.0, "Prediction horizon need be in H∈(0., 1.]. Predict H % of the trajectory"
            self.pred_horizon = pred_horizon
        elif isinstance(pred_horizon, int):
            assert pred_horizon >= 1, "At least we need to forecast a single dynamics step"
            self.pred_horizon = pred_horizon + 1
        self.device = device
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.prepared = False
        self.dynamic_regime = dynamic_regime
        self.rep_state = rep_state
        self.rep_ctrl = rep_ctrl

    def prepare_data(self):
        if self.prepared:
            return
        robot_data = self.get_robot_data()
        dynamic_regimes = list(robot_data.keys())
        if self.dynamic_regime != 'all':
            assert self.dynamic_regime in dynamic_regimes, f"No data found for {self.dynamic_regime}, available " \
                                                           f"regimes: {dynamic_regimes}"
            dynamic_regimes = [self.dynamic_regime]

        self.train_dataset = ClosedLoopDynDataset(path=[robot_data[d]["train"] for d in dynamic_regimes],
                                                  normalize=True, robot=self.robot, device=self.device,
                                                  augment=self.augment, pred_horizon=self.pred_horizon,
                                                  rep_state=self.rep_state, rep_ctrl=self.rep_ctrl)
        self.test_dataset = ClosedLoopDynDataset(path=[robot_data[d]["test"] for d in dynamic_regimes], normalize=True,
                                                 robot=self.robot, device=self.device, augment=False, # TODO: Change
                                                 pred_horizon=FULL_TRAJ,
                                                 state_scaler=self.train_dataset.standard_scaler[STATES],
                                                 ctrl_scaler=self.train_dataset.standard_scaler[CTRLS],
                                                 rep_state=self.rep_state, rep_ctrl=self.rep_ctrl)
        self.val_dataset = ClosedLoopDynDataset(path=[robot_data[d]["val"] for d in dynamic_regimes], normalize=True,
                                                robot=self.robot, device=self.device, augment=False, # TODO: Change
                                                pred_horizon=self.pred_horizon,
                                                state_scaler=self.train_dataset.standard_scaler[STATES],
                                                ctrl_scaler=self.train_dataset.standard_scaler[CTRLS],
                                                rep_state=self.rep_state, rep_ctrl=self.rep_ctrl)
        self.predict_dataset = ClosedLoopDynDataset(path=[robot_data[d]["test"] for d in dynamic_regimes],
                                                    normalize=True, robot=self.robot, device=self.device, augment=False,  # TODO: Change
                                                    pred_horizon=FULL_TRAJ,
                                                    state_scaler=self.train_dataset.standard_scaler[STATES],
                                                    ctrl_scaler=self.train_dataset.standard_scaler[CTRLS],
                                                    rep_state=self.rep_state, rep_ctrl=self.rep_ctrl)

        self.dt = self.train_dataset.dt
        self.prepared = True

    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        log.debug(f"Stage change to {stage}")

    def train_dataloader(self):
        return DataLoader(dataset=self.train_dataset, batch_size=self.batch_size, shuffle=True, sampler=None,
                          collate_fn=lambda x: self.train_dataset.collate_fn(x), num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(dataset=self.val_dataset, batch_size=self.batch_size, shuffle=False, sampler=None,
                          collate_fn=lambda x: self.val_dataset.collate_fn(x), num_workers=self.num_workers,
                          drop_last=False)

    def test_dataloader(self):
        return DataLoader(dataset=self.test_dataset, batch_size=self.batch_size, shuffle=False, sampler=None,
                          collate_fn=lambda x: self.test_dataset.collate_fn(x), num_workers=self.num_workers,
                          drop_last=False)

    def predict_dataloader(self):
        # IMPORTANT. Predict dataloader doesnt shuffle data
        return DataLoader(dataset=self.predict_dataset, batch_size=len(self.predict_dataset), shuffle=False,
                          sampler=None,
                          collate_fn=lambda x: self.predict_dataset.collate_fn(x), num_workers=self.num_workers,
                          drop_last=False)

    def get_robot_data(self):
        data = {}
        train_list = list(self._data_path.rglob("*train.pickle"))

        assert len(train_list) > 0, f"No trajectory data found in {self._data_path.absolute()}"

        for file in train_list:
            dynamic_mode = file.parent.parent.stem
            data[dynamic_mode] = {
                "train": file,
                "test": file.with_stem("test"),
                "val": file.with_stem("val")
            }
        return data

    def plot_test_performance(self, pl_model: pl.LightningModule, dataset: ClosedLoopDynDataset,
                              log_fig=False, show=False, eigvals=None, log_prefix=''):
        assert self.trainer is not None
        from utils.plotting import plotOCSolution, plotNdimTraj
        from torch.utils.tensorboard import SummaryWriter
        import numpy as np

        num_trajs = len(dataset)
        num_trajs_plot = 4

        idx = np.arange(num_trajs)
        gen = np.random.default_rng(12345)
        shuffled_idx = gen.permutation(idx)
        selected_trajs_idx = shuffled_idx[:num_trajs_plot]

        # Get ground truth
        gt = dataset.collate_fn([dataset[i] for i in selected_trajs_idx])
        # Get model predictions
        output = pl_model.predict_step(batch=gt, batch_idx=-1)
        pred = self.trainer.model.model.batch_pack(output)

        # Unstandarized ground truth data for plotting. TODO: Make a more elegant solution here
        gt[STATES] = (gt[STATES] * self.train_dataset._state_scale) + self.train_dataset._state_mean
        gt[CTRLS] = (gt[CTRLS] * self.train_dataset._ctrl_scale) + self.train_dataset._ctrl_mean

        fig, axs, fig_z, axs_z = None, None, None, None
        cmap = plt.cm.get_cmap('inferno', num_trajs_plot + 1)
        for i, (state, ctrl, state_pred, ctrl_pred) in enumerate(zip(gt[STATES], gt[CTRLS], pred[STATES], pred[CTRLS])):
            n_gt = torch.norm(state_pred[:, [0, 1]], dim=-1)
            n_pred = torch.norm(state[:, [0, 1]], dim=-1)
            assert torch.allclose(n_pred, torch.ones_like(n_pred))
            assert torch.allclose(n_gt, torch.ones_like(n_gt))
            fig, axs = plotOCSolution(xs=state.cpu().detach().numpy(), us=ctrl.cpu().detach().numpy(),
                                      xs_des=state_pred.cpu().detach().numpy(), us_des=ctrl_pred.cpu().detach().numpy(),
                                      robot_model=self.robot, show=False, dt=self.dt, plot_area=True, legend=False,
                                      fig=fig, color=cmap(i), markersize=0)
        artists_c = None
        for i, (state_obs, state_obs_pred) in enumerate(zip(pred['z'], pred['z_pred'])):
            state_obs_c = view_as_complex(state_obs)
            state_obs_c_pred = view_as_complex(state_obs_pred)
            ncols = int(np.sqrt(state_obs.shape[-1] // 2))
            fig_z, axs_z, artists_c = plotNdimTraj(traj=state_obs_c.cpu().detach().numpy(),
                                                   traj_des=state_obs_c_pred.cpu().detach().numpy(),  eigvals=eigvals,
                                                   var_name="ø(x)", show=False, dt=self.dt,
                                                   fig=fig_z, legend=False, color=cmap(i),
                                                   plot_grad_field=i==(len(pred['z']) - 1),
                                                   markersize=0, artists=artists_c, ncols=ncols)

        if log_fig:
            tb_logger = None
            for logger in self.trainer.loggers:
                if isinstance(logger, pl.loggers.TensorBoardLogger):
                    tb_logger = logger.experiment
                    break
            if isinstance(tb_logger, SummaryWriter):
                fig.canvas.draw()
                img_data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                img_data = img_data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                tb_logger.add_image(f"{log_prefix}pred", img_data, self.trainer.current_epoch, dataformats="HWC")
                fig_z.canvas.draw()
                img_obs_data = np.frombuffer(fig_z.canvas.tostring_rgb(), dtype=np.uint8)
                img_obs_data = img_obs_data.reshape(fig_z.canvas.get_width_height()[::-1] + (3,))
                tb_logger.add_image(f"{log_prefix}obs", img_obs_data, self.trainer.current_epoch, dataformats="HWC")
                tb_logger.flush()
                log.debug(f"Logged prediction of {num_trajs_plot} trajectories")
            else:
                warnings.warn("No Tensorboard logger")
        if show:
            fig.show()
            fig_z.show()
        print("")

