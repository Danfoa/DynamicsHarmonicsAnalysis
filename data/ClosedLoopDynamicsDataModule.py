import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from pinocchio import RobotWrapper
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from utils.mysc import append_dictionaries
from .ClosedLoopDynamics import ClosedLoopDynDataset, STATES, CTRLS, FULL_TRAJ

import logging

log = logging.getLogger(__name__)


class ClosedLoopDynDataModule(pl.LightningDataModule):

    def __init__(self, data_path: Path, window_size, batch_size: int = 256, num_workers: int = 0,
                 device=None, robot: RobotWrapper = None, augment: bool = False):
        super().__init__()
        assert data_path.exists(), f"Data folder not found {data_path.absolute()}"
        self._data_path = data_path
        self.robot = robot
        self.augment = augment
        assert window_size == FULL_TRAJ or window_size >= 2, "At least we need to forecast a single dynamics step"
        self.window_size = window_size
        self.device = device
        self.batch_size = batch_size
        self.num_workers = num_workers

    def prepare_data(self):
        robot_data = self.get_robot_data()
        dynamic_regime = "unstable_fix_point"

        self.train_dataset = ClosedLoopDynDataset(path=robot_data[dynamic_regime]["train"], normalize=True,
                                                  robot=self.robot, device=self.device, augment=self.augment,
                                                  window_size=self.window_size)
        self.test_dataset = ClosedLoopDynDataset(path=robot_data[dynamic_regime]["test"], normalize=True,
                                                 robot=self.robot, device=self.device, augment=True,
                                                 window_size=self.window_size,
                                                 state_scaler=self.train_dataset.standard_scaler[STATES],
                                                 ctrl_scaler=self.train_dataset.standard_scaler[CTRLS])
        self.val_dataset = ClosedLoopDynDataset(path=robot_data[dynamic_regime]["val"], normalize=True,
                                                robot=self.robot, device=self.device, augment=True,
                                                window_size=self.window_size,
                                                state_scaler=self.train_dataset.standard_scaler[STATES],
                                                ctrl_scaler=self.train_dataset.standard_scaler[CTRLS])
        self.predict_dataset = ClosedLoopDynDataset(path=robot_data[dynamic_regime]["test"], normalize=True,
                                                    robot=self.robot, device=self.device, augment=True,
                                                    window_size=FULL_TRAJ,
                                                    state_scaler=self.train_dataset.standard_scaler[STATES],
                                                    ctrl_scaler=self.train_dataset.standard_scaler[CTRLS])

        self.dt = self.train_dataset.dt

    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        log.info(f"Stage chanhe to {stage}")

    def train_dataloader(self):
        return DataLoader(dataset=self.train_dataset, batch_size=self.batch_size, shuffle=True, sampler=None,
                          collate_fn=lambda x: self.train_dataset.collate_fn(x), num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(dataset=self.val_dataset, batch_size=self.batch_size, shuffle=False, sampler=None,
                          collate_fn=lambda x: self.val_dataset.collate_fn(x), num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(dataset=self.test_dataset, batch_size=self.batch_size, shuffle=False, sampler=None,
                          collate_fn=lambda x: self.test_dataset.collate_fn(x), num_workers=self.num_workers)

    def predict_dataloader(self):
        # IMPORTANT. Predict dataloader doesnt shuffle data
        return DataLoader(dataset=self.predict_dataset, batch_size=self.batch_size, shuffle=False, sampler=None,
                          collate_fn=lambda x: self.predict_dataset.collate_fn(x), num_workers=self.num_workers)


    def get_robot_data(self):
        data = {}
        train_list = list(self._data_path.rglob("*train.pickle"))

        assert len(train_list) > 0, f"No trajectory data found in {self._data_path.absolute()}"

        for file in train_list:
            dynamic_mode = file.parent.parent.stem
            data = {dynamic_mode: {
                "train": file,
                "test": file.with_stem("test"),
                "val": file.with_stem("val")
            }
            }
        return data

    def losses_and_metrics(self, outputs_pred, outputs):
        """
        :param x: Batched sequence of consecutive states [x0, x1,..., xT], T=num_steps
        :param z: Batches embeddings/observations of `z = ø(x)` [z0, z1,..., zT]
        :param x_pred: Batches prediction of sequence of consecutive states `xt_pred = ø^-1(K^t•ø(x0))`:
                       [ø^-1(ø(x0)), ø^-1(Kø(x0)), ... ø^-1(K^T•ø(x0))]
        :param z_pred: Batches embeddings/observations of `x_pred` [z0, Kz0,...,K^T•z0]
        :return:
        """
        z_pred = outputs_pred['z_pred']
        x_pred = outputs_pred['x_pred']
        z = outputs_pred['z']
        x = outputs

        metrics = {}

        x_err = torch.abs(torch.sub(x, x_pred))  # ∀ t: |x_t - ø^-1(K^t•ø(x_0))|
        z_err = torch.abs(torch.sub(z, z_pred))  # ∀ t: |z_t - K^t•ø(x_0)|

        norm_x_err = torch.norm(x_err, dim=-1, p=2)
        norm_z_err = torch.norm(z_err, dim=-1, p=2)

        # Reconstruction loss of the system state x_0.
        reconstruction_loss = torch.mean(norm_x_err[:, 0])

        # Prediction loss. From the system state compute prediction accuracy from multi step predictions.
        avg_state_pred_err = torch.mean(norm_x_err[:, 1:], dim=1)

        if self.robot is not None:
            nq, nv = self.robot.nq, self.robot.nv
            # Get un-standarized error measurements
            q_err = x_err[:, :, :nq] / self.train_dataset._state_scale[:nq]
            dq_err = x_err[:, :, nq: nq + nv] / self.train_dataset._state_scale[nq: nq + nv]
            u_err = x_err[:, :, nq + nv:] / self.train_dataset._state_scale[nq + nv:]
            metrics.update(q_err=q_err.mean(), dq_err=dq_err.mean(), u_err=u_err.mean())

        # Linear dynamics of the observable/embedding.
        avg_obs_pred_err = torch.mean(norm_z_err[:, 1:], dim=1)

        # L-inf norm of reconstruction and single step prediction.
        x0, x1 = x[:, 0], x[:, 1]
        x0_rec, x1_pred = x_pred[:, 0], x_pred[:, 1]
        # Linf = ||x_0 - ø^-1(ø(x_0))||_inf + ||x_1 - ø^-1(K•ø(x_0))||_inf
        linf_loss = torch.norm(torch.sub(x0, x0_rec), p=float('inf'), dim=1) + \
                    torch.norm(torch.sub(x1, x1_pred), p=float('inf'), dim=1)

        metrics.update(pred_loss=avg_state_pred_err.mean(), lin_loss=avg_obs_pred_err.mean(), linf_loss=linf_loss.mean())

        loss = 0.01 * (reconstruction_loss + metrics["pred_loss"]) + metrics["lin_loss"]
        return loss, metrics

    def state_ctrl_to_x(self, batch):
        """
        Mapping from batch of ClosedLoopDynamics data points to NN model input-output data points
        """
        inputs = torch.concatenate([batch[STATES], batch[CTRLS]], dim=2)
        # outputs = torch.concatenate((batch["x_next"], batch["u_next"]), dim=2)
        return inputs

    def x_to_state_crtl(self, batch):
        x = batch.pop('x_pred')
        state = x[:, :, :self.robot.nq + self.robot.nv]
        ctrl = x[:, :, self.robot.nq + self.robot.nv:]
        batch.update({STATES: state, CTRLS: ctrl})
        return batch

    def plot_test_performance(self, pl_model: pl.LightningModule, log_fig=False, show=False):
        assert self.trainer is not None
        from utils.plotting import plotOCSolution
        from torch.utils.tensorboard import SummaryWriter
        import numpy as np

        num_trajs = len(self.predict_dataset)
        num_trajs_plot = 4

        idx = np.arange(num_trajs)
        shuffled_idx = np.random.permutation(idx)
        selected_trajs_idx = shuffled_idx[:num_trajs_plot]

        # Get ground truth
        gt = self.predict_dataset.collate_fn([self.predict_dataset[i] for i in selected_trajs_idx])
        # Get model predictions
        output = pl_model.predict_step(batch=gt, batch_idx=-1)
        pred = self.x_to_state_crtl(output)
        print("")

        fig, axs = None, None
        cmap = plt.cm.get_cmap('inferno', num_trajs_plot + 1)
        for i, (state, ctrl, state_pred, ctrl_pred) in enumerate(zip(gt[STATES], gt[CTRLS], pred[STATES], pred[CTRLS])):
            fig, axs = plotOCSolution(xs=state, us=ctrl,
                                      xs_des=state_pred.detach().numpy(), us_des=ctrl_pred.detach().numpy(),
                                      robot=self.robot, show=False, dt=self.dt,
                                      fig=fig, legend=fig is None, color=cmap(i), markersize=0)
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
                tb_logger.add_image(f"pred/pred", img_data, self.trainer.current_epoch, dataformats="HWC")
                tb_logger.flush()
                log.info(f"Logged prediction of {num_trajs_plot} trajectories")
            else:
                warnings.warn("No Tensorboard logger")
        if show:
            fig.show()
        print("")
