import logging
import pathlib
import time
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd
import plotly.graph_objs
import torch
import wandb
from lightning import LightningModule
from lightning.pytorch.utilities.types import STEP_OUTPUT

from nn.EquivLinearDynamics import EquivLinearDynamics
from nn.latent_markov_dynamics import LatentMarkovDynamics
from nn.LinearDynamics import LinearDynamics
from nn.markov_dynamics import MarkovDynamics

log = logging.getLogger(__name__)

from utils.mysc import flatten_dict, traj_from_states


class LightLatentMarkovDynamics(LightningModule):
    def __init__(
        self,
        lr: float,
        batch_size: int,
        test_epoch_metrics_fn=None,
        val_epoch_metrics_fn=None,
        log_figs_every_n_epochs=10,
        log_w=False,
        run_hps: Optional[dict] = None,
    ):
        super().__init__()
        # self.model_type = model.__class__.__name__
        self.model: LatentMarkovDynamics = None
        self.lr = lr
        self._batch_size = batch_size

        # self.model = model
        self._batch_unpack_fn = None
        self._loss_metrics_fn = None
        self.test_metrics_fn = test_epoch_metrics_fn
        self.val_metrics_fn = val_epoch_metrics_fn
        self._log_w = log_w
        self.log_figs_every_n_epochs = log_figs_every_n_epochs
        self._run_hps = dict(run_hps)
        # Save hyperparams in model checkpoint.
        self.save_hyperparameters()
        self._log_cache = {}

    def set_model(self, model: MarkovDynamics):
        self.model = model
        if hasattr(model, "eval_metrics"):
            self.test_metrics_fn = model.eval_metrics
            self.val_metrics_fn = model.eval_metrics

    def forward(self, batch):
        inputs = self._batch_unpack_fn(batch)
        return self.model(inputs)

    def training_step(self, batch, batch_idx):
        outputs = self.model(**batch)
        loss, metrics = self.model.compute_loss_and_metrics(**outputs, **batch)
        vector_metrics, scalar_metrics = self.separate_vector_scalar_metrics(metrics)

        self.log("loss/train", loss, prog_bar=False)
        self.log_metrics(scalar_metrics, suffix="train", batch_size=self._batch_size)
        self.log_vector_metrics(vector_metrics, type_sufix="train", batch_size=self._batch_size)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self.model(**batch)
        loss, metrics = self.model.compute_loss_and_metrics(**outputs, **batch)
        vector_metrics, scalar_metrics = self.separate_vector_scalar_metrics(metrics)

        self.log("loss/val", loss, prog_bar=False)
        self.log_metrics(scalar_metrics, suffix="val", batch_size=self._batch_size)
        self.log_vector_metrics(vector_metrics, type_sufix="val", batch_size=self._batch_size)
        return {"output": outputs, "input": batch}

    def test_step(self, batch, batch_idx):
        outputs = self.model(**batch)
        loss, metrics = self.model.compute_loss_and_metrics(**outputs, **batch)
        vector_metrics, scalar_metrics = self.separate_vector_scalar_metrics(metrics)

        # Compute the error in observation in original units
        state_traj = outputs["pred_state_traj"]
        gt_state_traj = traj_from_states(batch["state"], batch["next_state"])

        state_var = torch.from_numpy(self.trainer.datamodule.state_var).to(device=state_traj.device)
        state_traj_scaled = state_traj * state_var
        gt_state_traj_scaled = gt_state_traj * state_var

        err = state_traj_scaled - gt_state_traj_scaled
        mse = torch.mean(err**2, dim=(0, 1))

        q_ang = torch.reshape(state_traj[:, :, :24], (-1, 12, 2))
        q_ang = torch.arctan2(q_ang[:, :, 1], q_ang[:, :, 0])
        q_ang_gt = torch.reshape(gt_state_traj[:, :, :24], (-1, 12, 2))
        q_ang_gt = torch.arctan2(q_ang_gt[:, :, 1], q_ang_gt[:, :, 0])

        q_err = torch.mean((q_ang - q_ang_gt) ** 2, dim=0)
        dq_err = mse[24:36]
        z_err = mse[36]
        base_vel_err = mse[37:40]
        base_ori_err = mse[40:43]
        base_ang_vel_err = mse[43:46]

        scalar_metrics.update(
            state_q_err=torch.mean(q_err),
            state_dq_err=torch.mean(dq_err),
            state_z_err=torch.mean(z_err),
            state_base_vel_err=torch.mean(base_vel_err),
            state_base_ori_err=torch.mean(base_ori_err),
            state_base_ang_vel_err=torch.mean(base_ang_vel_err),
        )

        self.log("loss/test", loss, prog_bar=False)
        self.log_metrics(scalar_metrics, suffix="test", batch_size=self._batch_size)
        self.log_vector_metrics(vector_metrics, type_sufix="test", batch_size=self._batch_size)
        return {"output": outputs, "input": batch}

    def predict_step(self, batch, batch_idx, **kwargs):
        return self(batch)

    def on_train_epoch_start(self) -> None:
        self._epoch_start_time = time.time()

    def on_train_epoch_end(self) -> None:
        self.log("time_per_epoch", time.time() - self._epoch_start_time, prog_bar=False, on_epoch=True)

    def on_train_batch_end(self, outputs: STEP_OUTPUT, batch: Any, batch_idx: int) -> None:
        # Distributions have to be logged manually. Why keep using Lightning ? :(
        if self.trainer.global_step % self.trainer.log_every_n_steps == 0:
            self.log_vector_metrics(flush=True)

    def on_fit_start(self) -> None:
        # Ensure datamodule has the function for preprocessing batches and function for computing losses and metrics
        batch_unpack_fn = getattr(self.model, "batch_unpack", None)
        try:
            loss_metrics_fn = getattr(self.model, "compute_loss_and_metrics")
            assert callable(loss_metrics_fn)
        except:
            raise RuntimeError(
                f"Model {self.model.__class__.__name__} is expected to implement the function "
                f"`compute_loss_and_metrics`, returning (loss:Tensor, metrics:dict)"
            )

        self._batch_unpack_fn = batch_unpack_fn if callable(batch_unpack_fn) else lambda x: x
        self._loss_metrics_fn = loss_metrics_fn

    def on_train_start(self):
        # self.log("noise_level", self.trainer.datamodule.noise_level, prog_bar=False, on_epoch=True)

        if hasattr(self.model, "approximate_transfer_operator"):
            metrics = self.model.approximate_transfer_operator(self.trainer.datamodule.predict_dataloader())
            vector_metrics, scalar_metrics = self.separate_vector_scalar_metrics(metrics)
            self.log_metrics(scalar_metrics, suffix="")
            self.log_vector_metrics(vector_metrics, type_sufix="")

        # if self.val_metrics_fn is not None:
        # self.compute_figure_metrics(self.val_metrics_fn, self.trainer.datamodule.train_dataloader(), suffix="train")
        # self.compute_figure_metrics(self.val_metrics_fn, self.trainer.datamodule.val_dataloader(), suffix="val")
        # self.compute_figure_metrics(self.val_metrics_fn, self.trainer.datamodule.test_dataloader(), suffix="test")

    def on_train_end(self) -> None:
        ckpt_call = self.trainer.checkpoint_callback
        self.log_vector_metrics(flush=True)
        self.logger.save()

        if ckpt_call is not None:
            ckpt_path = pathlib.Path(ckpt_call.dirpath).joinpath(
                ckpt_call.CHECKPOINT_NAME_LAST + ckpt_call.FILE_EXTENSION
            )
            best_path = pathlib.Path(ckpt_call.dirpath).joinpath(ckpt_call.filename + ckpt_call.FILE_EXTENSION)
            if ckpt_path.exists() and best_path.exists():
                # Remove last model ckpt leave only best, to hint training successful termination.
                ckpt_path.unlink()
                log.info(f"Removing last ckpt {ckpt_path} from successful training run.")

        # Save train plots.
        # if self.val_metrics_fn is not None:
        #     self.compute_figure_metrics(self.val_metrics_fn, self.trainer.datamodule.train_dataloader(), suffix="train")
        #     self.compute_figure_metrics(self.val_metrics_fn, self.trainer.datamodule.val_dataloader(), suffix="val")

    def on_validation_start(self) -> None:
        if hasattr(self.model, "approximate_transfer_operator") and self.trainer.current_epoch % 2 == 0:
            metrics = self.model.approximate_transfer_operator(self.trainer.datamodule.predict_dataloader())
            vector_metrics, scalar_metrics = self.separate_vector_scalar_metrics(metrics)
            self.log_metrics(scalar_metrics, suffix="")
            self.log_vector_metrics(vector_metrics, type_sufix="")

    def on_validation_end(self) -> None:
        self.log_vector_metrics(flush=True)

        # if self.val_metrics_fn is not None and self.trainer.current_epoch % self.log_figs_every_n_epochs == 0:
        #     self.compute_figure_metrics(self.val_metrics_fn, self.trainer.datamodule.val_dataloader(), suffix="val")

    def on_test_start(self) -> None:
        if hasattr(self.model, "approximate_transfer_operator"):
            metrics = self.model.approximate_transfer_operator(self.trainer.datamodule.predict_dataloader())
            vector_metrics, scalar_metrics = self.separate_vector_scalar_metrics(metrics)
            self.log_metrics(scalar_metrics, suffix="")
            self.log_vector_metrics(vector_metrics, type_sufix="")

    def on_test_end(self) -> None:
        self.log_vector_metrics(flush=True)
        if self.test_metrics_fn is not None:
            self.compute_figure_metrics(self.test_metrics_fn, self.trainer.datamodule.test_dataloader(), suffix="test")

        # get the wandb logger
        wandb_logger = self.logger.experiment
        # Get the learned transfer operator.
        linear_dynamics = self.model.obs_space_dynamics
        if isinstance(linear_dynamics, EquivLinearDynamics):
            df_eigvals = None
            model_state_type = linear_dynamics.state_type
            koop_operator, _ = linear_dynamics.transfer_op.expand_parameters()
            iso_koop_operators = {}
            for field_start, field_end, field_rep in zip(
                model_state_type.fields_start, model_state_type.fields_end, model_state_type.representations
            ):
                iso_irrep_id = field_rep.irreps[0]
                iso_koop_operators[iso_irrep_id] = koop_operator[field_start:field_end, field_start:field_end]
            # Get the eigenvalues of each iso-koop operator.
            for iso_irrep_id, koop_op in iso_koop_operators.items():
                eig_vals = torch.linalg.eigvals(koop_op).detach().cpu().numpy()
                # Stack the eigvals in the data frame vertically.
                iso_df_eigvals = pd.DataFrame({"real": eig_vals.real, "imag": eig_vals.imag})
                # Set a column with the irrep id for all eigvals
                iso_df_eigvals = iso_df_eigvals.assign(irrep=str(iso_irrep_id))
                if df_eigvals is None:
                    df_eigvals = iso_df_eigvals
                else:
                    df_eigvals = pd.concat([df_eigvals, iso_df_eigvals], axis=0, ignore_index=True)

            wandb_eigvals_table = wandb.Table(dataframe=df_eigvals)

        elif isinstance(linear_dynamics, LinearDynamics):
            koop_operator = linear_dynamics.transfer_op.weight
            eig_vals = torch.linalg.eigvals(koop_operator).detach().cpu().numpy()
            # Get the eigenvalues of each kooman operator.
            df_eigvals = pd.DataFrame({"real": eig_vals.real, "imag": eig_vals.imag})
            df_eigvals = df_eigvals.assign(irrep=None)
            wandb_eigvals_table = wandb.Table(dataframe=df_eigvals)

        wandb_logger.log({"trans_op_eigvals": wandb_eigvals_table, "trainer/global_step": self.trainer.global_step})

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def log_metrics(self, metrics: dict, suffix="", batch_size=None):
        flat_metrics = flatten_dict(metrics)
        for k, v in flat_metrics.items():
            name = f"{k}/{suffix}"
            self.log(name, v, prog_bar=False, batch_size=batch_size)

    @torch.no_grad()
    def log_vector_metrics(self, metrics: Optional[dict] = None, type_sufix="", batch_size=None, flush=False):
        if metrics is None and flush is False:
            return

        flat_metrics = flatten_dict(metrics) if metrics is not None else {}
        # get the wandb logger
        wandb_logger = self.logger.experiment

        for metric, vector in flat_metrics.items():
            assert vector.ndim >= 1, f"Vector metric {metric} has to be of shape (n_samples,) or (batch, time_steps)."
            metric_log_name = f"{metric}/{type_sufix}"
            if "_t/" in metric_log_name or "_dist/" in metric_log_name:
                # Separate the last _sufix part from the key to obtain the metric name.
                tmp = metric.split("_")  # Average value will use this name, vector metric will use the full name.
                metric_name, metric_sufix = "_".join(tmp[:-1]), tmp[-1]
                self.log(f"{metric_name}/{type_sufix}", torch.mean(vector), prog_bar=False, batch_size=batch_size)
            else:
                self.log(metric_log_name, torch.mean(vector), prog_bar=False, batch_size=batch_size)

            if type_sufix == "train" or type_sufix == "val":
                continue

            if metric_log_name in self._log_cache:
                self._log_cache[metric_log_name] = np.concatenate(
                    [self._log_cache[metric_log_name], vector.detach().cpu().numpy()], axis=0
                )
            else:
                self._log_cache[metric_log_name] = vector.detach().cpu().numpy()

        log_vector_metrics_n_steps = 10
        if self.trainer.current_epoch % log_vector_metrics_n_steps == 0 or flush:
            metrics_to_wipe = []
            for metric_log_name, metric_vector in self._log_cache.items():
                if "_t/" in metric_log_name:  # Vector to be plotted against time
                    assert metric_vector.ndim == 2, (
                        f"{metric_log_name} Expected (batch, time_steps) vector, got {metric_vector.shape}."
                    )
                    dt = self.model.dt if hasattr(self.model, "dt") else 1
                    time_horizon = np.arange(metric_vector.shape[-1]) * dt
                    y = np.mean(metric_vector, axis=0)

                    df = pd.DataFrame(columns=["time", f"{metric_log_name}", "epoch"])
                    df["time"] = time_horizon
                    df[f"{metric_log_name}"] = y
                    df["epoch"] = self.trainer.global_step

                    for t, x in zip(df["time"], df[f"{metric_log_name}"]):
                        wandb_logger.log(
                            {f"{metric_log_name}": x, "time": t, "trainer/global_step": self.trainer.global_step}
                        )

                    metrics_to_wipe.append(metric_log_name)

                elif "_dist/" in metric_log_name:  # Distribution to be logged as a histogram.
                    wandb_logger.log(
                        {
                            f"{metric_log_name}": wandb.Histogram(metric_vector),
                            "trainer/global_step": self.trainer.global_step,
                        }
                    )
                    metrics_to_wipe.append(metric_log_name)

            self._log_cache = {}

    def log_figures(self, figs: dict[str, plotly.graph_objs.Figure], suffix=""):
        """Log plotly figures to wandb."""
        wandb_logger = self.logger.experiment
        for fig_name, fig in figs.items():
            wandb_logger.log({f"{fig_name}/{suffix}": fig, "trainer/global_step": self.trainer.global_step})

    @torch.no_grad()
    def compute_figure_metrics(self, metrics_fn: Callable, dataloader, suffix=""):
        batch = next(iter(dataloader))
        outputs = self.model(**batch)

        figs, metrics = metrics_fn(**outputs, **batch)

        if metrics is not None:
            self.log_metrics(metrics, suffix=suffix, batch_size=self._batch_size)
        if figs is not None:
            self.log_figures(figs, suffix=suffix)

    # def on_s
    def get_metrics(self):
        # don't show the version number on console logs.
        items = super().get_metrics()
        items.pop("v_num", None)
        return items

    def separate_vector_scalar_metrics(self, metrics: dict):
        vector_metrics = []
        scalar_metrics = []

        for k, v in metrics.items():
            if len(v.shape) >= 1:  # Metric to be plotted against time or in a histogram.
                vector_metrics.append(k)
            else:
                scalar_metrics.append(k)

        vector_metrics = {k: v for k, v in metrics.items() if k in vector_metrics}
        scalar_metrics = {k: v for k, v in metrics.items() if k in scalar_metrics}

        return vector_metrics, scalar_metrics
