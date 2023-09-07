import pathlib
import time
from typing import Any, Union, Callable, Optional

import numpy as np
import torch

import logging

import wandb
from lightning import LightningModule
from lightning.pytorch.utilities.types import STEP_OUTPUT

from nn.markov_dynamics import MarkovDynamicsModule

log = logging.getLogger(__name__)

from utils.mysc import flatten_dict


class LightningModel(LightningModule):

    def __init__(self,
                 lr: float,
                 batch_size: int,
                 test_epoch_metrics_fn=None,
                 val_epoch_metrics_fn=None,
                 log_preact=False,
                 log_w=False,
                 run_hps: Optional[dict] = None):
        super().__init__()
        # self.model_type = model.__class__.__name__
        self.model = None
        self.lr = lr
        self._batch_size = batch_size

        # self.model = model
        self._batch_unpack_fn = None
        self._loss_metrics_fn = None
        self.test_metrics_fn = test_epoch_metrics_fn
        self.val_metrics_fn = val_epoch_metrics_fn
        self._log_w = log_w
        self._log_preact = log_preact
        self._run_hps = dict(run_hps)
        # Save hyperparams in model checkpoint.
        self.save_hyperparameters()
        self._log_cache = {}

    def set_model(self, model: MarkovDynamicsModule):
        self.model = model

    def forward(self, batch):
        inputs = self._batch_unpack_fn(batch)
        return self.model(inputs)

    def training_step(self, batch, batch_idx):
        n_steps = batch['next_state'].shape[1]
        outputs = self.model(**batch, n_steps=n_steps)
        loss, metrics = self.model.compute_loss_and_metrics(**outputs, **batch)

        self.log("train/loss", loss, prog_bar=False)
        self.log_metrics(metrics, prefix="train/", batch_size=self._batch_size)
        return loss

    def validation_step(self, batch, batch_idx):
        n_steps = batch['next_state'].shape[1]
        outputs = self.model(**batch, n_steps=n_steps)
        loss, metrics = self.model.compute_loss_and_metrics(**outputs, **batch)

        if self.val_metrics_fn is not None:
            val_metrics = self.val_metrics_fn(outputs, batch)
            metrics.update(val_metrics)

        self.log("val/loss", loss, prog_bar=False)
        self.log_metrics(metrics, prefix="val/", batch_size=self._batch_size)
        return {'output': outputs, 'input': batch}

    def test_step(self, batch, batch_idx):
        n_steps = batch['next_state'].shape[1]
        outputs = self.model(**batch, n_steps=n_steps)

        loss, metrics = self.model.compute_loss_and_metrics(**outputs, **batch)

        if self.val_metrics_fn is not None:
            test_metrics = self.test_metrics_fn(outputs, batch)
            metrics.update(test_metrics)

        self.log("test/loss", loss, prog_bar=False)
        self.log_metrics(metrics, prefix="test/", batch_size=self._batch_size)
        return {'output': outputs, 'input': batch}

    def predict_step(self, batch, batch_idx, **kwargs):
        return self(batch)

    def on_train_epoch_start(self) -> None:
        self._epoch_start_time = time.time()

    def on_train_epoch_end(self) -> None:
        self.log('time_per_epoch', time.time() - self._epoch_start_time, prog_bar=False, on_epoch=True)

    def on_train_batch_end(self, outputs: STEP_OUTPUT, batch: Any, batch_idx: int) -> None:
        # Distributions have to be logged manually. Why keep using Lightning ? :(
        if self.trainer.global_step % self.trainer.log_every_n_steps == 0:
            self.log_distribution(flush=True)

    def on_fit_start(self) -> None:
        # Ensure datamodule has the function for preprocessing batches and function for computing losses and metrics
        batch_unpack_fn = getattr(self.model, 'batch_unpack', None)
        try:
            loss_metrics_fn = getattr(self.model, 'compute_loss_and_metrics')
            assert callable(loss_metrics_fn)
        except:
            raise RuntimeError(f"Model {self.model.__class__.__name__} is expected to implement the function "
                               f"`compute_loss_and_metrics`, returning (loss:Tensor, metrics:dict)")

        self._batch_unpack_fn = batch_unpack_fn if callable(batch_unpack_fn) else lambda x: x
        self._loss_metrics_fn = loss_metrics_fn

    def on_train_start(self):
        # TODO: Add number of layers and hidden channels dimensions.
        hparams = flatten_dict(self._run_hps)
        if hasattr(self.model, "get_hparams"):
            hparams.update(flatten_dict(self.model.get_hparams()))
        # Get the labels of metrics
        if self.logger:
            metrics = {"val/loss": 0, "test/loss": 0, "train/loss": 0}
            if callable(getattr(self.model, 'get_metric_labels', None)):
                metric_labels = self.model.get_metric_labels()
                for k in metric_labels:
                    metrics[f"test/{k}"] = 0
                    metrics[f"val/{k}"] = 0
            else:
                log.warning(f"Model does not implement `get_metric_labels` function. Only default metrics "
                            f"{list(metrics.keys())} will be logged to tensorboard hyperparam metrics")

    def on_train_end(self) -> None:
        ckpt_call = self.trainer.checkpoint_callback
        self.log_distribution(flush=True)
        self.logger.save()

        if ckpt_call is not None:
            ckpt_path = pathlib.Path(ckpt_call.dirpath).joinpath(
                ckpt_call.CHECKPOINT_NAME_LAST + ckpt_call.FILE_EXTENSION)
            best_path = pathlib.Path(ckpt_call.dirpath).joinpath(ckpt_call.filename + ckpt_call.FILE_EXTENSION)
            if ckpt_path.exists() and best_path.exists():
                # Remove last model ckpt leave only best, to hint training successful termination.
                ckpt_path.unlink()
                log.info(f"Removing last ckpt {ckpt_path} from successful training run.")

    def on_validation_end(self) -> None:
        self.log_distribution(flush=True)

    def on_test_start(self) -> None:
        if hasattr(self.model, "approximate_transfer_operator"):
            self.model.approximate_transfer_operator(self.trainer.datamodule.predict_dataloader())

    def on_validation_start(self) -> None:
        if hasattr(self.model, "approximate_transfer_operator") and self.global_step > 0:
            self.model.approximate_transfer_operator(self.trainer.datamodule.predict_dataloader())

    def on_test_end(self) -> None:
        self.log_distribution(flush=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def log_metrics(self, metrics: dict, prefix='', batch_size=None):
        flat_metrics_dic = flatten_dict(metrics)
        for k, v in flat_metrics_dic.items():
            name = f"{prefix}{k}"
            if v.ndim == 0:  # Single scalars.
                self.log(name, v, prog_bar=False, batch_size=batch_size)
            else:
                self.log(f"{name}", torch.mean(v), prog_bar=False)
                # self.log_distribution(name, v, flush=False)

    def log_distribution(self, name: Optional[str] = None, value: Optional[torch.Tensor] = None, flush=False):
        wandb_logger = self.logger.experiment
        if self.trainer.global_step > 0 and name is not None and value is not None:
            if name in self._log_cache:
                self._log_cache[name] = np.concatenate([self._log_cache[name], value.detach().cpu().numpy()])
            else:
                self._log_cache[name] = value.detach().cpu().numpy()

        if flush or self.trainer.global_step == self.trainer.log_every_n_steps:
            logged_dist = []
            for name, value in self._log_cache.items():
                if value.size > 30:
                    wandb_logger.log({name: wandb.Histogram(self._log_cache[name]),
                                      'trainer/global_step': self.trainer.global_step})
                    logged_dist.append(name)
            for logged_name in logged_dist:
                del self._log_cache[logged_name]

    # def on_s
    def get_metrics(self):
        # don't show the version number on console logs.
        items = super().get_metrics()
        items.pop("v_num", None)
        return items
