import math
import pathlib
import time
from typing import Union, Callable, Optional

import numpy as np
import pytorch_lightning as pl
import torch
from torch.nn import Module

import logging

from nn.markov_dynamics import MarkovDynamicsModule

log = logging.getLogger(__name__)

from utils.mysc import flatten_dict

class LightningModel(pl.LightningModule):

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

    def set_model(self, model: MarkovDynamicsModule):
        self.model = model

    def forward(self, batch):
        inputs = self._batch_unpack_fn(batch)
        return self.model(inputs)

    def training_step(self, batch, batch_idx):
        n_steps = batch['next_state'].shape[1]
        outputs = self.model(**batch, n_steps=n_steps)
        loss, metrics = self.model.loss_and_metrics(outputs, batch)

        self.log("train/loss", loss, prog_bar=False)
        self.log_metrics(metrics, prefix="train/", batch_size=self._batch_size)
        return loss

    def validation_step(self, batch, batch_idx):
        n_steps = batch['next_state'].shape[1]
        outputs = self.model(**batch, n_steps=n_steps)
        loss, metrics = self.model.loss_and_metrics(outputs, batch)
        
        if self.val_metrics_fn is not None:
            val_metrics = self.val_metrics_fn(outputs, batch)
            metrics.update(val_metrics)

        self.log("val/loss", loss, prog_bar=False)
        self.log_metrics(metrics, prefix="val/", batch_size=self._batch_size)
        return {'output': outputs, 'input': batch}

    def test_step(self, batch, batch_idx):
        n_steps = batch['next_state'].shape[1]
        outputs = self.model(**batch, n_steps=n_steps)
        loss, metrics = self.model.loss_and_metrics(outputs, batch)
        
        if self.val_metrics_fn is not None:
            test_metrics = self.test_metrics_fn(outputs, batch)
            metrics.update(test_metrics)

        self.log("test/loss", loss, prog_bar=False)
        self.log_metrics(metrics, prefix="test/", batch_size=self._batch_size)
        return {'output': outputs, 'input': batch}

    def predict_step(self, batch, batch_idx, **kwargs):
        return self(batch)

    def on_train_epoch_start(self) -> None:
        self.epoch_start_time = time.time()

    def training_epoch_end(self, outputs):
        self.log('time_per_epoch', time.time() - self.epoch_start_time, prog_bar=False, on_epoch=True)
        if self._log_w: self.log_weights()
        if self._log_preact: self.log_preactivations()

    def on_validation_start(self) -> None:
        pass
        # if isinstance(self.model, VAMP):
        #     # If there is a new function space we need to update the Koopman approximation
        #     if not self.model.updated_eigenmatrix:
        #         train_dataloader = self.trainer.datamodule.train_dataloader()
        #         batched_outputs = []
        #         for i, batch in enumerate(train_dataloader):
        #             batched_outputs.append(self.predict_step(batch, i))
        #         self.model.approximate_koopman_op(batched_outputs)
        #
        #         if self.val_metrics_fn is not None:
        #             # Compute the training error in prediction of observation dynamics
        #             for i, batch in enumerate(train_dataloader):
        #                 obs = self.predict_step(batch, i)
        #                 metrics = self.val_metrics_fn(obs, self._batch_unpack_fn(batch))
        #                 self.log_metrics(metrics, prefix="train/")

    def on_test_start(self) -> None:
        pass
        # if isinstance(self.model, VAMP):
        #     # If there is a new function space we need to update the Koopman approximation
        #     if not self.model.updated_eigenmatrix:
        #         train_dataloader = self.trainer.datamodule.train_dataloader()
        #         batched_outputs = []
        #         for i, batch in enumerate(train_dataloader):
        #             batched_outputs.append(self.predict_step(batch, i))
        #         self.model.approximate_koopman_op(batched_outputs)

    def on_fit_start(self) -> None:
        # Ensure datamodule has the function for preprocessing batches and function for computing losses and metrics
        batch_unpack_fn = getattr(self.model, 'batch_unpack', None)
        try:
            loss_metrics_fn = getattr(self.model, 'compute_loss_metrics')
            assert callable(loss_metrics_fn)
        except:
            raise RuntimeError(f"Model {self.model.__class__.__name__} is expected to implement the function "
                               f"`compute_loss_metrics`, returning a metric of (loss:Tensor, metrics:dict)")

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
            # self.logger.experiment.config.update(hparams)

    def on_train_end(self) -> None:
        ckpt_call = self.trainer.checkpoint_callback
        self.logger.save()

        if ckpt_call is not None:
            ckpt_path = pathlib.Path(ckpt_call.dirpath).joinpath(
                ckpt_call.CHECKPOINT_NAME_LAST + ckpt_call.FILE_EXTENSION)
            best_path = pathlib.Path(ckpt_call.dirpath).joinpath(ckpt_call.filename + ckpt_call.FILE_EXTENSION)
            if ckpt_path.exists() and best_path.exists():
                # Remove last model ckpt leave only best, to hint training successful termination.
                ckpt_path.unlink()
                log.info(f"Removing last ckpt {ckpt_path} from successful training run.")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def log_metrics(self, metrics: dict, prefix='', batch_size=None):
        for k, v in metrics.items():
            name = f"{prefix}{k}"
            self.log(name, v, prog_bar=False, batch_size=batch_size)

    def log_weights(self):
        raise NotImplementedError()
        if not self.logger: return
        tb_logger = self.logger.experiment
        layer_index = 0  # Count layers by linear operators not position in network sequence
        for layer in self.dp_net.net:
            layer_name = f"Layer{layer_index:02d}"
            if isinstance(layer, EquivariantBlock) or isinstance(layer, BasisLinear):
                lin = layer.linear if isinstance(layer, EquivariantBlock) else layer
                W = lin.weight.view(-1).detach()
                basis_coeff = lin.basis_coeff.view(-1).detach()
                tb_logger.add_histogram(tag=f"{layer_name}/c", values=basis_coeff, global_step=self.current_epoch)
                tb_logger.add_histogram(tag=f"{layer_name}/W", values=W, global_step=self.current_epoch)
                layer_index += 1
            elif isinstance(layer, LinearBlock) or isinstance(layer, torch.nn.Linear):
                lin = layer.linear if isinstance(layer, LinearBlock) else layer
                W = lin.weight.view(-1).detach()
                tb_logger.add_histogram(tag=f"{layer_name}/W", values=W, global_step=self.current_epoch)
                layer_index += 1

    def log_preactivations(self, ):
        raise NotImplementedError()
        if not self.logger: return
        tb_logger = self.logger.experiment
        layer_index = 0  # Count layers by linear operators not position in network sequence
        for layer in self.dp_net.net:
            layer_name = f"Layer{layer_index:02d}"
            if isinstance(layer, EquivariantBlock) or isinstance(layer, LinearBlock):
                tb_logger.add_histogram(tag=f"{layer_name}/pre-act", values=layer._preact,
                                        global_step=self.current_epoch)
                layer_index += 1

    def get_metrics(self):
        # don't show the version number on console logs.
        items = super().get_metrics()
        items.pop("v_num", None)
        return items
