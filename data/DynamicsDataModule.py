import warnings
from pathlib import Path
from typing import Optional, Union

import escnn.group
import matplotlib.pyplot as plt
import numpy as np
import pinocchio
import torch
from escnn.group import Representation, directsum
from escnn.nn import FieldType
from lightning.pytorch.loggers import WandbLogger
from pinocchio import RobotWrapper
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from data.DynamicsRecording import get_dynamics_dataset, get_train_test_val_file_paths, map_state_next_state
from utils.plotting import plot_observations, plot_state_actions

import logging

log = logging.getLogger(__name__)


class DynamicsDataModule(pl.LightningDataModule):

    def __init__(self,
                 data_path: Path,
                 pred_horizon: Union[int, float] = 0.5,
                 batch_size: int = 256,
                 frames_per_step: int = 1,
                 num_workers: int = 0,
                 augment: bool = False,
                 state_measurements: Optional[list[str]] = None,
                 action_measurements: Optional[list[str]] = None,
                 ):
        super().__init__()
        assert data_path.exists(), f"Data folder not found {data_path.absolute()}"
        self._data_path = data_path
        self.augment = augment
        self.frames_per_step = frames_per_step
        if isinstance(pred_horizon, float):
            assert 0 < pred_horizon <= 1.0, "Prediction horizon need be in H∈(0., 1.]. Predict H % of the trajectory"
            self.pred_horizon = pred_horizon
        elif isinstance(pred_horizon, int):
            assert pred_horizon >= 1, "At least we need to forecast a single dynamics step"
            self.pred_horizon = pred_horizon + 1
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.state_measurements = state_measurements
        self.action_measurements = action_measurements
        self._val_dataloader, self._test_dataloader, self._train_dataloader = None, None, None
        # Symmetry parameters
        self.symm_group = None
        self.measurements_reps = {}
        self.gspace = None
        self.state_field_type, self.action_field_type = None, None

    def prepare_data(self):
        if self.prepared:
            return

        path_to_dyn_sys_data = set([a.parent for a in list(self._data_path.rglob('*train.pkl'))])
        # TODO: Handle multiple files from
        system_data_path = path_to_dyn_sys_data.pop()
        train_data, test_data, val_data = get_train_test_val_file_paths(system_data_path)
        # Obtain hugging face Iterable datasets instances
        datasets, metadata = get_dynamics_dataset(train_shards=train_data,
                                                  test_shards=test_data,
                                                  val_shards=val_data,
                                                  frames_per_state=self.frames_per_step,
                                                  prediction_horizon=self.pred_horizon,
                                                  state_measurements=self.state_measurements,
                                                  action_measurements=self.action_measurements)
        self.dt = metadata.dynamics_parameters['dt']
        # In case no measurements are passed, we recover the ones from the DynamicsRecording
        self.state_measurements = metadata.state_measurements
        self.action_measurements = metadata.action_measurements

        # Ensure samples contain torch.Tensors and not numpy arrays.
        # Apply map to obtain flat state/next_state action/next_action values
        train_dataset, test_dataset, val_dataset = datasets
        train_dataset = train_dataset.map(
            map_state_next_state, batched=True, fn_kwargs={'state_measurements': self.state_measurements}).shuffle()
        test_dataset = test_dataset.with_format("torch").map(
            map_state_next_state, batched=True, fn_kwargs={'state_measurements': self.state_measurements})
        val_dataset = val_dataset.with_format("torch").map(
            map_state_next_state, batched=True, fn_kwargs={'state_measurements': self.state_measurements})

        # Rebuilt the ESCNN representations of measurements _________________________________________________________
        # TODO: Handle dyn systems without symmetries
        G_domain = escnn.group.O3()
        G_id = metadata.dynamics_parameters['group']['subgroup_id']  # Identifier of symmetry group
        self.symm_group, _, _ = G_domain.subgroup(G_id)
        for measurement, irreps_ids in metadata.reps_irreps.items():
            Q = metadata.reps_change_of_basis[measurement]
            rep = Representation(self.symm_group, name=measurement,
                                 irreps=list(irreps_ids), change_of_basis=np.asarray(Q))
            self.measurements_reps[measurement] = rep

        # Construct state (and action) representations considering the `frames_per_step` and the concatenation
        # convention of the function `map_state_next_state` in `DynamicsRecording.py`. Which defines the state
        # with `F=frames_per_step` delayed coordinates as s_t = [m1_f,..., m1_f+F, m2_f,..., m2_f+F, ...] where mi_k
        # is the measurement i at frame k.
        state_reps = [[self.measurements_reps[m]] * self.frames_per_step for m in self.state_measurements]
        state_reps = [rep for frame_reps in state_reps for rep in frame_reps]  # flatten list of reps

        # Use as default no basis space # TODO make more flexible
        self.gspace = escnn.gspaces.no_base_space(self.symm_group)
        # Define the state field type
        self.state_field_type = FieldType(self.gspace, representations=state_reps)
        # Define the action field type
        self.action_field_type = None
        if len(self.action_measurements) > 0:
            # construct action representations analog to the state representations
            action_reps = [[self.measurements_reps[m]] * self.frames_per_step for m in self.action_measurements]
            action_reps = [rep for frame_reps in action_reps for rep in frame_reps]  # flatten list of reps
            self.action_field_type = FieldType(self.gspace, representations=action_reps)

        if self.augment:  # Apply data augmentation to the state and state trajectories
            train_dataset = train_dataset.map(self.augment_data_map, batched=True, num_proc=self.num_workers)

        self._train_dataloader = DataLoader(dataset=train_dataset, batch_size=self.batch_size, shuffle=False,
                                            num_workers=self.num_workers)
        self._test_dataloader = DataLoader(dataset=test_dataset, batch_size=self.batch_size, shuffle=False,
                                           num_workers=self.num_workers)
        self._val_dataloader = DataLoader(dataset=val_dataset, batch_size=self.batch_size, shuffle=False,
                                          num_workers=self.num_workers)

    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        log.debug(f"Stage change to {stage}")

    def train_dataloader(self):
        return self._train_dataloader

    def val_dataloader(self):
        return self._val_dataloader

    def test_dataloader(self):
        return self._test_dataloader

    def predict_dataloader(self):
        return self._test_dataloader

    @property
    def prepared(self):
        return self._train_dataloader is not None

    def augment_data_map(self, sample: dict) -> dict:
        state = sample['state']
        next_state = sample['next_state']
        action = sample.get(['action'], None)
        next_action = sample.get(['next_action'], None)

        # Sample a random symmetry transformation
        g = self.symm_group.sample()
        if g == self.symm_group.identity:  # Avoid the computational overhead of applying the identity
            return sample

        g_state = self.state_field_type.transform_fibers(input=state, group_element=g)
        g_next_state = self.state_field_type.transform_fibers(input=next_state, group_element=g)
        sample['state'] = g_state
        sample['next_state'] = g_next_state
        if action is not None:
            g_action = self.action_field_type.transform_fibers(input=action, group_element=g)
            g_next_action = self.action_field_type.transform_fibers(input=next_action, group_element=g)
            sample['action'] = g_action
            sample['next_action'] = g_next_action
        return sample

    # def plot_test_performance(self, pl_model: pl.LightningModule, dataset: ClosedLoopDynDataset,
    #                           log_fig=False, show=False, eigvals=None, log_prefix=''):
    #     assert self.trainer is not None
    #     from utils.plotting import plotOCSolution, plotNdimTraj
    #     from torch.utils.tensorboard import SummaryWriter
    #     import numpy as np
    #
    #     num_trajs = len(dataset)
    #     num_trajs_plot = 2
    #
    #     idx = np.arange(num_trajs)
    #     gen = np.random.default_rng(12345)
    #     shuffled_idx = gen.permutation(idx)
    #     selected_trajs_idx = shuffled_idx[:num_trajs_plot]
    #
    #     # Get ground truth
    #     gt = dataset.collate_fn([dataset[i] for i in selected_trajs_idx])
    #     # Get model forecasts
    #     x = self.trainer.model.model.batch_unpack(gt)
    #     output = pl_model.model.forecast(x)
    #     pred = self.trainer.model.model.batch_pack(output)
    #
    #     cmap = plt.cm.get_cmap('inferno', num_trajs_plot + 1)
    #     if STATES in pred and CTRLS in pred:
    #         # Unstandarized ground truth data for plotting. TODO: Make a more elegant solution here
    #         raise NotImplementedError("Check this is working back again")
    #         # gt[STATES] = (gt[STATES] * self.train_dataset._state_scale) + self.train_dataset._state_mean
    #         # gt[CTRLS] = (gt[CTRLS] * self.train_dataset._ctrl_scale) + self.train_dataset._ctrl_mean
    #         #
    #         # fig = plot_state_actions(states=gt[STATES], ctrls=gt[CTRLS], states_pred=pred[STATES], ctrls_pred=pred[
    #         # CTRLS],
    #         #                          robot=self.robot, dt=self.dt, cmap=cmap)
    #     if STATES_OBS in pred:
    #         eigvals = None
    #         fig_z, artists_c = plot_observations(pred[STATES_OBS], pred[f'{STATES_OBS}_pred'], eigvals=eigvals,
    #                                              dt=self.dt, cmap=cmap)
    #         fig_z_obs = None
    #         if 'z_obs' in pred:
    #             fig_z_obs, artists_c_obs = plot_observations(pred['z_obs'], pred[f'z_pred_obs'], dt=self.dt,
    #             cmap=cmap)
    #
    #     if log_fig:
    #         wandb_logger = None
    #         for logger in self.trainer.loggers:
    #             if isinstance(logger, WandbLogger):
    #                 logger.log_image(key=f"{log_prefix}eigf", images=[fig_z], step=self.trainer.current_epoch)
    #
    #                 # logger.log({f"{log_prefix}eigfd": fig_z})
    #
    #             # fig.canvas.draw()
    #             # img_data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    #             # img_data = img_data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    #             # tb_logger.add_image(f"{log_prefix}pred", img_data, self.trainer.current_epoch, dataformats="HWC")
    #             # fig_z.canvas.draw()
    #             # imgs = []
    #             # img_obs_data = np.frombuffer(fig_z.canvas.tostring_rgb(), dtype=np.uint8)
    #             # img_obs_data = img_obs_data.reshape(fig_z.canvas.get_width_height()[::-1] + (3,))
    #             # imgs.append(img_obs_data)
    #             # wandb_logger.log_image(f"{log_prefix}eigf", img_obs_data, )
    #             # if fig_z_obs is not None:
    #             #     fig_z_obs.canvas.draw()
    #             #     img_obs_data = np.frombuffer(fig_z_obs.canvas.tostring_rgb(), dtype=np.uint8)
    #             #     img_obs_data = img_obs_data.reshape(fig_z_obs.canvas.get_width_height()[::-1] + (3,))
    #             #     wandb_logger.log_image(f"{log_prefix}obs", img_obs_data, self.trainer.current_epoch,
    #             #     dataformats="HWC")
    #             #     imgs.append(img_obs_data)
    #             # wandb_logger.log_image(f"{log_prefix}eigf", imgs, self.trainer.current_epoch)
    #             log.debug(f"Logged prediction of {num_trajs_plot} trajectories")
    #     if show:
    #         # fig.show()
    #         fig_z.show()
    #         if fig_z_obs is not None:
    #             fig_z_obs.show()
    #
    #     print("")


if __name__ == "__main__":
    path_to_data = Path(__file__).parent
    assert path_to_data.exists(), f"Invalid Dataset path {path_to_data.absolute()}"

    # Find all dynamic systems recordings
    path_to_data /= 'linear_systems'
    path_to_dyn_sys_data = set([a.parent for a in list(path_to_data.rglob('*train.pkl'))])
    # Select a dynamical system
    mock_path = path_to_dyn_sys_data.pop()

    data_module = DynamicsDataModule(data_path=mock_path,
                                     pred_horizon=2,
                                     frames_per_step=5,
                                     num_workers=1,
                                     batch_size=10,
                                     augment=False
                                     )

    # Test loading of the DynamicsRecording
    data_module.prepare_data()

    for i, batch in enumerate(data_module.train_dataloader()):
        if i > 5000:
            break
