import logging
import time
from pathlib import Path
from typing import Optional, Union

import escnn.group
import numpy as np
import torch
from datasets.distributed import split_dataset_by_node
from escnn.group import Representation
from escnn.nn import FieldType
from lightning import LightningDataModule
from torch.utils.data import DataLoader, default_collate
from tqdm import tqdm

from data.DynamicsRecording import DynamicsRecording, get_dynamics_dataset, get_train_test_val_file_paths
from utils.mysc import traj_from_states
from utils.plotting import plot_system_3D, plot_trajectories, plot_two_panel_trajectories

log = logging.getLogger(__name__)


class DynamicsDataModule(LightningDataModule):

    def __init__(self,
                 data_path: Path,
                 pred_horizon: Union[int, float] = 0.25,
                 eval_pred_horizon: Union[int, float] = 0.5,
                 system_cfg: Optional[dict] = None,
                 batch_size: int = 256,
                 frames_per_step: int = 1,
                 num_workers: int = 0,
                 augment: bool = False,
                 device='cpu',
                 state_obs: Optional[tuple[str]] = None,
                 action_obs: Optional[tuple[str]] = None,
                 ):
        super().__init__()
        if system_cfg is None:
            system_cfg = {}
        self._data_path = data_path
        self.system_cfg = system_cfg if system_cfg is not None else {}
        self.noise_level = self.system_cfg.get('noise_level', None)
        self.augment = augment
        self.frames_per_step = frames_per_step
        if isinstance(pred_horizon, float):
            assert 0 < pred_horizon <= 1.0, "Prediction horizon need be in H∈(0., 1.]. Predict H % of the trajectory"
            self.pred_horizon = pred_horizon
        elif isinstance(pred_horizon, int):
            assert pred_horizon >= 1, "At least we need to forecast a single dynamics step"
            self.pred_horizon = pred_horizon
        self.eval_pred_horizon = eval_pred_horizon
        self.batch_size = batch_size
        self.num_workers = num_workers
        # Metadata and dynamics information
        self.metadata, self.dt = None, None
        self.state_obs = state_obs
        self.action_obs = action_obs
        self._val_dataloader, self._test_dataloader, self._train_dataloader = None, None, None
        # Symmetry parameters
        self.symm_group = None
        self.measurements_reps = {}
        self.gspace = None
        self.state_field_type, self.action_field_type = None, None
        self.device = device

    def prepare_data(self):

        if self.prepared:
            self._train_dataset = self._train_dataset.shuffle(buffer_size=self._train_dataset.dataset_size / 2)
            self._train_dataloader = DataLoader(dataset=self._train_dataset, batch_size=self.batch_size,
                                                num_workers=self.num_workers,
                                                persistent_workers=True if self.num_workers > 0 else False,
                                                collate_fn=self.data_augmentation_collate_fn if self.augment else
                                                self.collate_fn)
            log.info(f"Train dataset reshuffled")
            return

        start_time = time.time()
        log.info(f"Preparing datasets {self._data_path}")

        a = list(self._data_path.rglob('*train.pkl'))
        dyn_sys_data = set([a.parent for a in list(self._data_path.rglob('*train.pkl'))])
        if self.noise_level is not None:
            system_data_path = [path for path in dyn_sys_data if f"noise_level={self.noise_level}" in str(path)]
        else:
            system_data_path = dyn_sys_data

        if len(system_data_path) > 1:
            raise RuntimeError(f"Multiple potential paths {system_data_path} found")
        elif len(system_data_path) == 0:
            raise ValueError(f"No recordings found in {self._data_path}")
        system_data_path = system_data_path.pop()

        train_data, test_data, val_data = get_train_test_val_file_paths(system_data_path)
        # Obtain hugging face Iterable datasets instances
        datasets, dynamics_recording = get_dynamics_dataset(train_shards=train_data,
                                                            test_shards=test_data,
                                                            val_shards=val_data,
                                                            train_pred_horizon=self.pred_horizon,
                                                            eval_pred_horizon=self.eval_pred_horizon,
                                                            frames_per_step=self.frames_per_step,
                                                            state_obs=self.state_obs,
                                                            action_obs=self.action_obs)
        self.metadata: DynamicsRecording = dynamics_recording
        # observations_names = self.metadata.
        self.dt = dynamics_recording.dynamics_parameters['dt']
        # In case no measurements are passed, we recover the ones from the DynamicsRecording
        self.state_obs = dynamics_recording.state_obs if self.state_obs is None else self.state_obs
        self.action_obs = dynamics_recording.action_obs if self.action_obs is None else self.action_obs

        # Ensure samples contain torch.Tensors and not numpy arrays.
        # Apply map to obtain flat state/next_state action/next_action values
        train_dataset, test_dataset, val_dataset = datasets

        # Ensure what we shuffle the train dataset:
        train_dataset = train_dataset.shuffle(buffer_size=min(train_dataset.dataset_size // 4, 5000))
        test_dataset = test_dataset.shuffle(buffer_size=min(train_dataset.dataset_size // 4, 1000), seed=18)
        val_dataset = val_dataset.shuffle(buffer_size=min(train_dataset.dataset_size // 4, 1000), seed=18)
        # Convert to torch. Apply map to get samples containing state and next state
        obs_to_remove = train_dataset.features.keys()
        train_dataset = train_dataset.with_format("torch").map(
            DynamicsRecording.map_state_next_state, batched=True, fn_kwargs={'state_observations': self.state_obs},
            remove_columns=tuple(obs_to_remove))
        test_dataset = test_dataset.with_format("torch").map(
            DynamicsRecording.map_state_next_state, batched=True, fn_kwargs={'state_observations': self.state_obs},
            remove_columns=tuple(obs_to_remove))
        val_dataset = val_dataset.with_format("torch").map(
            DynamicsRecording.map_state_next_state, batched=True, fn_kwargs={'state_observations': self.state_obs},
            remove_columns=tuple(obs_to_remove))

        self._train_dataset = train_dataset
        self._test_dataset = test_dataset
        self._val_dataset = val_dataset

        # Rebuilt the ESCNN representations of measurements _________________________________________________________
        # TODO: Handle dyn systems without symmetries
        # G_domain = escnn.group.O3()
        self.symm_group = dynamics_recording.dynamics_parameters['group']

        # Construct state (and action) representations considering the `frames_per_step` and the concatenation
        # convention of the function `map_state_next_state` in `DynamicsRecording.py`. Which defines the state
        # with `F=frames_per_step` delayed coordinates as s_t = [m1_f,..., m1_f+F, m2_f,..., m2_f+F, ...] where mi_k
        # is the measurement i at frame k.
        state_reps = [[dynamics_recording.obs_representations[m]] * self.frames_per_step for m in self.state_obs]
        state_reps = [rep for frame_reps in state_reps for rep in frame_reps]  # flatten list of reps

        # Use as default no basis space # TODO make more flexible
        self.gspace = escnn.gspaces.no_base_space(self.symm_group)
        # Define the state field type
        self.state_field_type = FieldType(self.gspace, representations=state_reps)
        # Define the action field type
        self.action_field_type = None
        if len(self.action_obs) > 0:
            # construct action representations analog to the state representations
            action_reps = [[dynamics_recording.obs_representations[m]] * self.frames_per_step for m in self.action_obs]
            action_reps = [rep for frame_reps in action_reps for rep in frame_reps]  # flatten list of reps
            self.action_field_type = FieldType(self.gspace, representations=action_reps)

        self._train_dataloader = DataLoader(dataset=train_dataset, batch_size=self.batch_size,
                                            num_workers=self.num_workers,
                                            persistent_workers=True if self.num_workers > 0 else False,
                                            collate_fn=self.data_augmentation_collate_fn if self.augment else
                                            self.collate_fn)
        batch_size = min(self.batch_size, test_dataset.dataset_size)
        self._test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False,
                                           persistent_workers=True if self.num_workers > 0 else False,
                                           num_workers=self.num_workers,
                                           # pin_memory=True,
                                           collate_fn=self.collate_fn)
        batch_size = min(self.batch_size, test_dataset.dataset_size)
        self._val_dataloader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False,
                                          persistent_workers=True if self.num_workers > 0 else False,
                                          num_workers=self.num_workers,
                                          # pin_memory=True,
                                          collate_fn=self.collate_fn)

        # Configure the prediction dataloader for the approximating and evaluating the transfer operator. This will
        # be a dataloader passing state and next state single step measurements:
        datasets, dynamics_recording = get_dynamics_dataset(train_shards=train_data,
                                                            test_shards=test_data,
                                                            val_shards=val_data,
                                                            train_pred_horizon=1,
                                                            eval_pred_horizon=1,
                                                            frames_per_step=self.frames_per_step,
                                                            state_obs=self.state_obs,
                                                            action_obs=self.action_obs)
        transfer_op_train_dataset, _, _ = datasets
        transfer_op_train_dataset = transfer_op_train_dataset.with_format("torch").map(
            DynamicsRecording.map_state_next_state, batched=True, fn_kwargs={'state_observations': self.state_obs})

        self._trans_op_dataloader = DataLoader(dataset=transfer_op_train_dataset,
                                               batch_size=transfer_op_train_dataset.dataset_size,  # Single batch
                                               pin_memory=False, num_workers=self.num_workers, shuffle=False,
                                               collate_fn=self.collate_fn)

        log.info(f"Data preparation done in {time.time() - start_time:.2f} [s]")

    def compute_loss_metrics(self, predictions: dict, inputs: dict) -> (torch.Tensor, dict):
        """
        Compute the loss and metrics from the predictions and inputs
        :param predictions: dict of tensors with the predictions
        :param inputs: dict of tensors with the inputs
        :return: loss: torch.Tensor, metrics: dict
        """
        raise NotImplementedError("Implement this function in the derived class")

    def train_dataloader(self):
        return self._train_dataloader

    def val_dataloader(self):
        return self._val_dataloader

    def test_dataloader(self):
        return self._test_dataloader

    def predict_dataloader(self):
        return self._trans_op_dataloader

    @property
    def prepared(self):
        return self._train_dataloader is not None

    def collate_fn(self, batch_list: list) -> dict:
        batch = torch.utils.data.default_collate(batch_list)
        for k, v in batch.items():
            batch[k] = v.to(self.device)
        return batch

    def data_augmentation_collate_fn(self, batch_list: list) -> dict:
        batch = torch.utils.data.default_collate(batch_list)
        state = batch['state']
        next_state = batch['next_state']

        action = batch.get('action', None)
        next_action = batch.get('next_action', None)

        # Sample a random symmetry transformation
        g = self.symm_group.sample()
        if g == self.symm_group.identity:  # Avoid the computational overhead of applying the identity
            return batch

        rep_state = self.state_field_type.fiber_representation(g).to(dtype=state.dtype, device=state.device)
        # Use einsum notation to apply the tensor operations required. Here o=state_dim is the output dimension,
        # s=state_dim is the input dimension, b=batch_size, t=horizon, ... = arbitrary dimensions
        g_state = torch.einsum("os,bs...->bo...", rep_state, state)
        g_next_state = torch.einsum("os,bts...->bto...", rep_state, next_state)

        batch['state'] = g_state
        batch['next_state'] = g_next_state
        if action is not None:
            rep_action = self.action_field_type.fiber_representation(g).to(dtype=state.dtype, device=state.device)
            g_action = torch.einsum("oa,ba...->bo...", rep_action, action)
            g_next_action = torch.einsum("oa,bta...->bto...", rep_action, next_action)
            batch['action'] = g_action
            batch['next_action'] = g_next_action
        return batch

    def plot_sample_trajs(self):
        num_trajs = 5
        fig = None
        styles = {'Train': dict(width=3, dash='solid'),
                  'Test': dict(width=2, dash='2px'),
                  'Val': dict(width=1, dash='5px')}
        for partition, dataloader in zip(['Train', 'Test', 'Val'],
                                         [self.train_dataloader(), self.test_dataloader(), self.val_dataloader()]):
            batch = next(iter(dataloader))
            state = batch['state']
            next_state = batch['next_state']
            state_traj = traj_from_states(state, next_state)

            fig = plot_trajectories(state_traj, fig=fig, dt=self.dt, main_style=styles[partition],
                                    main_legend_label=partition, n_trajs_to_show=num_trajs, title="Sample Trajectories")
        fig.show()


if __name__ == "__main__":
    path_to_data = Path(__file__).parent
    assert path_to_data.exists(), f"Invalid Dataset path {path_to_data.absolute()}"

    # Find all dynamic systems recordings
    path_to_data /= Path('mini_cheetah') / 'recordings' / 'grass'
    path_to_dyn_sys_data = set([a.parent for a in list(path_to_data.rglob('*train.pkl'))])
    # Select a dynamical system
    mock_path = path_to_dyn_sys_data.pop()

    data_module = DynamicsDataModule(data_path=mock_path,
                                     pred_horizon=10,
                                     eval_pred_horizon=100,
                                     frames_per_step=1,
                                     num_workers=1,
                                     batch_size=1000,
                                     augment=False,
                                     state_obs=('q',),
                                     action_obs=tuple(),
                                     )

    # Test loading of the DynamicsRecording
    data_module.prepare_data()
    data_module.plot_sample_trajs()
    s = next(iter(data_module._train_dataset))
    states, state_trajs = None, None
    fig = None

    for partition, dataloader in zip(['Test', 'Train', 'Validation'],
                                     [data_module.test_dataloader(), data_module.train_dataloader(),
                                      data_module.val_dataloader()]):
        start_time = time.time()
        print(f"Testing {partition} set")
        print(f"Shuffling...")
        next(iter(dataloader))
        print(f"Shuffling done in {time.time() - start_time:.2f} [s]")
        n_samples = 0
        for i, batch in enumerate(dataloader):
            states = batch['state']
            next_states = batch['next_state']
            n_samples += states.shape[0]
            if n_samples > 1000:
                break
        # print the time in [ms]
        print(f"Average time per sample in {partition} set: {(time.time() - start_time) / n_samples * 1000:.2f} [ms]")
        # break
