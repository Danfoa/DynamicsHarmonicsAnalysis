import copy
import logging
import time
from pathlib import Path
from typing import Any, Optional, Union

import escnn.group
import numpy as np
import torch
from datasets.distributed import split_dataset_by_node
from escnn.group import Representation
from escnn.nn import FieldType
from lightning import LightningDataModule
from torch.utils.data import DataLoader, default_collate
from tqdm import tqdm

from morpho_symm.data.DynamicsRecording import DynamicsRecording, split_train_val_test
from utils.mysc import traj_from_states
from utils.plotting import plot_system_3D, plot_trajectories, plot_two_panel_trajectories
from kooplearn.data import multi_traj_to_context, TensorContextDataset

log = logging.getLogger(__name__)


class DynamicsDataModule(LightningDataModule):

    def __init__(self,
                 data_path: Path,
                 pred_horizon: Union[int, float],
                 val_pred_horizon: Union[int, float],
                 test_pred_horizon: Union[int, float],
                 lookback_len: int = 1,
                 split_ratios: tuple = (0.7, 0.15, 0.15),  # Train, Test, Validation
                 train_ratio: float = 1.0,   # Percentage of trining data to use. Useful for debugging
                 data_augmentation: bool = False,
                 state_obs: Optional[tuple[str]] = None,
                 standardize: bool = True,
                 system_cfg: Optional[dict] = None,
                 batch_size: int = 256,
                 num_workers: int = 0,
                 device: torch.device = torch.device('cpu')
                 ):
        """ Data module for dynamics model training and evaluation using data from a DynamicsRecording object.

        Args:
            data_path (Path): Path to the directory containing the recordings of the dynamics system.
            pred_horizon (Union[int, float], optional): Length of the context_windows (past/present frames + future
            frames).
                If float, it represents the fraction of the entire trajectory time horizon.
            val_pred_horizon (Union[int, float], optional): Length of the context_windows (past/present frames +
            future frames)
                for the validation set. If float, it represents the fraction of the entire trajectory time horizon.
            test_pred_horizon (Union[int, float], optional): Length of the context_windows (past/present frames +
            future frames)
                for the test set. If float, it represents the fraction of the entire trajectory time horizon.
            lookback_len (int, optional): Number of frames to consider in the past for each time step. Defaults to 1.
            split_ratios (tuple, optional): Ratios for splitting the dataset into train, test, and validation sets.
                Defaults to (0.7, 0.15, 0.15).
            data_augmentation (bool, optional): Apply symmetric data augmentation to training and validation sets if
                the state group representation is defined in the DynamicsRecording. Defaults to False.
            state_obs (Optional[tuple[str]], optional): Ordered names of the scalar/vector-valued observations in the
                DynamicsRecording that compose the system's state. E.g. if the DynamicsRecording has the observations
                ['energy', 'momentum', 'position', 'velocity'], and state_obs=('position', 'velocity'), then the
                state samples `x_t = [<position>, <velocity>]` will be composed only by the 'position' and 'velocity'
                observations. Defaults to None, in which case the DynamicsRecording's default state_obs is used.
            standardize (bool, optional): Standardize the state and action observations by subtracting the mean and
                dividing by the standard deviation. Defaults to True. If the state group representation is defined this
                standardization computes the symmetry-corrected mean and variance.
            system_cfg (Optional[dict], optional): Configuration dictionary with system parameters. Defaults to None.
            batch_size (int, optional): Batch size for the dataloaders. Defaults to 256.
            num_workers (int, optional): Number of workers for the dataloaders. Defaults to 0.
        """

        super().__init__()
        if system_cfg is None:
            system_cfg = {}
        self._data_path = data_path
        self.system_cfg = system_cfg if system_cfg is not None else {}
        assert np.isclose(np.sum(split_ratios), 1.0), "Dataset train/val/split ratios must sum to 1.0"
        self.split_ratios = split_ratios
        assert 0.01 < train_ratio <= 1.0, "Train ratio must be in (0.01, 1.0]"
        self.train_ratio = train_ratio
        self.augment = data_augmentation
        self.lookback_len = lookback_len

        if isinstance(pred_horizon, float):
            assert 0 < pred_horizon <= 1.0, "Prediction horizon need be in H∈(0., 1.]. Predict H % of the trajectory"
            self.pred_horizon = pred_horizon
        elif isinstance(pred_horizon, int):
            assert pred_horizon >= 1, "At least we need to forecast a single dynamics step"
            self.pred_horizon = pred_horizon

        self.eval_pred_horizon = val_pred_horizon
        self.test_pred_horizon = test_pred_horizon
        self.batch_size = batch_size
        self.num_workers = num_workers
        # Metadata and dynamics information
        self.state_obs = state_obs
        self.standardize = standardize

        # Symmetry group and group representations parameters
        self.symm_group = None
        self.gspace = None
        self.state_type = None  # escnn.FieldType object. Only used when DynamicsRecording has symmetry information
        self.device = device

        self.dt = None, None
        self.state_mean, self.state_var = None, None
        self._dynamic_recording = None  # DynamicsRecording object/s containing the recordings of observations. Private!
        self._val_dataloader, self._test_dataloader, self._train_dataloader = None, None, None
        self.train_dataset, self.val_dataset, self.test_dataset = None, None, None

    def prepare_data(self):

        if self.prepared:
            # Shuffle trajectory. TODO: This should not be done here, but with the Dataset class
            data = self.train_dataset.data
            idx = torch.randperm(data.shape[0])
            self.train_dataset.data = data[idx]
            log.info(f"Train dataset reshuffled")
            return

        # Find the DynamicsRecording files in the data path
        dyn_record_files = list(self._data_path.rglob('*.pkl'))
        self._log_data_sources(dyn_record_files)

        start_time = time.time()

        if len(dyn_record_files) > 1:
            raise NotImplementedError(f"Havent implemented gathering data from multiple recordings yet")
        elif len(dyn_record_files) == 0:
            raise ValueError(f"No `DynamicsRecordings` found in {self._data_path}")

        self._file_paths = copy.deepcopy(dyn_record_files)
        dyn_record_path = dyn_record_files.pop()

        # Load only the observables that are required to memory to avoid memory overflow
        self._dynamic_recording: DynamicsRecording = DynamicsRecording.load_from_file(
            dyn_record_path, obs_names=self.state_obs
            )

        train_data, val_data, test_data = split_train_val_test(
            dyn_recording=self._dynamic_recording, partition_sizes=self.split_ratios
            )

        if self.standardize:  # If standardization is enabled, use training data to estimate mean and variance
            train_data.state_moments()
            test_data.obs_moments = train_data.obs_moments
            val_data.obs_moments = train_data.obs_moments

        # Get the raw state trajectories of shape (traj_id, time, *state_dims) | GC will collect this references
        train_trajs = train_data.get_state_trajs(standardize=self.standardize)
        val_trajs = val_data.get_state_trajs(standardize=self.standardize)
        test_trajs = test_data.get_state_trajs(standardize=self.standardize)
        # Convert to np float32
        train_trajs = np.float32(train_trajs)
        val_trajs = np.float32(val_trajs)
        test_trajs = np.float32(test_trajs)

        # Split data into context windows using kooplean. Load data to device if available, and store reference of the
        # raw dataset to do required regression tasks.
        self.train_dataset = multi_traj_to_context(train_trajs,
                                                   context_window_len=self.pred_horizon + self.lookback_len,
                                                   backend="torch",
                                                   device=self.device)
        self.val_dataset = multi_traj_to_context(val_trajs,
                                                 context_window_len=self.eval_pred_horizon + self.lookback_len,
                                                 backend="torch",
                                                 device=self.device)
        self.test_dataset = multi_traj_to_context(test_trajs,
                                                  context_window_len=self.test_pred_horizon + self.lookback_len,
                                                  backend="torch",
                                                  device=self.device)

        # TODO: Remove from here.
        # Shuffle the samples of the partitions
        idx = torch.randperm(self.train_dataset.data.shape[0])
        self.train_dataset.data = self.train_dataset.data[idx]
        idx = torch.randperm(self.val_dataset.data.shape[0])
        self.val_dataset.data = self.val_dataset.data[idx]
        idx = torch.randperm(self.test_dataset.data.shape[0])
        self.test_dataset.data = self.test_dataset.data[idx]

        # TODO: Do somewhere else or more beautifully
        # If required (self.train_ratio< 1.0), reduce the training dataset size, for sample efficiency experiments
        if self.train_ratio < 1.0:
            n_samples = int(self.train_ratio * self.train_dataset.data.shape[0])
            self.train_dataset.data = self.train_dataset.data[:n_samples]

        self.dt = self._dynamic_recording.dynamics_parameters['dt']
        # In case no measurements are passed, we recover the ones from the DynamicsRecording file.
        self.state_obs = self._dynamic_recording.state_obs if self.state_obs is None else self.state_obs
        # Store normalization mean and variance parameters for the state space.
        self.state_mean, self.state_var = self._dynamic_recording.state_moments()

        # Rebuilt the ESCNN representations of measurements _________________________________________________________
        # G_domain = escnn.group.O3()
        self.symm_group = self._dynamic_recording.dynamics_parameters.get('group', None)
        if self.symm_group is not None:
            # TODO: This seems a bit to "experiment specific" code and should be removed from here.
            # however, we might need to provide this flexibility to the user.
            if 'subgroup_id' in self.system_cfg and self.system_cfg['subgroup_id'] is not None:
                subgroup_id = eval(self.system_cfg['subgroup_id'])
                if subgroup_id is not None:
                    # Restrict the symmetry group of the system to the subgroup with the given id
                    Gsub, sub2group, group2sub = self.symm_group.subgroup(subgroup_id)
                    new_reps = {}
                    for obs_name, obs_rep in self._dynamic_recording.obs_representations.items():
                        new_reps[obs_name] = obs_rep.restrict(subgroup_id)
                    self._dynamic_recording.obs_representations = new_reps
                    self.symm_group = Gsub
            # Use as default no basis space # TODO make more flexible
            self.gspace = escnn.gspaces.no_base_space(self.symm_group)
            # Define the state field type
            state_reps = self._dynamic_recording.state_representations()
            self.state_type = FieldType(self.gspace, representations=state_reps)
        log.info(f"Data preparation done in {time.time() - start_time:.2f} [s]")

    def _log_data_sources(self, dyn_record_files):
        # Group files by their parent folder
        data_folders = {}
        for file in dyn_record_files:
            parent_folder = str(file.parent)
            if parent_folder not in data_folders:
                data_folders[parent_folder] = []
            data_folders[parent_folder].append(file.name)

        log.info("Loading data from:")
        # Print to console where the data is coming from
        for folder, files in data_folders.items():
            log.info(f"[DIR]: {folder}:")
            for i, file in enumerate(files, start=1):
                log.info(f"\t\t [FILE] {i}: {file}")

    def setup(self, stage: str) -> None:
        log.info(f"Setting up {stage} dataset")

    def compute_loss_metrics(self, predictions: dict, inputs: dict) -> (torch.Tensor, dict):
        """
        Compute the loss and metrics from the predictions and inputs
        :param predictions: dict of tensors with the predictions
        :param inputs: dict of tensors with the inputs
        :return: loss: torch.Tensor, metrics: dict
        """
        raise NotImplementedError("Implement this function in the derived class")

    def train_dataloader(self):
        return DataLoader(dataset=self.train_dataset,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          collate_fn=self.collate_fn if not self.augment else self.data_augmentation_collate_fn,
                          persistent_workers=True if self.num_workers > 0 else False, drop_last=False)

    def val_dataloader(self):
        return DataLoader(dataset=self.val_dataset,
                          batch_size=self.batch_size,
                          persistent_workers=True if self.num_workers > 0 else False,
                          num_workers=self.num_workers,
                          collate_fn=self.collate_fn,
                          drop_last=False)

    def test_dataloader(self):
        return DataLoader(dataset=self.test_dataset,
                          batch_size=self.batch_size,
                          persistent_workers=True if self.num_workers > 0 else False,
                          collate_fn=self.collate_fn,
                          num_workers=self.num_workers,
                          drop_last=False)

    def predict_dataloader(self):
        return DataLoader(dataset=self.test_dataset,
                          batch_size=self.batch_size,
                          persistent_workers=True if self.num_workers > 0 else False,
                          collate_fn=self.collate_fn,
                          num_workers=self.num_workers,
                          drop_last=False)

    @property
    def prepared(self):
        return self.train_dataset is not None

    def collate_fn(self, batch) -> Any:
        if isinstance(batch, TensorContextDataset):
            return batch
        elif isinstance(batch, list) and isinstance(batch[0], TensorContextDataset):
            batch = torch.utils.data.default_collate([a.data for a in batch])
            return TensorContextDataset(batch)
        else:
            batch = torch.utils.data.default_collate(batch)
        return batch

    def data_augmentation_collate_fn(self, batch) -> Any:
        batch = self.collate_fn(batch)

        state = batch.data
        # Sample a random symmetry transformation
        g = self.symm_group.sample()
        if g == self.symm_group.identity:  # Avoid the computational overhead of applying the identity
            return batch

        rep_state = self.state_type.fiber_representation(g).to(dtype=state.dtype, device=state.device)
        # Use einsum notation to apply the tensor operations required. Here o=state_dim is the output dimension,
        # s=state_dim is the input dimension, b=batch_size, t=horizon, ... = arbitrary dimensions
        g_state = torch.einsum("os,...s->...o", rep_state, state)
        g_batch = TensorContextDataset(g_state)
        return g_batch

    def plot_sample_trajs(self):
        num_trajs = 3
        fig = None
        styles = {'Train': dict(width=3, dash='solid'),
                  'Test':  dict(width=2, dash='2px'),
                  'Val':   dict(width=1, dash='5px')}
        for partition, dataloader in zip(['Train', 'Test', 'Val'],
                                         [self.train_dataloader(), self.test_dataloader(), self.val_dataloader()]):
            state_traj = next(iter(dataloader))
            state_traj = state_traj.data.detach().cpu()
            fig = plot_trajectories(state_traj, fig=fig, dt=self.dt, main_style=styles[partition],
                                    main_legend_label=partition, n_trajs_to_show=num_trajs, title="Sample Trajectories")
        return fig



if __name__ == "__main__":
    path_to_data = Path(__file__).parent
    assert path_to_data.exists(), f"Invalid Dataset path {path_to_data.absolute()}"

    # Find all dynamic systems recordings
    path_to_data /= Path('mini_cheetah') / 'raysim_recordings' / 'flat' / 'forward_minus_0_4'
    # path_to_data = Path('/home/danfoa/Projects/koopman_robotics/data/linear_system/group=C10-dim=10/n_constraints=0/'
    #                     'f_time_constant=1.5[s]-frames=200-horizon=8.7[s]/noise_level=0')
    path_to_dyn_sys_data = set([a.parent for a in list(path_to_data.rglob('*train.pkl'))])
    # Select a dynamical system
    mock_path = path_to_dyn_sys_data.pop()

    data_module = DynamicsDataModule(data_path=mock_path,
                                     pred_horizon=10,
                                     val_pred_horizon=10,
                                     test_pred_horizon=150,
                                     lookback_len=1,
                                     num_workers=1,
                                     batch_size=1000,
                                     data_augmentation=False,
                                     standardize=True,
                                     # state_obs=('base_z_error', 'base_vel_error', 'base_ang_vel_error', ),
                                     # action_obs=tuple(),
                                     )

    # Test loading of the DynamicsRecording
    data_module.prepare_data()
    s = next(iter(data_module.train_dataset))

    data_module.plot_sample_trajs()
    states, state_trajs = None, None
    fig = None

    for partition, dataloader in zip(['Test', 'Train', 'Validation'],
                                     [data_module.test_dataloader(),
                                      data_module.train_dataloader(),
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
