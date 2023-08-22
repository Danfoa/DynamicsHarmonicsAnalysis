import logging
import math
import pickle
from dataclasses import dataclass, field
from math import floor
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import datasets
import numpy as np
from datasets import Features, IterableDataset

from utils.mysc import compare_dictionaries

log = logging.getLogger(__name__)


@dataclass
class DynamicsRecording:
    """Data structure to store recordings of a Markov Dynamics."""

    description: Optional[str] = None
    dynamics_parameters: Dict = field(default_factory=lambda: {'dt': None})

    # Dictionary providing the map between measurement name and measurement dimension
    measurements: Dict[str, int] = field(default_factory=dict)

    # Ordered list of measurements composing to the state and action space of the Markov Process.
    state_measurements: List[str] = field(default_factory=list)
    action_measurements: List[str] = field(default_factory=list)

    # Named group representations needed to transform all measurements.
    # The keys are the representation names used by `measurements_representations`.
    reps_irreps: Dict[str, Iterable] = field(default_factory=dict)
    reps_change_of_basis: Dict[str, Iterable] = field(default_factory=dict)
    # Map from measurement name to the measurement representation name. This name should be in `group_representations`.
    measurements_representations: Dict[str, str] = field(default_factory=dict)

    recordings: Dict[str, Iterable] = field(default_factory=dict)

    def save_to_file(self, file_path: Path):
        with file_path.with_suffix(".pkl").open('wb') as file:
            pickle.dump(self, file, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load_from_file(file_path: Path, only_metadata=False):
        with file_path.with_suffix(".pkl").open('rb') as file:
            data = pickle.load(file)
            if only_metadata:
                del data.recordings
        return data


def load_data_generator(shards: list[Path],
                        frames_per_state: int = 1,
                        prediction_horizon: int = 1,
                        state_measurements: Optional[list[str]] = None,
                        action_measurements: Optional[list[str]] = None):
    """Generator that yields measurement samples of length `n_frames_per_state` from the Markov Dynamics recordings.

    Args:
        shards (list[pathlib.Path]): List of Path files containing DynamicsRecordings.
        frames_per_state: Number of frames to compose a single measurement sample at time `t`. E.g. if `f` is provided
        the state samples will be of shape [f, measurement_dim].
        prediction_horizon: Number of future time steps to include in the next time samples. E.g., if `n` is provided
        the next_measurement samples will be of shape [n, frames_per_state, measurement_dim]
        state_measurements: Ordered list of measurements names composing the state space.
        action_measurements: Ordered list of measurements names composing the action space.

    Returns:
        A dictionary containing the measurement samples at time `t` and `[t+1, t + pred_horizon]` for each measurement.
    """
    for file_path in shards:
        file_data = DynamicsRecording.load_from_file(file_path)
        if state_measurements is not None:
            file_data.state_measurements = file_data.measurements
        if action_measurements is not None:
            file_data.action_measurements = file_data.measurements

        recordings = file_data.recordings
        # Get any measurement list of trajectories and count the number of trajectories
        # We assume all measurements have the same number of trajectories
        n_trajs = len(next(iter(recordings.values())))

        # Since we assume trajectories can have different lengths, we iterate over each trajectory
        # and generate samples of length `n_frames_per_state` from each trajectory.
        for traj_id in range(n_trajs):
            traj_length = next(iter(recordings.values()))[traj_id].shape[0]

            # Iterate over the frames of the trajectory
            total_steps = math.floor(traj_length / frames_per_state)
            for frame in range(traj_length - frames_per_state):
                current_step = math.floor((frame + 1) / frames_per_state) + 1
                remaining_steps = total_steps - current_step
                # Collect the next steps until the end of the trajectory. If the prediction horizon is larger than
                # the remaining steps (prediction outside trajectory length), then we continue to the next traj.
                # This is better computationally for avoiding copying while batch processing data later.
                if remaining_steps - 1 <= prediction_horizon:
                    continue
                sample = {}
                for measurement, trajs in recordings.items():
                    sample[measurement] = trajs[traj_id][frame:frame + frames_per_state]
                    horizon = []
                    for h in range(1, min(prediction_horizon + 1, remaining_steps)):
                        horizon.append(
                            trajs[traj_id][frame + (frames_per_state * h):
                                           frame + (frames_per_state * h) + frames_per_state])
                    sample[f"next_{measurement}"] = np.asarray(horizon)
                yield sample


def map_state_action_state(sample, state_measurements: List[str], action_measurements: List[str]) -> dict:
    """Map composing multiple measurements to single state, action, next_state samples."""
    flat_sample = map_state_next_state(sample, state_measurements)
    # Reuse the same function to for flattening action and next_action
    action_sample = map_state_next_state(sample, action_measurements)
    action_sample["action"] = action_sample.pop("state")
    action_sample["next_action"] = action_sample.pop("next_state")
    flat_sample.update(action_sample)
    return flat_sample


def map_state_next_state(sample: dict, state_measurements: List[str]) -> dict:
    """Map composing multiple frames of measurements into a flat vectors `state` and `next_state` samples.

    This method constructs the state `s_t` and history of nex steps `s_t+1` of the Markov Process.
    The state is defined as a set of measurements within a window of fps=`frames_per_state`.
    E.g.: Consider the state is defined by the measurements [m=momentum, p=position] at `fps` consecutive frames.
        Then the state at time `t` is defined as `s_t = [m_f, m_f+1,..., m_f+fps, p_f, p_f+1, ..., p_f+fps]`.
        Where we use f to denote frame in time to make the distinction from the time index `t` of the Markov Process.
        Then, the next state is defined as `s_t+1 = [m_f+fps,..., m_fps+fps, p_f+fps, ..., p_f+fps+fps]`.
    Args:
        sample (dict): Dictionary containing the measurements of the system of shape [state_time, f].
        state_measurements: Ordered list of measurements names composing the state space.
    Returns:
        A dictionary containing the MDP state `s_t` and the next_state/s `[s_t+1, s_t+2, ..., s_t+pred_horizon]`.
    """

    # Flatten measurements a_t = [a_f, a_f+1, af+2, ..., a_f+F] s.t. a_t in R^{F * dim(a)}, a_f in R^{dim(a)}
    flat_s, flat_next_s = [], []
    for m in state_measurements:
        measurement = np.asarray(sample[m])                 # To avoid copying we must force all samples of equal shape
        next_measurement = np.asarray(sample[f"next_{m}"])  # To avoid copying we must force all samples of equal shape
        # Preserve the prediction horizon and batch dimensions and flatten the rest
        flat_s.append(np.reshape(measurement, newshape=measurement.shape[:-2] + (-1,)))
        flat_next_s.append(np.reshape(next_measurement, newshape=next_measurement.shape[:-2] + (-1,)))
    # Define the state at time t and the states at time [t+1, t+pred_horizon]
    state = np.concatenate(flat_s, axis=-1)
    next_state = np.concatenate(flat_next_s, axis=-1)
    return dict(state=state, next_state=next_state)


def get_dynamics_dataset(train_shards: list[Path],
                         test_shards: list[Path],
                         val_shards: List[Path],
                         num_proc: int = 1,
                         frames_per_state: int = 1,
                         prediction_horizon: int = 1,
                         state_measurements: Optional[list[str]] = None,
                         action_measurements: Optional[list[str]] = None
                         ) -> tuple[list[IterableDataset], DynamicsRecording]:
    """Load Markov Dynamics recordings from a list of files and return a train, test and validation dataset."""
    # TODO: ensure all shards come from the same dynamical system
    metadata = DynamicsRecording.load_from_file(train_shards[0], only_metadata=True)
    test_metadata = DynamicsRecording.load_from_file(test_shards[0], only_metadata=True)

    dyn_params_diff = compare_dictionaries(metadata.dynamics_parameters, test_metadata.dynamics_parameters)
    assert len(dyn_params_diff) == 0, "Different dynamical systems loaded in train/test sets"

    if state_measurements is not None:
        assert np.all([m in metadata.measurements for m in state_measurements])
        metadata.state_measurements = state_measurements

    if action_measurements is not None:
        assert np.all([m in metadata.measurements for m in action_measurements])
        metadata.action_measurements = action_measurements

    features = {}
    for measurement, dim in metadata.measurements.items():
        features[measurement] = datasets.Array2D(shape=(frames_per_state, dim), dtype='float32')
        features[f"next_{measurement}"] = datasets.Array2D(shape=(frames_per_state, dim), dtype='float32')

    part_datasets = []
    for partition, partition_shards in zip(["train", "test", "val"], [train_shards, test_shards, val_shards]):
        dataset = IterableDataset.from_generator(load_data_generator,
                                                 features=Features(features),
                                                 gen_kwargs=dict(shards=partition_shards,
                                                                 frames_per_state=frames_per_state,
                                                                 prediction_horizon=prediction_horizon,
                                                                 state_measurements=state_measurements,
                                                                 action_measurements=action_measurements),
                                                 )
        dataset.info.dataset_name = f"[{partition}] Linear dynamics"
        dataset.info.description = metadata.description
        part_datasets.append(dataset)

    return part_datasets, metadata


def get_train_test_val_file_paths(data_path: Path):
    """Search in folder for files ending in train/test/val.pkl and return a list of paths to each file."""
    train_data, test_data, val_data = [], [], []
    for file_path in data_path.iterdir():
        if file_path.name.endswith("train.pkl"):
            train_data.append(file_path)
        elif file_path.name.endswith("test.pkl"):
            test_data.append(file_path)
        elif file_path.name.endswith("val.pkl"):
            val_data.append(file_path)
    return train_data, test_data, val_data


if __name__ == "__main__":
    path_to_data = Path(__file__).parent
    assert path_to_data.exists(), f"Invalid Dataset path {path_to_data.absolute()}"

    # Find all dynamic systems recordings
    path_to_data /= 'linear_systems'
    path_to_dyn_sys_data = set([a.parent for a in list(path_to_data.rglob('*train.pkl'))])
    # Select a dynamical system
    mock_path = path_to_dyn_sys_data.pop()
    # Obtain the training, testing and validation file paths containing distinct trajectories of motion.
    train_data, test_data, val_data = get_train_test_val_file_paths(mock_path)
    # Obtain hugging face Iterable datasets instances
    pred_horizon = 2
    frames_per_state = 2
    (train_dataset, test_dataset, val_dataset), metadata = get_dynamics_dataset(train_shards=train_data,
                                                                                test_shards=test_data,
                                                                                val_shards=val_data,
                                                                                frames_per_state=frames_per_state,
                                                                                prediction_horizon=pred_horizon)
    sample = next(iter(train_dataset))
    # Test the flattening of the state and next_state
    # test_flat = map_state_next_state(sample, metadata.state_measurements)
    # Test the map
    torch_train = train_dataset.with_format("torch").shuffle()
    torch_train = torch_train.map(map_state_next_state, batched=True, fn_kwargs={'state_measurements': ['state']})

    # Test errors while looping.
    for i, s in enumerate(torch_train):
        if i > 10000:
            break
