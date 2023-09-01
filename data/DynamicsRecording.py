import logging
import math
import pickle
from dataclasses import dataclass, field
from math import floor
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Union

import datasets
import numpy as np
from datasets import Features, IterableDataset

from utils.mysc import compare_dictionaries

log = logging.getLogger(__name__)


@dataclass
class DynamicsRecording:
    """Data structure to store recordings of a Markov Dynamics."""

    description: Optional[str] = None
    info: Dict[str, object] = field(default_factory=dict)
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


def load_data_generator(recordings: list[DynamicsRecording],
                        frames_per_step: int = 1,
                        prediction_horizon: Union[int, float] = 1,
                        state_measurements: Optional[list[str]] = None,
                        action_measurements: Optional[list[str]] = None):
    """Generator that yields measurement samples of length `n_frames_per_state` from the Markov Dynamics recordings.

    Args:
        recordings (list[DynamicsRecording]): List of DynamicsRecordings.
        frames_per_step: Number of frames to compose a single measurement sample at time `t`. E.g. if `f` is provided
        the state samples will be of shape [f, measurement_dim].
        prediction_horizon (int, float): Number of future time steps to include in the next time samples.
            E.g: if `n` is an integer the samples will be of shape [n, frames_per_state, measurement_dim]
            If `n` is a float, then the samples will be of shape [int(n*traj_length), frames_per_state, measurement_dim]
        state_measurements: Ordered list of measurements names composing the state space.
        action_measurements: Ordered list of measurements names composing the action space.

    Returns:
        A dictionary containing the measurement samples at time `t` and `[t+1, t + pred_horizon]` for each measurement.
    """
    for file_data in recordings:
        # file_data = DynamicsRecording.load_from_file(file_path)
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
            if isinstance(prediction_horizon, float):
                steps_in_pred_horizon = floor((prediction_horizon * traj_length) // frames_per_step) - 1
            else:
                steps_in_pred_horizon = prediction_horizon

            remnant = traj_length % frames_per_step
            frames_in_pred_horizon = steps_in_pred_horizon * frames_per_step
            # Iterate over the frames of the trajectory
            for frame in range(traj_length - frames_per_step):
                # Collect the next steps until the end of the trajectory. If the prediction horizon is larger than
                # the remaining steps (prediction outside trajectory length), then we continue to the next traj.
                # This is better computationally for avoiding copying while batch processing data later.
                if frame + frames_per_step + frames_in_pred_horizon > (traj_length - remnant):
                    continue
                sample = {}
                for measurement, trajs in recordings.items():
                    # Enforce Float 32
                    trajs = np.asarray(trajs, dtype=np.float32)
                    sample[measurement] = trajs[traj_id][frame:frame + frames_per_step]
                    horizon = []
                    num_steps = frames_in_pred_horizon // frames_per_step
                    for step_id in range(1, num_steps + 1):
                        step = trajs[traj_id][frame + (frames_per_step * step_id):
                                              frame + (frames_per_step * step_id) + frames_per_step]
                        horizon.append(step)
                    sample[f"next_{measurement}"] = np.asarray(horizon)
                if len(sample[f"next_{measurement}"]) != steps_in_pred_horizon:
                    raise ValueError(f"Issue")
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
        measurement = np.asarray(sample[m])
        next_measurement = np.asarray(sample[f"next_{m}"])
        # Preserve the prediction horizon and batch dimensions and flatten the rest
        flat_s.append(np.reshape(measurement, newshape=measurement.shape[:-2] + (-1,)))
        flat_next_s.append(np.reshape(next_measurement, newshape=next_measurement.shape[:-2] + (-1,)))
    # Define the state at time t and the states at time [t+1, t+pred_horizon]
    state = np.concatenate(flat_s, axis=-1)
    next_state = np.concatenate(flat_next_s, axis=-1)
    return dict(state=state, next_state=next_state)


def estimate_dataset_size(recordings: list[DynamicsRecording], prediction_horizon: Union[int, float] = 1,
                          frames_per_step: int = 1):
    num_trajs = 0
    num_samples = 0
    steps_pred_horizon = []
    for r in recordings:
        r_num_trajs = r.info['num_traj']
        r_traj_length = r.info['trajectory_length']
        if isinstance(prediction_horizon, float):
            steps_in_pred_horizon = floor((prediction_horizon * r_traj_length) // frames_per_step)
        else:
            steps_in_pred_horizon = prediction_horizon
        steps_pred_horizon.append(steps_in_pred_horizon)
        frames_in_pred_horizon = steps_in_pred_horizon * frames_per_step
        samples = r_traj_length - frames_in_pred_horizon - (r_traj_length % frames_per_step) + 1
        num_samples += r_num_trajs * samples
        num_trajs += r_num_trajs
    steps_pred_horizon = np.mean(steps_pred_horizon)
    log.debug(f"Steps in prediction horizon {int(steps_pred_horizon)}")
    return num_trajs, num_samples


def get_dynamics_dataset(train_shards: list[Path],
                         test_shards: list[Path],
                         val_shards: List[Path],
                         num_proc: int = 1,
                         frames_per_step: int = 1,
                         train_pred_horizon: Union[int, float] = 1,  # 1 step ahead
                         eval_pred_horizon: Union[int, float] = 0.5,  # 50% of the trajectory
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
        features[measurement] = datasets.Array2D(shape=(frames_per_step, dim), dtype='float32')
        features[f"next_{measurement}"] = datasets.Array2D(shape=(frames_per_step, dim), dtype='float32')

    # Calculate the size of all train shard files in Mbs
    train_data_size = sum([f.stat().st_size for f in train_shards]) / (1024 ** 2)
    keep_in_memory = train_data_size < 100
    part_datasets = []
    for partition, partition_shards in zip(["train", "test", "val"], [train_shards, test_shards, val_shards]):
        recordings = [DynamicsRecording.load_from_file(f) for f in partition_shards]
        if partition == "train":
            pred_horizon = train_pred_horizon
        elif partition == "val":
            pred_horizon = eval_pred_horizon
        else:
            pred_horizon = 0.5

        num_trajs, num_samples = estimate_dataset_size(recordings, pred_horizon, frames_per_step)
        dataset = IterableDataset.from_generator(load_data_generator,
                                                 # keep_in_memory=keep_in_memory,
                                                 features=Features(features),
                                                 gen_kwargs=dict(recordings=recordings,
                                                                 frames_per_step=frames_per_step,
                                                                 prediction_horizon=pred_horizon,
                                                                 state_measurements=state_measurements,
                                                                 action_measurements=action_measurements),
                                                 )

        for sample in dataset:
            state = sample['state']
            next_state = sample['next_state']
            time_horizon = next_state.shape[1] + 1
            log.debug(f"[Dataset {partition} - Trajs:{num_trajs} - Samples: {num_samples} - "
                      f"Frames per sample : {frames_per_step * time_horizon}]-----------------------------")
            log.debug(f"\tstate: {state.shape} = (frames_per_step, state_dim)")
            log.debug(f"\tnext_state: {next_state.shape} = (pred_horizon, frames_per_step, state_dim)")
            break

        dataset.info.dataset_size = num_samples
        dataset.info.dataset_name = f"[{partition}] Linear dynamics"
        dataset.info.description = metadata.description
        # dataset = partition
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
    path_to_data /= 'linear_system'
    path_to_dyn_sys_data = set([a.parent for a in list(path_to_data.rglob('*train.pkl'))])
    # Select a dynamical system
    mock_path = path_to_dyn_sys_data.pop()
    # Obtain the training, testing and validation file paths containing distinct trajectories of motion.
    train_data, test_data, val_data = get_train_test_val_file_paths(mock_path)
    # Obtain hugging face Iterable datasets instances
    pred_horizon = .1
    frames_per_state = 10
    (train_dataset, test_dataset, val_dataset), metadata = get_dynamics_dataset(train_shards=train_data,
                                                                                test_shards=test_data,
                                                                                val_shards=val_data,
                                                                                frames_per_step=frames_per_state,
                                                                                train_pred_horizon=pred_horizon,
                                                                                eval_pred_horizon=0.5)

    # test_flat = map_state_next_state(sample, metadata.state_measurements)
    # Test the map
    torch_dataset = test_dataset.with_format("torch").shuffle()
    torch_dataset = torch_dataset.map(map_state_next_state, batched=True, fn_kwargs={'state_measurements': ['state']})

    # Test errors while looping.
    for i, s in enumerate(torch_dataset):
        pass
        # log.debug(s)
        # if i > 1000:
        #     break

    log.debug(i)
