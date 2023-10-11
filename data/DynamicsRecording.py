import logging
import math
import pickle
from dataclasses import dataclass, field
from math import floor
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union

import datasets
import numpy as np
from datasets import Features, IterableDataset
from escnn.group import Representation, groups_dict

from utils.mysc import compare_dictionaries

log = logging.getLogger(__name__)


@dataclass
class DynamicsRecording:
    """Data structure to store recordings of a Markov Dynamics."""

    description: Optional[str] = None
    info: Dict[str, object] = field(default_factory=dict)
    dynamics_parameters: Dict = field(default_factory=lambda: {'dt': None})

    # Ordered list of observations composing to the state and action space of the Markov Process.
    state_obs: Tuple[str] = field(default_factory=list)
    action_obs: Tuple[str] = field(default_factory=list)

    # Map from observation name to the observation representation name. This name should be in `group_representations`.
    obs_representations: Dict[str, Representation] = field(default_factory=dict)

    recordings: Dict[str, Iterable] = field(default_factory=dict)

    def save_to_file(self, file_path: Path):
        # Store representations and groups without serializing
        if len(self.obs_representations) > 0:
            self._obs_rep_irreps = {k: rep.irreps for k, rep in self.obs_representations.items()}
            self._obs_rep_names = {k: rep.name for k, rep in self.obs_representations.items()}
            self._obs_rep_Q = {k: rep.change_of_basis for k, rep in self.obs_representations.items()}
            group = self.obs_representations[self.state_obs[0]].group
            self._group_keys = group._keys
            self._group_name = group.__class__.__name__
            # Remove non-serializable objects
            del self.obs_representations
            self.dynamics_parameters.pop('group', None)

        with file_path.with_suffix(".pkl").open('wb') as file:
            pickle.dump(self, file, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load_from_file(file_path: Path, only_metadata=False):
        with file_path.with_suffix(".pkl").open('rb') as file:
            data: DynamicsRecording = pickle.load(file)
            if only_metadata:
                del data.recordings

            if hasattr(data, '_group_name'):
                group = groups_dict[data._group_name]._generator(**data._group_keys)  # Instanciate symmetry group
                data.dynamics_parameters['group'] = group
                data.obs_representations = {}
                for obs_name in data._obs_rep_irreps.keys():
                    irreps_ids = data._obs_rep_irreps[obs_name]
                    rep_name = data._obs_rep_names[obs_name]
                    rep_Q = data._obs_rep_Q[obs_name]
                    if rep_name in group.representations:
                        data.obs_representations[obs_name] = group.representations[rep_name]
                    else:
                        data.obs_representations[obs_name] = Representation(group, name=rep_name,
                                                                            irreps=irreps_ids, change_of_basis=rep_Q)
                    group.representations[rep_name] = data.obs_representations[obs_name]
        return data

    @property
    def obs_dims(self):
        """ Dictionary providing the map between observation name and observation dimension """
        return {k: v.shape[-1] for k, v in self.recordings.items()}

    @staticmethod
    def load_data_generator(dynamics_recordings: list["DynamicsRecording"],
                            frames_per_step: int = 1,
                            prediction_horizon: Union[int, float] = 1,
                            state_obs: Optional[list[str]] = None,
                            action_obs: Optional[list[str]] = None):
        """Generator that yields observation samples of length `n_frames_per_state` from the Markov Dynamics recordings.

        Args:
            recordings (list[DynamicsRecording]): List of DynamicsRecordings.
            frames_per_step: Number of frames to compose a single observation sample at time `t`. E.g. if `f` is provided
            the state samples will be of shape [f, obs_dim].
            prediction_horizon (int, float): Number of future time steps to include in the next time samples.
                E.g: if `n` is an integer the samples will be of shape [n, frames_per_state, obs_dim]
                If `n` is a float, then the samples will be of shape [int(n*traj_length), frames_per_state, obs_dim]
            state_obs: Ordered list of observations names composing the state space.
            action_obs: Ordered list of observations names composing the action space.

        Returns:
            A dictionary containing the observations of shape (time_horizon, frames_per_step, obs_dim)
        """
        for file_data in dynamics_recordings:
            recordings = file_data.recordings
            relevant_obs = set(file_data.state_obs).union(set(file_data.action_obs))
            if state_obs is not None:
                relevant_obs = set(state_obs)
            if action_obs is not None:
                relevant_obs = relevant_obs.union(set(action_obs))

            # Get any observation list of trajectories and count the number of trajectories
            # We assume all observations have the same number of trajectories
            n_trajs = len(next(iter(recordings.values())))

            # Since we assume trajectories can have different lengths, we iterate over each trajectory
            # and generate samples of length `n_frames_per_state` from each trajectory.
            for traj_id in range(n_trajs):
                traj_length = next(iter(recordings.values()))[traj_id].shape[0]
                if isinstance(prediction_horizon, float):
                    steps_in_pred_horizon = floor((prediction_horizon * traj_length) // frames_per_step) - 1
                else:
                    steps_in_pred_horizon = prediction_horizon
                assert steps_in_pred_horizon > 0, f"Invalid prediction horizon {steps_in_pred_horizon}"

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
                    for obs_name, trajs in recordings.items():
                        if obs_name not in relevant_obs:  # Do not process unrequested observations
                            continue
                        num_steps = (frames_in_pred_horizon // frames_per_step) + 1
                        # Compute the indices for the start and end of each "step" in the time horizon
                        start_indices = np.arange(0, num_steps) * frames_per_step + frame
                        end_indices = start_indices + frames_per_step
                        # Use these indices to slice the relevant portion of the trajectory
                        obs_time_horizon = trajs[traj_id][start_indices[0]:end_indices[-1]]
                        # Reshape the slice to have the desired shape (time, frames_per_step, obs_dim)
                        obs_dim = file_data.obs_dims[obs_name]
                        obs_time_horizon = obs_time_horizon.reshape((num_steps, frames_per_step, obs_dim))

                        # Test no copy is being made (too costly to do at runtime)
                        # assert np.shares_memory(obs_time_horizon, trajs[traj_id])
                        assert len(obs_time_horizon) == steps_in_pred_horizon + 1, \
                            f"{len(obs_time_horizon)} != {steps_in_pred_horizon + 1}"
                        sample[obs_name] = obs_time_horizon
                    # print(frame)
                    yield sample

    @staticmethod
    def map_state_next_state(sample: dict, state_observations: List[str]) -> dict:
        """Map composing multiple frames of observations into a flat vectors `state` and `next_state` samples.

        This method constructs the state `s_t` and history of nex steps `s_t+1` of the Markov Process.
        The state is defined as a set of observations within a window of fps=`frames_per_state`.
        E.g.: Consider the state is defined by the observations [m=momentum, p=position] at `fps` consecutive frames.
            Then the state at time `t` is defined as `s_t = [m_f, m_f+1,..., m_f+fps, p_f, p_f+1, ..., p_f+fps]`.
            Where we use f to denote frame in time to make the distinction from the time index `t` of the Markov Process.
            Then, the next state is defined as `s_t+1 = [m_f+fps,..., m_fps+fps, p_f+fps, ..., p_f+fps+fps]`.
        Args:
            sample (dict): Dictionary containing the observations of the system of shape [state_time, f].
            state_observations: Ordered list of observations names composing the state space.
        Returns:
            A dictionary containing the MDP state `s_t` and the next_state/s `[s_t+1, s_t+2, ..., s_t+pred_horizon]`.
        """
        batch_size = len(sample[f"{state_observations[0]}"])
        time_horizon = len(sample[f"{state_observations[0]}"][0])
        # Flatten observations a_t = [a_f, a_f+1, af+2, ..., a_f+F] s.t. a_t in R^{F * dim(a)}, a_f in R^{dim(a)}
        state_obs = [sample[m] for m in state_observations]
        # Define the state at time t and the states at time [t+1, t+pred_horizon]
        state_traj = np.concatenate(state_obs, axis=-1).reshape(batch_size, time_horizon, -1)
        return dict(state=state_traj[:, 0], next_state=state_traj[:, 1:])

    @staticmethod
    def map_state_action_state(sample, state_observations: List[str], action_observations: List[str]) -> dict:
        """Map composing multiple observations to single state, action, next_state samples."""
        flat_sample = DynamicsRecording.map_state_next_state(sample, state_observations)
        # Reuse the same function to for flattening action and next_action
        action_sample = DynamicsRecording.map_state_next_state(sample, action_observations)
        action_sample["action"] = action_sample.pop("state")
        action_sample["next_action"] = action_sample.pop("next_state")
        flat_sample.update(action_sample)
        return flat_sample



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
                         train_pred_horizon: Union[int, float] = 1,
                         eval_pred_horizon: Union[int, float] = 10,
                         state_obs: Optional[tuple[str]] = None,
                         action_obs: Optional[tuple[str]] = None
                         ) -> tuple[list[IterableDataset], DynamicsRecording]:
    """Load Markov Dynamics recordings from a list of files and return a train, test and validation dataset."""
    # TODO: ensure all shards come from the same dynamical system
    metadata: DynamicsRecording = DynamicsRecording.load_from_file(train_shards[0])
    test_metadata = DynamicsRecording.load_from_file(test_shards[0], only_metadata=True)

    dyn_params_diff = compare_dictionaries(metadata.dynamics_parameters, test_metadata.dynamics_parameters)
    assert len(dyn_params_diff) == 0, "Different dynamical systems loaded in train/test sets"

    state_obs = state_obs if state_obs is not None else metadata.state_obs
    action_obs = action_obs if action_obs is not None else metadata.action_obs
    relevant_obs = set(state_obs).union(set(action_obs))
    features = {}
    assert len(relevant_obs) > 0, f"Provide the names of the observations to be included in state (and action)"
    for obs_name in relevant_obs:
        assert obs_name in metadata.recordings.keys(), f"Observation {obs_name} not found in recordings"
    for obs_name in relevant_obs:
        features[obs_name] = datasets.Array2D(shape=(frames_per_step, metadata.obs_dims[obs_name]), dtype='float32')

    part_datasets = []
    for partition, partition_shards in zip(["train", "test", "val"], [train_shards, test_shards, val_shards]):
        recordings = [DynamicsRecording.load_from_file(f) for f in partition_shards]
        if partition == "train":
            pred_horizon = train_pred_horizon
        elif partition == "val":
            pred_horizon = eval_pred_horizon
        else:
            pred_horizon = eval_pred_horizon

        num_trajs, num_samples = estimate_dataset_size(recordings, pred_horizon, frames_per_step)
        dataset = IterableDataset.from_generator(DynamicsRecording.load_data_generator,
                                                 features=Features(features),
                                                 gen_kwargs=dict(dynamics_recordings=recordings,
                                                                 frames_per_step=frames_per_step,
                                                                 prediction_horizon=pred_horizon,
                                                                 state_obs=tuple(state_obs),
                                                                 action_obs=tuple(action_obs))
                                                 )

        # for sample in dataset:
        log.debug(f"[Dataset {partition} - Trajs:{num_trajs} - Samples: {num_samples} - "
                  f"Frames per sample : {frames_per_step}]-----------------------------")
            # log.debug(f"\tstate: {state.shape} = (frames_per_step, state_dim)")
            # log.debug(f"\tnext_state: {next_state.shape} = (pred_horizon, frames_per_step, state_dim)")
            # break

        dataset.info.dataset_size = num_samples
        dataset.info.dataset_name = f"[{partition}] Linear dynamics"
        dataset.info.description = metadata.description
        # dataset = partition
        part_datasets.append(dataset)

    metadata.state_obs = state_obs
    metadata.action_obs = action_obs
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

    # test_flat = map_state_next_state(sample, metadata.state_observations)
    # Test the map
    torch_dataset = test_dataset.with_format("torch").shuffle()
    torch_dataset = torch_dataset.map(map_state_next_state, batched=True, fn_kwargs={'state_observations': ['state']})

    # Test errors while looping.
    for i, s in enumerate(torch_dataset):
        pass
        # log.debug(s)
        # if i > 1000:
        #     break

    log.debug(i)
