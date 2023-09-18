import logging
from typing import Union

import escnn
import numpy as np
import torch
from escnn.nn import FieldType, GeometricTensor

from nn.LinearDynamics import EquivariantLinearDynamics
from nn.markov_dynamics import MarkovDynamicsModule
from nn.mlp import MLP
from nn.emlp import EMLP
from utils.representation_theory import isotypic_basis

log = logging.getLogger(__name__)


def compute_invariant_features(x: torch.Tensor, field_type: FieldType) -> torch.Tensor:
    n_inv_features = len(field_type.irreps)
    # TODO: Ensure isotypic basis i.e irreps of the same type are consecutive to each other.
    inv_features = []
    for field_start, field_end, rep in zip(field_type.fields_start, field_type.fields_end, field_type.representations):
        # Each field here represents a representation of an Isotypic Subspace. This rep is only composed of a single
        # irrep type.
        x_field = x[..., field_start:field_end]
        num_G_stable_spaces = len(rep.irreps)  # Number of G-invariant features = multiplicity of irrep
        # Again this assumes we are already in an Isotypic basis
        assert len(np.unique(rep.irreps, axis=0)) == 1, "This only works for now on the Isotypic Basis"
        # This basis is useful because we can apply the norm in a vectorized way
        # Reshape features to [batch, num_G_stable_spaces, num_features_per_G_stable_space]
        x_field_p = torch.reshape(x_field, (x_field.shape[0], num_G_stable_spaces, -1))
        # Compute G-invariant measures as the norm of the features in each G-stable space
        inv_field_features = torch.norm(x_field_p, dim=-1)
        # Append to the list of inv features
        inv_features.append(inv_field_features)
    # Concatenate all the invariant features
    inv_features = torch.cat(inv_features, dim=-1)
    assert inv_features.shape[-1] == n_inv_features, f"Expected {n_inv_features} got {inv_features.shape[-1]}"
    return inv_features


class EquivDynamicsAutoEncoder(MarkovDynamicsModule):
    TIME_DIM = 1

    def __init__(self,
                 state_type: FieldType,
                 obs_state_dimension: int,
                 dt: float,
                 num_encoder_layers: int = 4,
                 num_encoder_hidden_neurons: int = 128,
                 activation: Union[torch.nn.Module, escnn.nn.EquivariantModule] = torch.nn.Tanh,
                 **kwargs):

        super(EquivDynamicsAutoEncoder, self).__init__(state_dim=state_type.size, dt=dt, **kwargs)

        gspace = state_type.gspace
        self.symm_group = gspace.fibergroup
        # Number of regular fields in obs state and hidden layers of encoder/decoder
        num_regular_field = obs_state_dimension // state_type.size
        if num_regular_field < 1:
            raise ValueError(f"State-dim:{state_type.size}, |G|={self.symm_group.order()}, "
                             f"obs_dim:{obs_state_dimension}")

        # Define the input and output field types for ESCNN.
        self.state_type = state_type
        # Compute the observation space Isotypic Rep from the regular representation
        obs_space_iso_reps = isotypic_basis(self.symm_group, num_regular_field, prefix='ObsSpace')
        # Define the observation space in the ISOTYPIC BASIS!
        self.obs_state_type = FieldType(gspace, [rep_iso for rep_iso in obs_space_iso_reps.values()])

        # Define the encoder and decoder networks
        self.encoder = EMLP(in_type=self.state_type,
                            out_type=self.obs_state_type,
                            num_hidden_units=num_encoder_hidden_neurons,
                            num_layers=num_encoder_layers,
                            activation=activation,
                            with_bias=True)
        self.decoder = EMLP(in_type=self.obs_state_type,
                            out_type=self.state_type,
                            num_hidden_units=num_encoder_hidden_neurons,
                            num_layers=num_encoder_layers,
                            with_bias=True)
        # Define the linear dynamics module.
        self.obs_state_dynamics = EquivariantLinearDynamics(in_type=self.obs_state_type,
                                                            dt=self.dt,
                                                            trainable=True,  # Params will come from the eigval net
                                                            **kwargs)

        self.eigval_net = False  #
        if self.eigval_net:
            # The eigenvalue network needs to be a G-invariant network. For obtaining G-invariant features for this
            # network
            # We can exploit the fact that in the isotypic basis every irreducible G-stable space can be used to
            # obtain a
            # G-invariant feature: The norm of the state in that irreducible G-stable space is G-invariant.
            num_G_stable_spaces = sum([len(rep.irreps) for rep in obs_space_iso_reps.values()])
            num_degrees_of_freedom = self.obs_state_dynamics.equiv_lin_map.basisexpansion.dimension()
            self.eigval_net = MLP(in_dim=num_G_stable_spaces,
                                  out_dim=num_degrees_of_freedom,
                                  num_hidden_units=num_encoder_hidden_neurons,
                                  num_layers=3,
                                  activation=torch.nn.ReLU,
                                  with_bias=True)
            raise NotImplementedError("TODO: Need to implement this. "
                                      "There is no easy way to get batched parametrization of equivariant maps")

    def forcast(self, state: GeometricTensor, n_steps: int = 1, **kwargs) -> [dict[str, GeometricTensor]]:
        batch_dim = state.tensor.shape[0]

        obs_state = self.encoder(state)

        if self.eigval_net:
            inv_features = compute_invariant_features(obs_state.tensor, self.obs_state_type)
            # Compute G-invariant estimation of the basis coefficients of the equivariant linear map determining the
            # linear dynamics of the observable state.
            coefficients = self.eigval_net(inv_features)
            # TODO: Handle the basis expansion

        if self.obs_state_dynamics is None or n_steps < 1:
            obs_state_traj = obs_state
            state_pred = self.decoder(obs_state_traj)
        else:
            # Obtain predictions per step [batch, time, obs_state_dim] torch.Tensor
            obs_state_traj = self.obs_state_dynamics(state=obs_state, n_steps=n_steps)['next_state']

            # We create an enormous GeometricTensor with the obs_state type by appending in the batch dimension.
            obs_traj_tensor = torch.reshape(obs_state_traj, (-1, obs_state_traj.shape[-1]))
            # Decode the predicted obs_state trajectory into a state trajectory
            state_pred_traj = self.decoder(self.obs_state_type(obs_traj_tensor))

            # Reshape the state trajectory into [batch, time, state_dim]
            state_pred = torch.reshape(state_pred_traj.tensor, (batch_dim, n_steps, -1))
            assert state_pred.shape[1] == n_steps, f"Expected {n_steps} predictions, got {len(state_pred)}"

        pred = dict(next_state=state_pred, obs_state=obs_state, next_obs_state=obs_state_traj)
        if 'next_state' in kwargs:  # The ground truth next state need to be converted to observation ground truth
            gt_obs_state_traj = self.encoder(kwargs['next_state'])
            pred['gt_next_obs_state'] = torch.reshape(gt_obs_state_traj.tensor, obs_state_traj.shape)

        return pred

    def pre_process_state(self, state: torch.Tensor, **kwargs) -> Union[GeometricTensor, dict[str, GeometricTensor]]:
        input = dict(state=state, **kwargs)
        for measurement, value in input.items():
            if isinstance(value, torch.Tensor):
                if len(value.shape) == 2:
                    input[measurement] = self.state_type(state)
                elif len(value.shape) == 3:  # [batch, time, state_dim] -> [batch * time, state_dim]
                    geom_value = torch.reshape(value, (-1, value.shape[-1]))
                    input[measurement] = self.state_type(geom_value)
            elif isinstance(state, GeometricTensor):
                pass
            else:
                raise NotImplementedError("")
        return input

    def post_process_pred(self, predictions: dict) -> dict[str, torch.Tensor]:
        for measurement, value in predictions.items():
            if isinstance(value, list) and isinstance(value[0], GeometricTensor):
                predictions[measurement] = torch.stack([v.tensor for v in value], dim=self.TIME_DIM)
            elif isinstance(value, GeometricTensor):
                predictions[measurement] = value.tensor
        return predictions

    def get_hparams(self):
        return {'encoder':      self.encoder.get_hparams(),
                'decoder':      self.decoder.get_hparams(),
                'obs_dynamics': self.obs_state_dynamics.get_hparams()}


if __name__ == "__main__":
    G = escnn.group.DihedralGroup(3)
    rep_state = G.regular_representation
    obs_state_dimension = rep_state.size * 3
    gspace = escnn.gspaces.no_base_space(G)

    state_type = FieldType(gspace, [rep_state])

    dt = 0.1
    num_encoder_layers = 4
    num_encoder_hidden_neurons = obs_state_dimension * 2
    activation = torch.nn.Tanh
    equivariant = True

    model = EquivDynamicsAutoEncoder(state_type=state_type,
                                     obs_state_dimension=obs_state_dimension,
                                     num_encoder_layers=num_encoder_layers,
                                     num_encoder_hidden_neurons=num_encoder_hidden_neurons,
                                     activation=escnn.nn.ReLU,
                                     eigval_network=False,
                                     equivariant=equivariant,
                                     dt=dt)
    model.eval()
    g = G.sample()

    s = model.state_type(torch.randn(256, rep_state.size))
    g_s = model.state_type.transform_fibers(s.tensor, g)
    g_s = model.state_type(g_s)

    n_steps = 2
    s_pred = model(s.tensor, n_steps=n_steps)
    g_s_pred = model(g_s.tensor, n_steps=n_steps)

    for measurement in ['next_state']:
        for i in range(n_steps):
            s_next = s_pred[measurement][i]
            g_s_next = g_s_pred[measurement][i]

            for s, g_s in zip(s_next, g_s_next):
                rep_g = torch.Tensor(model.state_type.representation(g), device=s.device)
                g_s_next_true = rep_g @ s
                g_s_next_pred = g_s
                absolute_error = torch.norm(g_s_next_true - g_s_next_pred) / torch.norm(g_s_next_true)
                assert torch.allclose(absolute_error, torch.zeros(1), rtol=1e-5,
                                      atol=1e-5), f"t={i} - {absolute_error * 100:.2f}% error"

    print(f"Done here it is your Equivariant Dynamics Autoencoder :)")
