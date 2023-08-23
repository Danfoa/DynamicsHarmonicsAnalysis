import logging
import math
from collections import OrderedDict
from typing import Union

import escnn
import numpy as np
import torch
from escnn.group import Group, Representation, directsum
from escnn.nn import EquivariantModule, FieldType, GeometricTensor

from nn.LinearDynamics import EquivariantLinearDynamics
from nn.markov_dynamics import MarkovDynamicsModule
from nn.mlp import EMLP, MLP
from utils.representation_theory import identify_isotypic_spaces

log = logging.getLogger(__name__)


def isotypic_basis(group: Group, num_regular_fields: int, prefix=''):
    rep, _ = identify_isotypic_spaces(group.regular_representation)
    # Construct the obs state representation as a `num_regular_field` copies of the isotypic representation
    iso_reps = OrderedDict()
    for iso_irrep_id, reg_rep_iso in rep.attributes['isotypic_reps'].items():
        iso_reps[iso_irrep_id] = directsum([reg_rep_iso] * num_regular_fields,
                                           name=f"{prefix}_IsoSpace{iso_irrep_id}")
    return iso_reps


def compute_invariant_features(x: torch.Tensor, field_type: FieldType) -> torch.Tensor:
    n_inv_features = len(field_type.irreps)
    # TODO: Ensure isotypic basis.
    inv_features = []
    for field_start, field_end, rep in zip(field_type.fields_start, field_type.fields_end, field_type.representations):
        x_field = x[..., field_start:field_end]
        num_G_stable_spaces = len(rep.irreps)
        # irrep_dim = int((field_end - field_start) // num_G_stable_spaces)
        unique_irreps = np.unique(rep.irreps, axis=0)
        assert len(unique_irreps) == 1, "This only works for now on the Isotypic Basis"
        x_field_p = torch.reshape(x_field, (x_field.shape[0], num_G_stable_spaces, -1))
        # a = x_field[0, :irrep_dim]
        # b = x_field_p[0, 0, :]
        # n = torch.norm(a)
        inv_field_features = torch.norm(x_field_p, dim=-1)
        inv_features.append(inv_field_features)
    inv_features = torch.cat(inv_features, dim=-1)
    assert inv_features.shape[-1] == n_inv_features, f"Expected {n_inv_features} got {inv_features.shape[-1]}"
    return inv_features


class EquivDynamicsAutoEncoder(MarkovDynamicsModule):

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
                                                            trainable=False,  # Params will come from the eigval net
                                                            **kwargs)

        # The eigenvalue network needs to be a G-invariant network. For obtaining G-invariant features for this network
        # We can exploit the fact that in the isotypic basis every irreducible G-stable space can be used to obtain a
        # G-invariant feature: The norm of the state in that irreducible G-stable space is G-invariant.
        num_G_stable_spaces = sum([len(rep.irreps) for rep in obs_space_iso_reps.values()])
        num_degrees_of_freedom = self.obs_state_dynamics.num_parameters()
        self.eigval_net = MLP(in_dim=num_G_stable_spaces,
                              out_dim=num_degrees_of_freedom,
                              num_hidden_units=num_encoder_hidden_neurons,
                              num_layers=3,
                              activation=torch.nn.ReLU,
                              with_bias=True)

    def forcast(self, initial_state: GeometricTensor, n_steps: int = 1, **kwargs) -> [dict[str, GeometricTensor]]:
        batch_dim = initial_state.tensor.shape[0]
        obs_state = self.encoder(initial_state)

        inv_features = compute_invariant_features(obs_state.tensor, self.obs_state_type)
        lin_dynamics_parameters = self.eigval_net(inv_features)
        self.obs_state_dynamics._equiv_lin_layer.weights.data = lin_dynamics_parameters

        if self.obs_state_dynamics is None or n_steps < 1:
            obs_state_pred = obs_state
            state_pred = self.decoder(obs_state_pred)
        else:
            # Obtain predictions per step
            obs_state_pred = self.obs_state_dynamics(initial_state=obs_state, n_steps=n_steps)['state']
            # TODO: GeometricTensor is not defined to handle index dims such as the one needed for time index
            #  see github.com/QUVA-Lab/escnn/issues/75. Work around is to use the batch dimension as indexing dim
            # We create an enormous GeometricTensor with the obs_state type by appending in the batch dimension.
            obs_traj_tensor = torch.cat([obs_state.tensor for obs_state in obs_state_pred], dim=0)
            obs_state_traj = self.obs_state_type(obs_traj_tensor)

            state_pred_traj = self.decoder(obs_state_traj)
            # Split the enormous GeometricTensor into a list of GeometricTensors with the same batch dimension
            state_pred = [self.state_type(s) for s in torch.split(state_pred_traj.tensor, batch_dim, dim=0)]
            assert len(state_pred) == n_steps, f"Expected {n_steps} predictions, got {len(state_pred)}"

        return {"state": state_pred, "obs_state": obs_state_pred}

    def get_hparams(self):
        return {'encoder':      self.encoder.get_hparams(),
                'decoder':      self.decoder.get_hparams(),
                'obs_dynamics': self.obs_state_dynamics.get_hparams()}


if __name__ == "__main__":
    G = escnn.group.DihedralGroup(10)
    rep_state = G.regular_representation
    obs_state_dimension = rep_state.size * 3
    dt = 0.1
    num_encoder_layers = 4
    num_encoder_hidden_neurons = obs_state_dimension * 2
    activation = torch.nn.Tanh
    equivariant = True

    model = EquivDynamicsAutoEncoder(state_type=rep_state,
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

    n_steps = 500
    s_pred = model(s, n_steps=n_steps)
    g_s_pred = model(g_s, n_steps=n_steps)

    for measurement in ['state']:
        for i in range(n_steps):
            s_next = s_pred[measurement][i]
            g_s_next = g_s_pred[measurement][i]

            g_s_next_true = model.state_type.transform_fibers(s_next.tensor, g)
            g_s_next_pred = g_s_next.tensor

            absolute_error = torch.norm(g_s_next_true - g_s_next_pred) / torch.norm(g_s_next_true)
            assert torch.allclose(absolute_error, torch.zeros(1), rtol=1e-5,
                                  atol=1e-5), f"t={i} - {absolute_error * 100:.2f}% error"

    print(f"Done here it is your Equivariant Dynamics Autoencoder :)")
