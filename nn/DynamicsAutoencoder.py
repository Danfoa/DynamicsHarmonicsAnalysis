import logging
from collections import OrderedDict
from typing import Union

import escnn
import torch
from escnn.group import Representation, directsum
from escnn.nn import FieldType

from nn.linear_dynamics import EquivariantLinearDynamics
from nn.markov_dynamics import MarkovDynamicsModule
from nn.mlp import EMLP, MLP
from utils.representation_theory import identify_isotypic_spaces

log = logging.getLogger(__name__)


class DynamicsAutoEncoder(MarkovDynamicsModule):

    def __init__(self,
                 rep_state: Representation,
                 num_regular_field: int,
                 dt: float,
                 num_encoder_layers: int = 4,
                 num_encoder_hidden_neurons: int = 32,
                 activation: Union[torch.nn.Module, escnn.nn.EquivariantModule] = torch.nn.Tanh,
                 equivariant: bool = True,
                 eigval_network: bool = False,
                 **kwargs):

        super(DynamicsAutoEncoder, self).__init__(state_dim=rep_state.size, dt=dt, **kwargs)

        self.symm_group = rep_state.group
        if equivariant:
            # Apply isotypic decomposition to the regular representation
            rep, _ = identify_isotypic_spaces(self.symm_group.regular_representation)
            # Construct the obs state representation as a `num_regular_field` copies of the isotypic representation
            obs_state_iso_reps = OrderedDict()
            for iso_irrep_id, reg_rep_iso in rep.attributes['isotypic_reps'].items():
                obs_state_iso_reps[iso_irrep_id] = directsum([reg_rep_iso] * num_regular_field,
                                                             name=f"ObsState_IsoSpace{iso_irrep_id}")
            # Obtain instance of the representation of the observable space
            rep_obs_state = directsum(list(obs_state_iso_reps.values()), name="ObsState_IsotypicBasis")
            # Include the isotypic decomposition, detailing the supported non-linearity per isotypic subspace.
            rep_obs_state.attributes['isotypic_reps'] = obs_state_iso_reps
            assert rep_obs_state.size == self.symm_group.order() * num_regular_field  # Check rep dimensionality

            # Define the input and output field types for ESCNN.
            gspace = escnn.gspaces.no_base_space(self.symm_group)
            self.state_field_type = FieldType(gspace, [rep_state])
            self.obs_state_field_type = FieldType(gspace, [rep_iso for rep_iso in obs_state_iso_reps.values()])

            self.encoder = EMLP(in_type=self.state_field_type,
                                out_type=self.obs_state_field_type,
                                num_hidden_units=num_encoder_hidden_neurons,
                                num_layers=num_encoder_layers,
                                with_bias=True,
                                )

            self.decoder = EMLP(in_type=self.obs_state_field_type,
                                out_type=self.state_field_type,
                                num_hidden_units=num_encoder_hidden_neurons,
                                num_layers=num_encoder_layers,
                                with_bias=True)

            self.obs_state_dynamics = EquivariantLinearDynamics(in_type=self.obs_state_field_type,
                                                                dt=self.dt,
                                                                trainable=not eigval_network,
                                                                **kwargs)
            if eigval_network:
                # TODO: Generate invariant features to feed to the eigval network a normal MLP.
                raise NotImplementedError("Eigval network not implemented yet")

        else:
            # Compare the same number of equivariant and non-equivariant observable functions
            obs_space_dim = self.symm_group.order() * num_regular_field
            self.encoder = MLP(in_dim=self.state_dim,
                               out_dim=obs_space_dim,
                               num_hidden_units=num_encoder_hidden_neurons,
                               num_layers=num_encoder_layers,
                               with_bias=True,
                               activation=activation,
                               init_mode="fan_in")
            self.decoder = MLP(in_dim=obs_space_dim,
                               out_dim=self.state_dim,
                               num_hidden_units=num_encoder_hidden_neurons,
                               num_layers=num_encoder_layers,
                               with_bias=True,
                               activation=activation,
                               init_mode="fan_in")

            self.obs_state_dynamics = EquivariantLinearDynamics(state_dim=num_regular_field, **kwargs)

        print(self)
        print("Done")


    def forcast(self, initial_state):

        obs_state = self.encoder(initial_state)
        if self.obs_state_dynamics is None:
            obs_state_pred = obs_state
        else:
            dts = torch.arange(0, obs_state.shape[1], device=initial_state.device) * self.dt
            obs_state_pred = self.obs_state_dynamics(obs_state[:, 0, :], dts)
        state_pred = self.decoder(obs_state_pred)

        return {"state_pred": state_pred, "obs_state_pred": obs_state_pred, "obs_state": obs_state}

    def get_hparams(self):
        return {'encoder':      self.encoder.get_hparams(),
                'decoder':      self.decoder.get_hparams(),
                'obs_dynamics': self.obs_state_dynamics.get_hparams()}


if __name__ == "__main__":
    G = escnn.group.DihedralGroup(10)
    rep_state = G.regular_representation
    num_regular_field = 2
    dt = 0.1
    num_encoder_layers = 4
    num_encoder_hidden_neurons = 32
    activation = torch.nn.Tanh
    equivariant = True

    model = DynamicsAutoEncoder(rep_state=rep_state,
                                num_regular_field=num_regular_field,
                                dt=dt,
                                num_encoder_layers=num_encoder_layers,
                                num_encoder_hidden_neurons=num_encoder_hidden_neurons,
                                activation=escnn.nn.ReLU,
                                eigval_network=False,
                                equivariant=equivariant)
