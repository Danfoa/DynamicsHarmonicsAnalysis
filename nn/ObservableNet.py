import copy
from typing import Optional, Union

import escnn.nn
import torch.nn
from escnn.nn import EquivariantModule

from nn.EquivLinearDynamics import EquivLinearDynamics
from nn.LinearDynamics import LinearDynamics
from nn.mlp import MLP
from nn.emlp import EMLP


class ObservableNet(torch.nn.Module):

    def __init__(self, 
                 obs_fn: Union[torch.nn.Module, EquivariantModule],
                 obs_fn_aux: Optional[Union[torch.nn.Module, EquivariantModule]] = None):
        super().__init__()
        self.equivariant = isinstance(obs_fn, EquivariantModule)
        self.use_aux_obs_fn = obs_fn_aux is not None

        self.obs = obs_fn
        self.obs_aux = None
        if self.use_aux_obs_fn:  # Use two twin networks to compute the main and auxiliary observable space.
            self.obs_aux = obs_fn_aux
        else:
            if self.equivariant:
                self.transfer_op_H_H_prime = escnn.nn.Linear(
                    in_type=self.obs.out_type, out_type=self.obs.out_type, bias=False)
            else:
                self.transfer_op_H_H_prime = torch.nn.Linear(
                    in_features=self.obs.out_dim, out_features=self.obs.out_dim, bias=False)

    def forward(self, input):

        obs_state = self.obs(input)

        if self.use_aux_obs_fn:
            obs_aux_state = self.obs_aux(input)
        else:
            obs_aux_state = self.transfer_op_H_H_prime(obs_state)

        return obs_state, obs_aux_state


