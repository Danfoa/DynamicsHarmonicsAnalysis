import copy
from typing import Optional, Union

import escnn.nn
import torch.nn
from escnn.nn import EquivariantModule

class ObservableNet(torch.nn.Module):
    """
    A network computing the observation state in the initial observable space H with measure μ(t) and the observable
    space H' representing the observable space after a Δt step, which potentially has a different measure μ(t+Δt).
    """
    def __init__(self,
                 encoder: Union[torch.nn.Module, EquivariantModule],
                 aux_encoder: Optional[Union[torch.nn.Module, EquivariantModule]] = None,
                 obs_dim: Optional[int] = None,
                 obs_type: Optional[escnn.nn.FieldType] = None):
        super().__init__()
        self.equivariant = isinstance(encoder, EquivariantModule)
        self.use_aux_encoder = aux_encoder is not None

        self.encoder = encoder
        self.aux_encoder = aux_encoder if self.use_aux_encoder else None

        # Setting the bias of the linear layer to true is equivalent to setting the constant function in the basis
        # of the space of functions. Then the bias of each dimension is the coefficient of the constant function.
        if self.equivariant:
            # Bias term (a.k.a the constant function) is present only on the trivial isotypic subspace
            assert obs_type is not None, f"obs state Field type must be provided when using equivariant encoder"
            self.obs_H = escnn.nn.Linear(
                in_type=self.encoder.out_type, out_type=obs_type, bias=True)
            self.obs_H_prime = escnn.nn.Linear(
                in_type=self.encoder.out_type, out_type=obs_type, bias=True)
        else:
            assert obs_dim is not None, f"obs state dimension must be provided when using non-equivariant encoder"
            self.obs_H = torch.nn.Linear(
                in_features=self.encoder.out_dim, out_features=obs_dim, bias=True)
            self.obs_H_prime = torch.nn.Linear(
                in_features=self.encoder.out_dim, out_features=obs_dim, bias=True)

    def forward(self, input):

        features = self.encoder(input)
        aux_features = self.aux_encoder(input) if self.use_aux_encoder else features

        obs_state_H = self.obs_H(features)
        obs_state_H_prime = self.obs_H_prime(aux_features)

        return obs_state_H, obs_state_H_prime


