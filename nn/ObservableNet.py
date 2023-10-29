import copy
from typing import Optional, Union

import escnn.nn
import torch.nn
from escnn.nn import EquivariantModule
from .EquivLinearDynamics import EquivLinearDynamics
from .LinearDynamics import LinearDynamics

class ObservableNet(torch.nn.Module):
    """
    A network computing the observation state in the initial observable space H with measure μ(t) and the observable
    space H' representing the observable space after a Δt step, which potentially has a different measure μ(t+Δt).
    """
    def __init__(self,
                 encoder: Union[torch.nn.Module, EquivariantModule],
                 obs_dim: Optional[int] = None,
                 obs_type: Optional[escnn.nn.FieldType] = None,
                 explicit_transfer_op=True):
        """ TODO

        Args:
            encoder:
            obs_dim:
            obs_type:
            explicit_transfer_op: H' = A H + b where A is a learnable linear map. If false H and H' are computed from
            the encoder features by two different linear maps.
        """
        super().__init__()
        self.equivariant = isinstance(encoder, EquivariantModule)
        self.explicit_transfer_op = explicit_transfer_op
        self.encoder = encoder

        # Setting the bias of the linear layer to true is equivalent to setting the constant function in the basis
        # of the space of functions. Then the bias of each dimension is the coefficient of the constant function.
        if self.equivariant:
            # Bias term (a.k.a the constant function) is present only on the trivial isotypic subspace
            assert obs_type is not None, f"obs state Field type must be provided when using equivariant encoder"
            self.obs_H = escnn.nn.Linear(
                in_type=self.encoder.out_type,
                out_type=obs_type,
                bias=True)
            if explicit_transfer_op:
                # EquivLinearDynamics handles initialization automatically for equivariant linear dynamics
                self.obs_H_prime = EquivLinearDynamics(state_type=obs_type, trainable=True, bias=True)
            else:
                self.obs_H_prime = escnn.nn.Linear(
                    in_type=self.encoder.out_type,
                    out_type=obs_type,
                    bias=True)
        else:
            assert obs_dim is not None, f"obs state dimension must be provided when using non-equivariant encoder"
            self.obs_H = torch.nn.Linear(
                in_features=self.encoder.out_dim,
                out_features=obs_dim,
                bias=True)
            if explicit_transfer_op:
                # Linear dynamics handles initialization automatically for linear dynamics
                self.obs_H_prime = LinearDynamics(state_dim=obs_dim, trainable=True, bias=True)
            else:
                self.obs_H_prime = torch.nn.Linear(
                    in_features=self.encoder.out_dim if not explicit_transfer_op else obs_dim,
                    out_features=obs_dim,
                    bias=True)

    def forward(self, input):

        features = self.encoder(input)

        obs_state_H = self.obs_H(features)
        if self.explicit_transfer_op:
            out_linear_dynamics = self.obs_H_prime(obs_state_H)
            obs_state_H_prime = out_linear_dynamics["pred_state_one_step"][:, -1, :]
            if self.equivariant:
                obs_state_H_prime = self.obs_H_prime.state_type(obs_state_H_prime)
        else:
            obs_state_H_prime = self.obs_H_prime(features)

        return obs_state_H, obs_state_H_prime


