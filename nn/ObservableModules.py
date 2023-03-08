from typing import Optional

import torch
from torch.nn import Module

from src.RobotEquivariantNN.groups.SparseRepresentation import SparseRep
from src.RobotEquivariantNN.nn.EMLP import EMLP, MLP
from src.RobotEquivariantNN.nn.EquivariantModules import EquivariantModel


class DynamicsAutoEncoder(Module):

    def __init__(self, in_dim: int, obs_dim: int, dt: float, num_hidden_layers=2, num_hidden_cells=32,
                 observation_module: Optional[Module] = None, activation: Module = torch.nn.Tanh, ):
        super().__init__()
        self.dt = dt

        self.encoder = MLP(d_in=in_dim, d_out=obs_dim, ch=num_hidden_cells, num_layers=num_hidden_layers,
                           activation=activation)
        self.decoder = MLP(d_in=obs_dim, d_out=in_dim, ch=num_hidden_cells, num_layers=num_hidden_layers,
                           activation=activation)

        self.observation_dynamics = LinearEigenvectorDynamics(dim=obs_dim)

    def forward(self, x):
        z = self.encoder(x)
        if self.observation_dynamics is None:
            z_pred = z
        else:
            dts = torch.arange(0, z.shape[1], device=x.device) * self.dt
            z_pred = self.observation_dynamics(z[:, 0, :], dts)
        x_pred = self.decoder(z_pred)
        return {"x_pred": x_pred, "z_pred": z_pred, "z": z}

    def get_hparams(self):
        return {'encoder': self.encoder.get_hparams(),
                'decoder': self.decoder.get_hparams(),
                'obs_dynamics': self.observation_dynamics.get_hparams()}


class EDynamicsAutoEncoder(EquivariantModel):

    def __init__(self, repX: SparseRep, obs_dim: int, dt: float, num_hidden_layers=2, num_hidden_cells=32,
                 observation_module: Optional[Module] = None, activation: Module = torch.nn.Tanh, ):
        super().__init__(rep_in=repX, rep_out=repX)
        self.dt = dt
        # TODO: Isotypal representation
        self.repZ = SparseRep(self.rep_in.G.canonical_group(obs_dim))

        self.encoder = EMLP(rep_in=self.rep_in, rep_out=self.repZ, ch=num_hidden_cells, num_layers=num_hidden_layers,
                            activation=activation)
        self.decoder = EMLP(rep_in=self.repZ, rep_out=self.rep_in, ch=num_hidden_cells, num_layers=num_hidden_layers,
                            activation=activation)
        self.observation_dynamics = LinearEigenvectorDynamics(dim=obs_dim)

        # EquivariantModel.test_module_equivariance(self, repX, repX)

    def forward(self, x):
        z = self.encoder(x)
        if self.observation_dynamics is None:
            z_pred = z
        else:
            dts = torch.arange(0, z.shape[1]) * self.dt
            z_pred = self.observation_dynamics(z[:, 0, :], dts)
        x_pred = self.decoder(z_pred)
        return {"x_pred": x_pred, "z_pred": z_pred, "z": z}

    def get_hparams(self):
        return {'encoder': self.encoder.get_hparams(),
                'decoder': self.decoder.get_hparams()}

class LinearEigenvectorDynamics(Module):

    def __init__(self, dim: int, state_is_real=True):
        super().__init__()
        assert dim % 2 == 0, "For now only cope with even dimensions."
        self.dim = dim
        eigvals = torch.view_as_complex(torch.randn(self.dim//2, 2))
        eigvals /= eigvals.abs()    # Initialize as stable system.
        self.eigvals = torch.nn.Parameter(eigvals, requires_grad=True)

    def forward(self, z, dt):
        assert torch.all(torch.isreal(z))
        z2d = z.view(z.shape[:-1] + (-1, 2))
        z_complex = torch.view_as_complex(z2d)
        eigvect_projection = self.interleave_with_conjugate(z_complex)
        eigvect_projection = torch.unsqueeze(eigvect_projection, 1)  # Add timestep dimension,

        matrix_eigvals = torch.unsqueeze(self.interleave_with_conjugate(self.eigvals), 0)

        # Matrix exponential of a diagonal matrix is the exponent of the diagonal elements.
        discrete_eigvals = torch.exp(torch.mul(torch.unsqueeze(dt, 1), matrix_eigvals))

        eigvect_projection_new = torch.mul(eigvect_projection, discrete_eigvals)
        assert torch.isclose(eigvect_projection_new[0, 0, 0], torch.conj(eigvect_projection_new[0, 0, 1])), "Complex error"
        z_complex_new = eigvect_projection_new[:, :, ::2]
        original_shape = (tuple(z_complex_new.shape[:2]) + (-1,))
        z_new = torch.reshape(torch.view_as_real(z_complex_new), original_shape)
        return z_new

    @staticmethod
    def interleave_with_conjugate(a: torch.Tensor):
        assert a.dtype == torch.cfloat or a.dtype == torch.cdouble
        new_shape = list(a.shape)
        if a.shape != 1:  # multi dimensional tensor
            d = a.shape[-1]
            new_shape[-1] = 2 * d
        else:
            d = 1
            new_shape = 2 * d

        a_conj_a = torch.concatenate([torch.unsqueeze(a, -1), torch.unsqueeze(torch.conj(a), -1)], dim=-1).view(new_shape)
        return a_conj_a

    def get_hparams(self):
        return {'num_complex_eigval': self.dim//2 }