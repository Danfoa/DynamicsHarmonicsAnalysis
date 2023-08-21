import logging
from collections import OrderedDict
from typing import Optional

import escnn
import numpy as np
import torch
from escnn.group import Representation
from escnn.nn import FieldType
from torch.nn import Module

from nn.markov_dynamics import MarkovDynamicsModule
from utils.representation_theory import identify_isotypic_spaces

log = logging.getLogger(__name__)


class EquivariantLinearDynamics(MarkovDynamicsModule):

    def __init__(self,
                 in_type: FieldType,
                 dt: float,
                 trainable=True,
                 eigval_init="stable",
                 **kwargs):
        super().__init__(state_dim=in_type.size, dt=dt, **kwargs)
        rep = in_type.representation
        self.is_trainable = trainable

        self.symm_group = rep.group

        # Assert that the input field type is already in a symmetry enabled basis / isotypic basis.
        for rep in in_type.representations:
            assert len(np.unique(rep.irreps, axis=0)) == 1, f"Field rep:{rep} is not a rep of an isotypic subspace"

        # Use ESCNN to create a G-equivariant linear layer.
        # TODO: Modify initialization to account not for information flow but rather for different dynamical properties
        #  such as stability, etc.
        self._equiv_lin_layer = escnn.nn.Linear(in_type=in_type,
                                                out_type=in_type,
                                                bias=False,
                                                basisexpansion='blocks',  # TODO: Improve basis expansion
                                                initialize=True,          # TODO: Modify initialization
                                                )
        # Initialize eigenvalues.
        # TODO: Init

    def forcast(self, initial_state: torch.Tensor, n_steps: int = 1):

        state_trajectory = torch.unsqueeze(initial_state, 1)  # Add time dimension

        # Evolve dynamics
        for step in range(n_steps):
            # TODO: Push for numerical efficiency doing block-diagonal matrix multiplication
            state_pred = self._equiv_lin_layer(state_trajectory[:, step, :])
            # Append pred state to state trajectory
            state_trajectory = torch.cat((state_trajectory, state_pred), dim=1)

        # If required prediction is only a single step, return only that step.
        if state_trajectory.shape[1] == 1:
            state_trajectory = state_trajectory.squeeze(1)

        return state_trajectory

    def pre_process_state(self, state: torch.Tensor) -> torch.Tensor:
        if not self.rep_trivial_change_of_basis:
            # Change basis to the isotypic decomposition basis
            state_iso = torch.matmul(self.Q_iso, state)
            return state_iso
        return state

    def post_process_state(self, state: torch.Tensor) -> torch.Tensor:
        if not self.rep_trivial_change_of_basis:
            # Change basis back to the original basis
            state_orig = torch.matmul(self.Q_iso.T, state)
            return state_orig
        return state

    def get_hparams(self):
        return {'state_dim': self.state_dim,
                'n_isotypic_spaces': len(self.isotypic_subspaces_reps),
                'isotypic_spaces_dim': [rep.size for rep in self.iso_reps.values()],
                'is_state_in_isotypic_basis': self.rep_trivial_change_of_basis,
                }

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class LinearDynamics(Module):

    def __init__(self, dim: int, trainable=True, eigval_init="stable", eigval_constraint="unconstrained", **kwargs):
        super().__init__()
        assert dim % 2 == 0, "For now only cope with even dimensions."
        self.dim = dim
        self.trainable = trainable
        self._eigval_init = eigval_init
        self._eigval_constraint = eigval_constraint

        # Create the parameters determining the learnable eigenvalues of the eigenmatrix.
        # Each eigval is parameterized as re^(iw)
        w = torch.rand(self.dim)
        r = torch.rand(self.dim)
        if eigval_constraint == "unconstrained":
            self.w = torch.nn.Parameter(w, requires_grad=self.trainable)
            self.r = torch.nn.Parameter(r, requires_grad=self.trainable)
        elif eigval_constraint == "unit_circle":
            self.w = torch.nn.Parameter(w, requires_grad=self.trainable)
            self.r = torch.nn.Parameter(r * 0 + 1., requires_grad=False)
        else:
            raise NotImplementedError(f"Eigval constraint {self._eigval_constraint} not implemented")

        # Initialize eigenvalues.
        self.reset_parameters(init_mode=eigval_init)

    def forcast(self, z0, dt, obs_in_eigenbasis=True):
        assert torch.is_complex(z0), "We assume complex observations."
        assert z0.ndim == 2, "Expected (batch_size, dim(z))"
        z = torch.unsqueeze(z0, 1)  # Add time dimension

        # If the observations are not already in the eigenbasis we can use the latest eigendecomposition of the Koopman
        # operator K=VΛV^-1, we can transform the observations to the eigenbasis `z_eigen` = V^-1·z.
        z_eigen = z if obs_in_eigenbasis else self.obs2eigenbasis(z)

        # In `z_eigen`, Each obs is assumed to be an eigenfunction, forced to experience linear/eigenvalue dynamics.
        # Forcasting is done by powers of the diagonal of the Eigenmatrix
        matrix_eigvals = torch.unsqueeze(self.eigvals, 0)
        # Matrix exponential of a diagonal matrix is the exponent of the diagonal elements.
        discrete_eigvals = torch.exp(torch.mul(torch.unsqueeze(dt, 1), matrix_eigvals))

        # Evolve the eigenfunctions z_t+dt = K·z_t
        z_pred_eigen = torch.mul(z_eigen, discrete_eigvals)
        # Handle change of basis from eigenbasis if required
        z_pred = z_pred_eigen if obs_in_eigenbasis else self.eigenbasis2obs(z_pred_eigen)
        # Reshape to original shape is required because of how view_as_real works.
        # original_shape = (tuple(z_pred_cplx.shape[:2]) + (-1,))
        # z_pred_real = torch.reshape(torch.view_as_real(z_pred_cplx), original_shape)
        if not obs_in_eigenbasis:
            return z_pred, z_pred_eigen
        return z_pred

    @property
    def eigvals(self) -> torch.Tensor:
        re_eig = self.r * torch.cos(self.w)
        img_eig = self.r * torch.sin(self.w)
        eigvals = torch.view_as_complex(torch.stack((re_eig, img_eig), dim=1))
        return eigvals

    def update_eigvals_and_eigvects(self, eigvals: torch.Tensor, eigvects: Optional[torch.Tensor] = None):
        assert eigvals.shape[0] == self.dim
        assert eigvects is None or eigvects.shape == (self.dim, self.dim)
        device = self.r.device
        self.r.data = torch.abs(eigvals).to(device)
        self.w.data = torch.angle(eigvals).to(device)
        self.V = eigvects.to(device)
        self.V_inv = torch.linalg.inv(eigvects).to(device)
        log.debug("Updated eigenvalues of the eigenmatrix")

    def obs2eigenbasis(self, obs: torch.Tensor) -> torch.Tensor:
        """
        This function takes an observation z and transforms it to a basis defined by the Koopman eigenvectors
        z_eigen = V^-1·z. We follow the convention `Y = K X` being Y:(M, D) X:(N, D) data matrices with D samples, and
        K:(M, N) the Koopman operator, which is assumed to be decomposed into K=VΛV^-1.
        :param obs: torch.Tensor of shape (batch_size, time, dim(z))
        :return: a tensor of same shape as `obs` with observations in eigen basis coordinates.
        """
        assert obs.ndim == 3, "Expected (batch_size, time, dim(z))"
        if not hasattr(self, "V_inv"):
            raise RuntimeError("Koopman operator data has not been provided. Call `update_eigvals_and_eigvects` first")
        obs_eig = torch.matmul(self.V_inv.to(obs.device), torch.transpose(obs, dim1=2, dim0=1))
        obs_eig = torch.transpose(obs_eig, dim1=2, dim0=1)
        return obs_eig

    def eigenbasis2obs(self, obs_eigen: torch.Tensor) -> torch.Tensor:
        """
        This function takes an observation z_eigen in the basis defined by the Koopman eigenvectors and returns the
        observation transformed on the original basis coordinates with which K was approximated: z = V·z_eigen.
        We follow the convention `Y = K X` being Y:(M, D) X:(N, D) data matrices with D samples, and
        K:(M, N) the Koopman operator, which is assumed to be decomposed into K=VΛV^-1.
        :param obs: torch.Tensor of shape (batch_size, time, dim(z))
        :return: a tensor of same shape as `obs` with observations in eigen basis coordinates.
        """
        assert obs_eigen.ndim == 3, "Expected (batch_size, time, dim(z))"
        if not hasattr(self, "V"):
            raise RuntimeError("Koopman operator data has not been provided. Call `update_eigvals_and_eigvects` first")
        obs = torch.matmul(self.V.to(obs_eigen.device), torch.transpose(obs_eigen, dim1=2, dim0=1))
        obs = torch.transpose(obs, dim1=2, dim0=1)
        return obs

    def reset_parameters(self, init_mode: str):
        self._eigval_init = init_mode
        if init_mode == "stable":
            torch.nn.init.uniform_(self.w, 0, 2 * torch.pi)
            torch.nn.init.ones_(self.r)
            eigvals = self.eigvals
            # Check stability
            assert torch.allclose(self.r, torch.ones_like(self.r)), f"Not stable eigenvalues"
        else:
            raise NotImplementedError(f"Eival init mode {init_mode} not implemented")
        log.info(f"Eigenvalues initialization to {init_mode}")

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
        a_conj_a = torch.concatenate([torch.unsqueeze(a, -1), torch.unsqueeze(torch.conj(a), -1)], dim=-1).view(
            new_shape)
        return a_conj_a

    def get_hparams(self):
        return {'n_cplx_eigval': self.dim // 2,
                'eigval_init': self._eigval_init,
                'eigval_constraint': self._eigval_constraint}

    def extra_repr(self):
        return f"EigMatrix: n_cplx_eigval:{self.dim//2}" + \
               "on unit circle" if self._eigval_constraint == "unit_circle" else "" + \
                                                                                 f" - init: {self._eigval_init}"
