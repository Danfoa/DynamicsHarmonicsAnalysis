from collections import OrderedDict
from typing import Optional

import escnn
import numpy as np
import torch
from escnn.group import Representation
from escnn.nn import FieldType
from torch.nn import Module

import logging

from utils.representation_theory import identify_isotypic_spaces

log = logging.getLogger(__name__)


class LinearDynamics(Module):

    def __init__(self, rep: Optional[Representation] = None, state_dim: int = -1,
                 trainable=True, eigval_init="stable", eigval_constraint="unconstrained", **kwargs):
        assert rep is not None or state_dim > 0, "Either provide a representation or a state dimension"
        assert state_dim % 2 == 0, "For now only cope with even dimensions."

        super().__init__()

        self.state_dim = state_dim if rep is None else rep.size
        self.is_trainable = trainable

        # Extract the Isotypic Decomposition information from the state representation.
        if rep is not None:
            self.G = rep.group
            if 'isotypic_reps' in rep.attributes:
                self.Q_iso = rep.change_of_basis
                self.rep = Representation(group=rep.group, irreps=rep.irreps, name=f"{rep.name}-iso",
                                          change_of_basis=np.eye(self.state_dim))
            else:
                self.rep, self.Q_iso = identify_isotypic_spaces(rep)
            # Check if features already come in an isotypic basis. If so avoid changing basis at each forward pass.
            self.is_state_in_isotypic_basis = np.allclose(self.Q_iso, np.eye(self.state_dim))
            # Dictionary mapping active irreps with their corresponding Isotypic Subspace group representation
            self.iso_reps = self.rep.attributes['isotypic_reps']
            assert isinstance(self.iso_reps, OrderedDict), "We rely on ordered dict"
            # Ordered list of active irreps
            self.iso_irreps = [self.G.irrep(*irrep_id) for irrep_id in self.iso_reps.keys()]
            # Define the input and output field types for ESCNN.
            gspace = escnn.gspaces.no_base_space(self.G)
            self.in_field_type = FieldType(gspace, self.iso_reps.values())
            self.out_field_type = FieldType(gspace, self.iso_reps.values())

        number_of_params = self.calc_num_trainable_params() if self.is_trainable else 0

        self._eigval_init = eigval_init
        self._eigval_constraint = eigval_constraint


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

    def calc_num_trainable_params(self):
        if self.is_equivariant:
            # Compute the number of trainable parameters per Isotypic Subspace. This number will depend on the type
            # of irrep associated with the space.
            for iso_re_irrep, iso_rep in zip(self.iso_irreps, self.iso_reps.values()):
                n_G_irred_spaces = iso_rep.size // iso_re_irrep.size  # Number of G-irreducible spaces in Iso subspace.
                basis = iso_re_irrep.endomorphism_basis()
                basis_dim = len(basis)

                n_params_per_irred_space = None
                if iso_re_irrep.type == "R":  # Only Isomorphism are scalar multiple of the identity
                    n_params_per_irred_space = 1
                elif iso_re_irrep.type == "C":  # Realification: [re(eig1), im(eig1), ...]
                    n_params_per_irred_space = 2
                elif iso_re_irrep.type == "H":  # Realification: [re(eig1), im_i(eig1), im_j(eig1), im_k(eig1),...]
                    n_params_per_irred_space = 4
                else:
                    raise NotImplementedError(f"What is this representation type:{type}? Dunno.")


        else:
            raise NotImplementedError("TODO: Implement this")
            # Create the parameters determining the learnable eigenvalues of the eigenmatrix.
            # Each eigval is parameterized as re^(iw)
            w = torch.rand(self.state_dim)
            r = torch.rand(self.state_dim)
            if eigval_constraint == "unconstrained":
                self.w = torch.nn.Parameter(w, requires_grad=self.is_trainable)
                self.r = torch.nn.Parameter(r, requires_grad=self.is_trainable)
            elif eigval_constraint == "unit_circle":
                self.w = torch.nn.Parameter(w, requires_grad=self.is_trainable)
                self.r = torch.nn.Parameter(r * 0 + 1., requires_grad=False)
            else:
                raise NotImplementedError(f"Eigval constraint {self._eigval_constraint} not implemented")

    @property
    def eigvals(self) -> torch.Tensor:
        re_eig = self.r * torch.cos(self.w)
        img_eig = self.r * torch.sin(self.w)
        eigvals = torch.view_as_complex(torch.stack((re_eig, img_eig), dim=1))
        return eigvals

    def update_eigvals_and_eigvects(self, eigvals: torch.Tensor, eigvects: Optional[torch.Tensor] = None):
        assert eigvals.shape[0] == self.state_dim
        assert eigvects is None or eigvects.shape == (self.state_dim, self.state_dim)
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
        obs_eig = self.V_inv @ torch.transpose(obs, dim1=2, dim0=1)
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
        obs = self.V.to(obs_eigen.device) @ torch.transpose(obs_eigen, dim1=2, dim0=1)
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

    @property
    def is_equivariant(self):
        return hasattr(self, 'rep') and self.rep is not None

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
        return {'n_cplx_eigval':     self.state_dim // 2,
                'eigval_init': self._eigval_init,
                'eigval_constraint': self._eigval_constraint}

    def extra_repr(self):
        return f"EigMatrix: n_cplx_eigval:{self.state_dim // 2}" + \
               "on unit circle" if self._eigval_constraint == "unit_circle" else "" + \
               f" - init: {self._eigval_init}"
