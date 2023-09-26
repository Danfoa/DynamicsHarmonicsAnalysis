import itertools
import logging
import random
from collections import OrderedDict
from typing import Optional, Union

import escnn
import numpy as np
import scipy
import torch
from escnn.group import Representation
from escnn.nn import FieldType, GeometricTensor
from escnn.nn.modules.basismanager import BlocksBasisExpansion
from torch import Tensor

from nn.LinearDynamics import DmdSolver, LinearDynamics
from nn.markov_dynamics import MarkovDynamics
from utils.linear_algebra import full_rank_lstsq, full_rank_lstsq_symmetric, represent_linear_map_in_basis
from utils.representation_theory import isotypic_basis

log = logging.getLogger(__name__)

class EquivLinearDynamics(LinearDynamics):

    def __init__(self,
                 state_type: FieldType,
                 dmd_algorithm: Optional[DmdSolver] = None,
                 dt: Optional[Union[float, int]] = 1,
                 trainable=False,
                 bias: bool = True,
                 init_mode: str = "identity",
                 group_avg_trick: bool = True):

        self.symm_group = state_type.fibergroup
        self.gspace = state_type.gspace
        self.state_rep: Representation = state_type.representation
        self.group_avg_trick = group_avg_trick
        dmd_algorithm = dmd_algorithm if dmd_algorithm is not None else full_rank_lstsq_symmetric
        # Find the Isotypic basis of the state space
        self.state_iso_reps, self.state_iso_dims, Q_iso2state = isotypic_basis(representation=self.state_rep,
                                                                               multiplicity=1,
                                                                               prefix='ELDstate')

        ordered_irreps = [id for iso_rep in self.state_iso_reps.values() for id in iso_rep.irreps]
        self.state_type = state_type
        if ordered_irreps == state_type.irreps and np.allclose(self.state_rep.change_of_basis, np.eye(self.state_rep.size)):
            self.state_type_iso = state_type
            Q_iso2state = Q_state2iso = None
        else:
            self.state_type_iso = FieldType(self.gspace, [rep_iso for rep_iso in self.state_iso_reps.values()])
            # Change of coordinates required for state to be in Isotypic basis.
            Q_iso2state = Tensor(Q_iso2state)
            Q_state2iso = Tensor(np.linalg.inv(Q_iso2state))

        super(EquivLinearDynamics, self).__init__(state_dim=state_type.size,
                                                  state_rep=state_type.representation,
                                                  dt=dt,
                                                  trainable=trainable,
                                                  bias=bias,
                                                  init_mode=init_mode,
                                                  dmd_algorithm=dmd_algorithm,
                                                  state_change_of_basis=Q_state2iso,
                                                  state_inv_change_of_basis=Q_iso2state)

        # self.compute_endomorphism_basis()
        self.iso_transfer_op = OrderedDict()
        self.iso_transfer_op_bias = OrderedDict()
        for irrep_id in self.state_iso_reps:  # Preserve the order of the Isotypic Subspaces
            self.iso_transfer_op[irrep_id] = None
            self.iso_transfer_op_bias[irrep_id] = None

    def forcast(self, state: GeometricTensor, n_steps: int = 1, **kwargs) -> Tensor:
        """ Predict the next `n_steps` states of the system.
        Args:
            state: (batch, state_dim) Initial state of the system.
            n_steps: (int) Number of steps to predict.
        Returns:
            pred_state_traj: (batch, n_steps + 1, state_dim)
        """
        batch, state_dim = state.tensor.shape
        assert state.type == self.state_type_iso, f"{state.type} != {self.state_type_iso}"

        # Use the transfer operator to compute the maximum likelihood prediction of the future trajectory
        pred_state_traj = [state]
        for step in range(n_steps):
            # Compute the next state prediction s_t+1 = K @ s_t
            current_state = pred_state_traj[-1]
            if self.is_trainable:
                next_obs_state = self.transfer_op(current_state)
            else:
                transfer_op, bias = self.get_transfer_op()
                if bias is not None:
                    next_obs_state = self.state_type_iso((transfer_op @ current_state.tensor.T + bias).T)
                else:
                    next_obs_state = self.state_type_iso((transfer_op @ current_state.tensor.T).T)
            pred_state_traj.append(next_obs_state)

        pred_state_traj = torch.stack([gt.tensor for gt in pred_state_traj], dim=1)

        assert pred_state_traj.shape == (batch, n_steps + 1, state_dim)
        return pred_state_traj

    def pre_process_state(self, state: Tensor, next_state: Optional[Tensor] = None) -> GeometricTensor:
        # Change basis to Isotypic basis.
        state_trajectory_iso_basis = super().pre_process_state(state=state, next_state=next_state)
        # Convert to Geometric Tensor
        return self.state_type_iso(state_trajectory_iso_basis)

    def post_process_state(self, state_traj: Union[GeometricTensor]) -> Tensor:
        # Change back from Isotypic basis to original basis. Return Tensor instead of Geometric Tensor
        state_traj_input_basis = super().post_process_state(
            state_traj=state_traj.tensor if isinstance(state_traj, GeometricTensor) else state_traj)
        return state_traj_input_basis

    def update_transfer_op(self, X: Tensor, X_prime: Tensor, group_avg_trick: bool = True):
        """ Use a DMD algorithm to update the empirical transfer operator
        Args:
            X: (state_dim, n_samples) Data matrix of states at time `t`.
            X_prime: (state_dim, n_samples) Data matrix of the states at time `t + dt`.
            group_avg_trick: (bool) Whether to use the group average trick to enforce equivariance.
        """
        if self.is_trainable:
            raise RuntimeError("This model was initialized as trainable")
        assert X.shape == X_prime.shape, f"X: {X.shape}, X_prime: {X_prime.shape}"
        assert X.shape[0] == self.state_dim, f"Invalid state dimension {X.shape[0]} != {self.state_dim}"

        state, next_state = X.T, X_prime.T
        iso_rec_error = []
        # For each Isotypic Subspace, compute the empirical transfer operator.
        for irrep_id, iso_rep in self.state_iso_reps.items():
            rep = iso_rep if irrep_id != self.symm_group.identity else None  # Check for Trivial Subspace

            # IsoSpace
            # Get the projection of the state onto the isotypic subspace
            state_iso = state[..., self.state_iso_dims[irrep_id]]
            next_state_iso = next_state[..., self.state_iso_dims[irrep_id]]

            # Generate the data matrices of x(w_t) and x(w_t+1)
            X_iso = state_iso.T             # (iso_state_dim, num_samples)
            X_iso_prime = next_state_iso.T  # (iso_state_dim, num_samples)

            # Compute the empirical transfer operator of this Observable Isotypic subspace
            A_iso, B_iso = self.dmd_algorithm(X_iso, X_iso_prime, bias=self.bias,
                                              rep_X=rep if self.group_avg_trick else None,
                                              rep_Y=rep if self.group_avg_trick else None)
            if self.bias:
                rec_error = torch.nn.functional.mse_loss(A_iso @ X_iso + B_iso, X_iso_prime)
            else:
                rec_error = torch.nn.functional.mse_loss(A_iso @ X_iso, X_iso_prime)

            iso_rec_error.append(rec_error)
            self.iso_transfer_op[irrep_id] = A_iso
            self.iso_transfer_op_bias[irrep_id] = B_iso

        transfer_op = torch.block_diag(*[self.iso_transfer_op[irrep_id] for irrep_id in self.state_iso_reps.keys()])
        assert transfer_op.shape == (self.state_dim, self.state_dim)
        self.transfer_op = transfer_op
        if self.bias:
            transfer_op_bias = torch.cat(
                [self.iso_transfer_op_bias[irrep_id] for irrep_id in self.state_iso_reps.keys()])
            assert transfer_op_bias.shape == (self.state_dim, 1), f"{transfer_op_bias.shape}!=({self.state_dim}, 1)"
            self.transfer_op_bias = transfer_op_bias

        iso_rec_error = Tensor(iso_rec_error)
        rec_error = torch.sum(iso_rec_error)
        self.transfer_op = transfer_op

        return dict(solution_op_rank=torch.linalg.matrix_rank(transfer_op.detach()).to(torch.float),
                    solution_op_cond_num=torch.linalg.cond(transfer_op.detach()).to(torch.float),
                    solution_op_error=rec_error.detach().to(torch.float),
                    solution_op_error_dist=iso_rec_error.detach().to(torch.float))

    def build_linear_map(self) -> escnn.nn.Linear:
        return escnn.nn.Linear(in_type=self.state_type_iso, out_type=self.state_type_iso, bias=self.bias)

    def compute_endomorphism_basis(self):
        # When approximating the transfer/Koopman operator from the symmetric observable space, we know the operator
        # belongs to the space of G-equivariant operators (Group Endomorphism of the observable space).
        # Using ESCNN we can compute the basis of the Endomorphism space, and use this basis to compute an empirical
        # G-equivariant approximation of the transfer operator. Since the observable space is defined in the Isotypic
        # basis, the operator is block-diagonal and the basis of the Endomorphism space is the block diagonal sum of
        # the basis of the Endomorphism space of each Isotypic subspace.
        # You can project operators to this basis space, or you can generate an operator by assigning a coefficient to
        # each basis element and summing them up. :)

        # self.iso_space_basis_mask = {}  # Used to mask empirical covariance between orthogonal observables
        for irrep_id, rep in self.state_iso_reps.items():
            iso_basis = BlocksBasisExpansion(in_reprs=[self.symm_group.irrep(*id) for id in rep.irreps],
                                             out_reprs=[self.symm_group.irrep(*id) for id in rep.irreps],
                                             basis_generator=self.gspace.build_fiber_intertwiner_basis,
                                             points=np.zeros((1, 1)))
            basis_coefficients = torch.rand((iso_basis.dimension(),)) + 2
            # mask = torch.logical_not(torch.isclose(non_zero_elements, torch.zeros_like(non_zero_elements), atol=1e-6))
            # self.iso_space_basis_mask[irrep_id] = mask
        raise NotImplementedError()

    def reset_parameters(self, init_mode: str):
        basis_expansion = self.transfer_op.basisexpansion
        identity_coefficients = torch.zeros((basis_expansion.dimension(),))
        if init_mode == "identity":
            # Incredibly shady hack in order to get the identity matrix as the initial transfer operator.
            for io_pair in basis_expansion._representations_pairs:
                # retrieve the basis
                block_expansion = getattr(basis_expansion, f"block_expansion_{basis_expansion._escape_pair(io_pair)}")
                # retrieve the indices
                start_coeff = basis_expansion._weights_ranges[io_pair][0]
                end_coeff = basis_expansion._weights_ranges[io_pair][1]
                # expand the current subset of basis vectors and set the result in the appropriate place in the filter

                # Basis Matrices spawing the space of equivariant linear maps of this block
                basis_set_linear_map = block_expansion.sampled_basis.detach().cpu().numpy()[:, :, :, 0]
                # We want to find the coefficients of this basis responsible for the identity matrix. These are the
                # elements of the basis having no effect on off-diagonal elements of the block.
                basis_dimension = basis_set_linear_map.shape[0]
                singlar_value_dimensions = []
                for element_num in range(basis_dimension):
                    # Get the basis matrix corresponding to this element
                    basis_matrix = basis_set_linear_map[element_num]
                    # Assert that all elements off-diagonal are zero
                    is_singular_value = np.allclose(basis_matrix, np.diag(np.diag(basis_matrix)), rtol=1e-4, atol=1e-4)
                    if is_singular_value:
                        singlar_value_dimensions.append(element_num)
                coefficients = torch.zeros((basis_dimension,))
                coefficients[singlar_value_dimensions] = 1

                # retrieve the linear coefficients for the basis expansion
                identity_coefficients[start_coeff:end_coeff] = coefficients

            self.transfer_op.weights.data = identity_coefficients
            matrix, _ = self.transfer_op.expand_parameters()
            eigvals = torch.linalg.eigvals(matrix)
            eigvals_real = eigvals.real.detach().cpu().numpy()
            eigvals_imag = eigvals.imag.detach().cpu().numpy()
            assert np.allclose(np.abs(eigvals_real), np.ones_like(eigvals_real), rtol=1e-4, atol=1e-4), \
                f"Eigenvalues with real part different from 1: {eigvals_real}"
            assert np.allclose(eigvals_imag, np.zeros_like(eigvals_imag), rtol=1e-4, atol=1e-4), \
                f"Eigenvalues with imaginary part: {eigvals_imag}"
        elif init_mode == "isotypic_identity":
            # Incredibly shady hack in order to set the identity operator between all irreps of the same type.
            block_coeff = []
            block_init_transfer_op = []
            for io_pair in basis_expansion._representations_pairs:
                # retrieve the basis
                block_expansion = getattr(basis_expansion, f"block_expansion_{basis_expansion._escape_pair(io_pair)}")
                basis_instance = block_expansion.basis
                assert len(basis_instance.irreps_bases) == 1, "We need to handle non-isotypic blocks"
                in_rep, out_rep = block_expansion.basis.in_repr, block_expansion.basis.out_repr
                group = in_rep.group
                in_irrep_id, out_irrep_id = np.unique(in_rep.irreps)[0], np.unique(out_rep.irreps)[0]
                # Cyclic group fucking Ids.
                in_irrep_id = (in_irrep_id,) if not isinstance(in_irrep_id, tuple) else in_irrep_id
                out_irrep_id = (out_irrep_id,) if not isinstance(out_irrep_id, tuple) else out_irrep_id
                basis_set_linear_map = block_expansion.sampled_basis.detach().cpu().numpy()[:, :, :, 0]
                basis_dimension, out_dim, in_dim = basis_set_linear_map.shape
                if group.irrep(*in_irrep_id) == in_rep.group.trivial_representation:
                    init_transfer_op = np.eye(out_dim, in_dim)
                else:
                    in_irrep, out_irrep = group.irrep(*in_irrep_id), group.irrep(*out_irrep_id)
                    in_multiplicity, out_multiplicity = in_dim // in_irrep.size, out_dim // out_irrep.size
                    irrep_init_transfer_op = np.eye(out_irrep.size, in_irrep.size) / max(out_multiplicity,
                                                                                         in_multiplicity)
                    init_transfer_op = scipy.linalg.kron(np.ones((out_multiplicity, in_multiplicity)),
                                                         irrep_init_transfer_op)
                    print("")

                coeff = represent_linear_map_in_basis(basis_linear_map=basis_set_linear_map,
                                                      in_linear_map=init_transfer_op)
                block_coeff.append(coeff)
                block_init_transfer_op.append(init_transfer_op)

            block_coeff = np.concatenate(block_coeff)
            self.transfer_op.weights.data = torch.from_numpy(block_coeff).to(device=self.transfer_op.weights.device,
                                                                             dtype=self.transfer_op.weights.dtype)
            matrix, _ = self.transfer_op.expand_parameters()
            matrix_np = matrix.detach().cpu().numpy()
            eigvals = torch.linalg.eigvals(matrix)
            eigvals_real = eigvals.real.detach().cpu().numpy()
            eigvals_imag = eigvals.imag.detach().cpu().numpy()
            assert np.allclose(matrix_np, scipy.linalg.block_diag(*block_init_transfer_op)), \
                f"Probably the block coefficients order of ESCNN is not what we assumed"
            raise NotImplementedError("This is not working yet")
        else:
            raise NotImplementedError(f"Eival init mode {init_mode} not implemented")

        log.info(f"Eigenvalues initialization to {init_mode}")


if __name__ == "__main__":

    G = escnn.group.DihedralGroup(4)
    rep = G.representations["regular"]
    test_equiv_lin = EquivLinearDynamics(state_rep=rep,
                                         dt=1,
                                         trainable=True),