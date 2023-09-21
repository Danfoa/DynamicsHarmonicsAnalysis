import logging
from collections import OrderedDict
from typing import Optional, Union

import escnn
import numpy as np
import torch
from escnn.group import Representation
from escnn.nn import FieldType, GeometricTensor
from escnn.nn.modules.basismanager import BlocksBasisExpansion
from torch import Tensor

from nn.LinearDynamics import DmdSolver, LinearDynamics
from nn.markov_dynamics import MarkovDynamics
from utils.mysc import full_rank_lstsq, full_rank_lstsq_symmetric
from utils.representation_theory import isotypic_basis

log = logging.getLogger(__name__)

class EquivLinearDynamics(LinearDynamics):

    def __init__(self,
                 state_type: FieldType,
                 dmd_algorithm: Optional[DmdSolver] = None,
                 dt: Optional[Union[float, int]] = 1,
                 trainable=False,
                 group_avg_trick: bool = True):


        self.symm_group = state_type.fibergroup
        self.gspace = state_type.gspace
        self.state_rep = state_type.representation
        self.group_avg_trick = group_avg_trick
        dmd_algorithm = dmd_algorithm if dmd_algorithm is not None else full_rank_lstsq_symmetric
        # Find the Isotypic basis of the state space
        self.state_iso_reps, self.state_iso_dims, Q_iso2state = isotypic_basis(representation=self.state_rep,
                                                                               multiplicity=1,
                                                                               prefix='ELDstate')
        self.iso_space_basis = {}
        # Each Field for ESCNN will be an Isotypic Subspace.
        self.state_type = state_type
        # Field type on isotypic basis.
        self.state_type_iso = FieldType(self.gspace, [rep_iso for rep_iso in self.state_iso_reps.values()])

        # Change of coordinates required for state to be in Isotypic basis.
        Q_iso2state = Tensor(Q_iso2state)
        Q_state2iso = Tensor(np.linalg.inv(Q_iso2state))

        super(EquivLinearDynamics, self).__init__(state_rep=state_type.representation,
                                                  dt=dt,
                                                  trainable=trainable,
                                                  dmd_algorithm=dmd_algorithm,
                                                  state_change_of_basis=Q_state2iso,
                                                  state_inv_change_of_basis=Q_iso2state)

        self.iso_transfer_op = OrderedDict()
        for irrep_id in self.state_iso_reps:  # Preserve the order of the Isotypic Subspaces
            self.iso_transfer_op[irrep_id] = None

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
                transfer_op = self.get_transfer_op()
                next_obs_state = torch.nn.functional.linear(current_state, transfer_op)
            pred_state_traj.append(next_obs_state)

        if self.is_trainable:
            pred_state_traj = torch.stack([gt.tensor for gt in pred_state_traj], dim=1)
        else:
            pred_state_traj = torch.stack(pred_state_traj, dim=1)
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
            A_iso = self.dmd_algorithm(X_iso, X_iso_prime,
                                       rep_X=rep if self.group_avg_trick else None,
                                       rep_Y=rep if self.group_avg_trick else None)
            rec_error = torch.nn.functional.mse_loss(A_iso @ X_iso, X_iso_prime)
            iso_rec_error.append(rec_error)
            self.iso_transfer_op[irrep_id] = A_iso
        transfer_op = torch.block_diag(*[self.iso_transfer_op[irrep_id] for irrep_id in self.state_iso_reps.keys()])
        assert transfer_op.shape == (self.state_dim, self.state_dim)
        self.transfer_op = transfer_op

        iso_rec_error = Tensor(iso_rec_error)
        rec_error = torch.sum(iso_rec_error)
        self.transfer_op = transfer_op

        return dict(solution_op_rank=torch.linalg.matrix_rank(transfer_op.detach()).to(torch.float),
                    solution_op_cond_num=torch.linalg.cond(transfer_op.detach()).to(torch.float),
                    solution_op_error=rec_error.detach().to(torch.float),
                    solution_op_error_dist=iso_rec_error.detach().to(torch.float))

    def build_linear_map(self) -> escnn.nn.Linear:
        return escnn.nn.Linear(in_type=self.state_type_iso, out_type=self.state_type_iso, bias=False)

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
            self.iso_space_basis[irrep_id] = iso_basis
            # basis_coefficients = torch.rand((iso_basis.dimension(),)) + 2
            # non_zero_elements = iso_basis(basis_coefficients)[:, :, 0]
            # mask = torch.logical_not(torch.isclose(non_zero_elements, torch.zeros_like(non_zero_elements), atol=1e-6))
            # self.iso_space_basis_mask[irrep_id] = mask

        print("D")

    def reset_parameters(self, init_mode: str):
        if init_mode == "stable":
            equiv_map, _ = self.transfer_op.expand_parameters()
            # Do eigenvalue decomposition of the weights
            eig_vals, eig_vecs = torch.linalg.eig(equiv_map)

            marginally_stable_eigvals = torch.complex(torch.ones_like(eig_vals.real) * -0.99,
                                                      eig_vals.imag * self.dt**2)
            A = eig_vecs @ torch.diag(marginally_stable_eigvals) @ torch.inverse(eig_vecs)
            # Reconstruct weight matrix
            self.transfer_op.copy_ = A
        else:
            raise NotImplementedError(f"Eival init mode {init_mode} not implemented")

        log.info(f"Eigenvalues initialization to {init_mode}")


if __name__ == "__main__":

    G = escnn.group.DihedralGroup(4)
    rep = G.representations["regular"]
    test_equiv_lin = EquivLinearDynamics(state_rep=rep,
                                         dt=1,
                                         trainable=True),