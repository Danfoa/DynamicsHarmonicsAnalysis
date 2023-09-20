from collections import OrderedDict
from typing import Optional, Union

import numpy as np
import torch
from escnn.group import Representation
from torch import Tensor

from nn.LinearDynamics import DmdSolver, LinearDynamics
from nn.markov_dynamics import MarkovDynamics
from utils.mysc import full_rank_lstsq, full_rank_lstsq_symmetric
from utils.representation_theory import isotypic_basis


class EquivLinearDynamics(LinearDynamics):

    def __init__(self,
                 state_rep: Representation = None,
                 dmd_algorithm: Optional[DmdSolver] = None,
                 dt: Optional[Union[float, int]] = 1,
                 trainable=False,
                 group_avg_trick: bool = True):

        self.symm_group = state_rep.group
        self.group_avg_trick = group_avg_trick
        # Find the Isotypic basis of the state space
        self.state_iso_reps, self.state_iso_dims, Q_iso2state = isotypic_basis(representation=state_rep,
                                                                               multiplicity=1,
                                                                               prefix='ELDstate')
        # Change of coordinates required for state to be in Isotypic basis.
        Q_iso2state = Tensor(Q_iso2state)
        Q_state2iso = Tensor(np.linalg.inv(Q_iso2state))

        self.iso_transfer_op = OrderedDict()
        for irrep_id in self.state_iso_reps:  # Preserve the order of the Isotypic Subspaces
            self.iso_transfer_op[irrep_id] = None

        self.is_trainable = trainable
        dmd_algorithm = dmd_algorithm if dmd_algorithm is not None else full_rank_lstsq_symmetric
        super(EquivLinearDynamics, self).__init__(state_rep=state_rep,
                                                  dt=dt,
                                                  dmd_algorithm=dmd_algorithm,
                                                  state_change_of_basis=Q_state2iso,
                                                  state_inv_change_of_basis=Q_iso2state)

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
