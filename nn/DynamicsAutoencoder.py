import warnings
from typing import Optional
from abc import ABC

import torch
from torch.nn import Module

from src.RobotEquivariantNN.groups.SparseRepresentation import SparseRep
from src.RobotEquivariantNN.nn.EMLP import EMLP, MLP
from src.RobotEquivariantNN.nn.EquivariantModules import EquivariantModule, BasisConv1d, BasisLinear

import logging

from utils.complex import view_as_complex

log = logging.getLogger(__name__)

UNBOUND_REVOLUTE = "JointModelRUB"


class DynamicsModule(Module):

    def __init__(self, state_dim: int, dt: float, robot=None, respect_state_topology=False, **kwargs):
        super().__init__()
        self.state_dim = state_dim
        self.dt = dt
        self.robot = robot
        self._input_mean = kwargs.get("input_mean", None)
        self._input_std = kwargs.get("input_std", None)

        if robot is not None:
            import pinocchio
            if not isinstance(robot, pinocchio.Model):
                raise AttributeError(f"robot must be a pinocchio.Model instance not {robot.__class__.__name__}")

        if respect_state_topology:
            if robot is None:
                raise AttributeError(f"No way of knowing state topology without robot model")
            self._post_fn_q, self._post_fn = self.conf_state_space_preprocess_fn()
        else:
            self._post_fn_q, self._post_fn = [], []

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        # Standarized the state if required
        # if self._input_std is not None and self._input_mean is not None:
        #     x = (x - self._input_mean) / self._input_std

        out = self.forcast(x)

        # Unstandarized the state if required
        if self._input_std is not None and self._input_mean is not None:
            out['x_pred'] = (out['x_pred'] * self._input_std.to(out['x_pred'].device)) + self._input_mean.to(out['x_pred'].device)

        # Execute any post-processing required, e.g., ensuring manifold constraints
        out['x_pred'] = self.forward_postprocess(out['x_pred'])

        return out

    def forcast(self, x):
        raise NotImplementedError("")

    def forward_postprocess(self, x: torch.Tensor):

        if len(self._post_fn) == 0:
            return x
        x_pos = x

        for idx_q, postprocess_fn in zip(self._post_fn_q, self._post_fn):
            # TODO: Ensure state is (q, dq, *other)
            x_pos = postprocess_fn(x, idx_q)

        return x_pos

    def conf_state_space_preprocess_fn(self):
        """
        This function will be used to post-process the estimate of the state-space of the system such that the model
        respects the topological structure of the space.
        For instance an unbounded revolute joint has a state variable `¢` is a variable evolving in the unite circle,
        which a NN better processes if we transformed it to x(¢)=[cos(¢), sin(¢)] such that x(¢) = x(¢ + 2pi). i.e.,
        model is not affected by the discontinuity of transitioning form 2pi to 0.
        The output of the NN in this case would be a vector x'=[x1', x2'] which will not necessarily be in the unit
        circle `S1`. Thus, the post-processing of this joint topology is the normalization of the estimation of the state to
        enfoce the estimate is in `S1`.
        This line of thought can be extended to quaternions/rot matrices describing body orientations in space.
        """
        REVOLUTE = "JointModelR"
        if self.robot is None:
            log.info("Dynamical system state space topology unknown. Assuming vector space")
            return

        joint_idx_q = []
        joint_idx_v = []
        post_process_fn = []

        for i, joint in enumerate(self.robot.joints):
            joint_type = joint.shortname()
            if joint.idx_q == -1 or joint.idx_v == -1:
                log.debug(f"Ignoring joint not in state space {joint_type}")
                continue
            log.debug(f"{joint_type} nq:{joint.nq} nv:{joint.nv} idx_q:{joint.idx_q} idx_v:{joint.idx_v}")

            if UNBOUND_REVOLUTE in joint_type:  # The DoF belongs to the S1: Unit Circle Lie Group

                # Little test to do a sanity check for future modifications.
                # TODO: Remove on release
                test_angles = torch.rand(2, 5, 4)
                idx_q = (0, 1)
                new_angles = DynamicsModule.postprocess_S1(test_angles, idx_q)
                norms = torch.norm(new_angles[..., idx_q], dim=-1, keepdim=True)
                assert torch.allclose(norms, torch.ones_like(norms))

                post_process_fn.append(DynamicsModule.postprocess_S1)
                joint_idx_q.append(list(range(joint.idx_q, joint.idx_q + joint.nq)))
                joint_idx_v.append(list(range(joint.idx_v, joint.idx_v + joint.nv)))

        return joint_idx_q, post_process_fn

    @staticmethod
    def postprocess_S1(x: torch.tensor, idx):
        """
        Convert an unnormalized 2D vector into a normalized vector. This turns any vector into a coordinate
        of the unit circle S1.
        x: tensor containing batched state information [..., N], being N the state dimension
        idx: (2,) indices of last dimension of `x` holding the estimates of cos(¢), sin(¢) coordinates.
        """
        assert len(idx) == 2, "We asusme dim [..., 2] where the last two dims refer to cos(¢) sin(¢)"
        new_x = x.clone()
        norm = torch.norm(x[..., idx], dim=-1, keepdim=True)
        new_x[..., idx] = x[..., idx] / norm
        return new_x

    def extra_repr(self) -> str:
        return f"state-dim:{self.state_dim}_dt:{self.dt}"


class DynamicsAutoEncoder(DynamicsModule):

    def __init__(self, state_dim: int, obs_dim: int, dt: float, num_hidden_layers=2, num_hidden_cells=32,
                 robot=None, activation: Module = torch.nn.Tanh, **kwargs):
        super(DynamicsAutoEncoder, self).__init__(state_dim=state_dim, dt=dt, robot=robot, **kwargs)

        self.encoder = MLP(d_in=state_dim, d_out=obs_dim * 2, ch=num_hidden_cells, num_layers=num_hidden_layers,
                           activation=activation)
        self.decoder = MLP(d_in=obs_dim * 2, d_out=state_dim, ch=num_hidden_cells, num_layers=num_hidden_layers,
                           activation=activation)

        self.observation_dynamics = LinearEigenvectorDynamics(dim=obs_dim, **kwargs)

    def forcast(self, x):

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


# TODO: Fix multiple inheritance DynamicsModule, EquivariantModule
class EDynamicsAutoEncoder(EquivariantModule, DynamicsModule):

    def __init__(self, repX: SparseRep, obs_dim: int, dt: float, num_hidden_layers=2, num_hidden_cells=32,
                 robot=None, activation: Module = torch.nn.Tanh, **kwargs):
        DynamicsModule.__init__(self, state_dim=repX.G.d, dt=dt, robot=robot, **kwargs)
        EquivariantModule.__init__(self, rep_in=repX, rep_out=repX)
        self.rep_in = repX
        self.rep_out = repX

        # Decompose the space of complex observations into blocks of spaces isomorphic to the regular rep space
        self.repZ = SparseRep(self.rep_in.G.isotypic_decomposition(obs_dim))

        # Encoder computes the complex observations
        self.encoder = EMLP(rep_in=self.rep_in, rep_out=self.repZ, ch=num_hidden_cells, n_layers=num_hidden_layers,
                            activation=[activation] * num_hidden_layers)
        self.encoder2re_obs = BasisLinear(rep_in=self.repZ, rep_out=self.repZ, bias=False)
        self.encoder2imag_obs = BasisLinear(rep_in=self.repZ, rep_out=self.repZ, bias=False)
        # The input of the decoder is complex, having the real and img the same symmetry transformation
        self.decoder = EMLP(rep_in=self.repZ + self.repZ, rep_out=self.rep_in, ch=num_hidden_cells,
                            n_layers=num_hidden_layers + 1,
                            activation=[activation] * num_hidden_layers + [torch.nn.Identity])
        self.observation_dynamics = LinearEigenvectorDynamics(dim=obs_dim, **kwargs)

        # print(list(self.parameters(recurse=True)))
        # Test equivariance after observations are evolved. That is, test for G-equivariant linear observation dynamics
        self.test_module_equivariance(repX, self.repZ, in_shape=(1, 10, repX.G.d), output_fn=lambda x: x['z_pred'])
        self.test_module_equivariance(repX, repX, in_shape=(1, 10, repX.G.d), output_fn=lambda x: x['x_pred'])

    def forward(self, x):
        return DynamicsModule.forward(self, x)

    def forcast(self, x):
        encoder_output = self.encoder(x)

        re_obs = self.encoder2re_obs(encoder_output)
        img_obs = self.encoder2img_obs(encoder_output)
        z_real = torch.stack((re_obs, img_obs), dim=3)
        z = torch.view_as_complex(z_real)

        if self.observation_dynamics is None:
            z_pred = z
        else:
            dts = torch.arange(0, z.shape[1], device=x.device) * self.dt
            z_pred = self.observation_dynamics(z[:, 0, :], dts)

        # Turn complex observations into stacked vectors [re(ø_1),...re(ø_m), img(ø_1),...img(ø_m)
        # Such that we can apply the symmetry as rep=block_diag(repZ, repZ)
        z_pred_real = torch.view_as_real(z_pred)
        z_

        x_pred = self.decoder(z_pred)
        return {"x_pred": x_pred, "z_pred": z_pred, "z": z}

    def get_hparams(self):
        return {'encoder': self.encoder.get_hparams(),
                'decoder': self.decoder.get_hparams()}


class LinearEigenvectorDynamics(Module):

    def __init__(self, dim: int, eigval_init="stable", eigval_constraint="unconstrained", **kwargs):
        super().__init__()
        assert dim % 2 == 0, "For now only cope with even dimensions."
        self.dim = dim
        self._eigval_init = eigval_init
        self._eigval_constraint = eigval_constraint

        # Create the parameters determining the learnable eigenvalues of the eigenmatrix.
        # Each eigval is parameterized as re^(iw)
        w = torch.rand(self.dim)
        r = torch.rand(self.dim)
        if eigval_constraint == "unconstrained":
            self.w = torch.nn.Parameter(w, requires_grad=True)
            self.r = torch.nn.Parameter(r, requires_grad=True)
        elif eigval_constraint == "unit_circle":
            self.w = torch.nn.Parameter(w, requires_grad=True)
            r = r * 0 + 1.
            self.r = torch.nn.Parameter(r, requires_grad=False)

        else:
            raise NotImplementedError(f"Eigval constraint {self._eigval_constraint} not implemented")

        # Initialize eigenvalues.
        self.reset_parameters(init_mode=eigval_init)

    def forward(self, z_real, dt):
        assert torch.all(torch.isreal(z_real)), "We assume each complex observation is passed as a 2D real vector"
        # Each complex observation is assumed to be an eigenfunction, forced to experience linear/eigenvalue dynamics
        z_cplx = view_as_complex(z_real)
        z_cplx = torch.unsqueeze(z_cplx, 1)  # Add timestep dimension,

        # Diagonal of the eigen matrix
        matrix_eigvals = torch.unsqueeze(self.eigvals, 0)

        # Matrix exponential of a diagonal matrix is the exponent of the diagonal elements.
        discrete_eigvals = torch.exp(torch.mul(torch.unsqueeze(dt, 1), matrix_eigvals))

        # Evolve the eigenfunctions z_t+dt = K·z_t
        z_pred_cplx = torch.mul(z_cplx, discrete_eigvals)
        # Reshape to original shape is required because of how view_as_real works.
        original_shape = (tuple(z_pred_cplx.shape[:2]) + (-1,))
        z_pred_real = torch.reshape(torch.view_as_real(z_pred_cplx), original_shape)
        return z_pred_real

    @property
    def eigvals(self) -> torch.Tensor:
        re_eig = self.r * torch.cos(self.w)
        img_eig = self.r * torch.sin(self.w)
        eigvals = torch.view_as_complex(torch.stack((re_eig, img_eig), dim=1))
        return eigvals

    def reset_parameters(self, init_mode: str):
        self._eigval_init = init_mode
        if init_mode == "stable":
            torch.nn.init.uniform_(self.w, torch.pi / 2, 3 * torch.pi / 2)
            torch.nn.init.ones_(self.r)
            eigvals = self.eigvals
            # Check stability
            assert torch.all(torch.real(eigvals) < 0), f"Not stable eigenvalues"
            # Check unit circle
            assert torch.allclose(torch.abs(eigvals), torch.ones_like(self.r)), f"Not in unit circle"
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