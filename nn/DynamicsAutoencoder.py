from typing import Iterable

import torch
from torch.nn import Module

from data.ClosedLoopDynamics import STATES, CTRLS
from nn.EigenDynamics import EigenspaceDynamics
from src.RobotEquivariantNN.groups.SparseRepresentation import SparseRep
from src.RobotEquivariantNN.nn.EMLP import EMLP, MLP
from src.RobotEquivariantNN.nn.EquivariantModules import EquivariantModule, BasisLinear

import logging
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
        self.pred_w = kwargs['pred_w']

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

    def compute_loss_metrics(self, outputs_pred, outputs_gt):
        """
        :param x: Batched sequence of consecutive states [x0, x1,..., xT], T=num_steps
        :param z: Batches embeddings/observations of `z = ø(x)` [z0, z1,..., zT]
        :param x_pred: Batches prediction of sequence of consecutive states `xt_pred = ø^-1(K^t•ø(x0))`:
                       [ø^-1(ø(x0)), ø^-1(Kø(x0)), ... ø^-1(K^T•ø(x0))]
        :param z_pred: Batches embeddings/observations of `x_pred` [z0, Kz0,...,K^T•z0]
        :return:
        """
        z_pred = outputs_pred['z_pred']
        x_pred = outputs_pred['x_pred']
        z = outputs_pred['z']
        x_unscaled = outputs_gt
        # The output of the dynamics model is already unscaled so we have to unstandarize the dataset samples to compare
        # the results in the appropiate scale.
        # TODO: Make a more elegant solution here
        mean = self._input_mean
        std = self._input_std
        x = (x_unscaled * std) + mean
        # n = torch.norm(x[..., [0, 1]], dim=-1) == 1.0
        nx = x.shape[-1]

        x_err = torch.abs(torch.sub(x, x_pred))  # ∀ t: |x_t - ø^-1(K^t•ø(x_0))|
        z_err = torch.abs(torch.sub(z, z_pred))  # ∀ t: |z_t - K^t•ø(x_0)|

        norm_x_err = torch.norm(x_err, dim=-1, p=2)
        norm_z_err = torch.norm(z_err, dim=-1, p=2)
        # Reconstruction loss of the system state x_0.
        reconstruction_loss = torch.mean(norm_x_err[:, 0])

        # Prediction loss. From the system state compute prediction accuracy from multi-step predictions.
        avg_state_pred_err = torch.mean(norm_x_err[:, 1:], dim=1)

        metrics = {}
        if self.robot is not None:
            nq, nv = self.robot.nq, self.robot.nv
            metrics.update(q_err_rec=x_err[:, 0, :nq].mean(), q_err_pred=x_err[:, 1:, :nq].mean(),
                           dq_err_rec=x_err[:, 0, nq: nq + nv].mean(), dq_err_pred=x_err[:, 1:, nq: nq + nv].mean(),
                           u_err_rec=x_err[:, 0, nq + nv: nx].mean(), u_err_pred=x_err[:, 1:, nq + nv: nx].mean())

        # Linear dynamics of the observable/embedding.
        avg_obs_pred_err = torch.mean(norm_z_err[:, 1:], dim=1)

        # L-inf norm of reconstruction and single step prediction.
        x0, x1 = x[:, 0], x[:, 1]
        x0_rec, x1_pred = x_pred[:, 0], x_pred[:, 1]
        # Linf = ||x_0 - ø^-1(ø(x_0))||_inf + ||x_1 - ø^-1(K•ø(x_0))||_inf
        linf_loss = torch.norm(torch.sub(x0, x0_rec), p=float('inf'), dim=1) + \
                    torch.norm(torch.sub(x1, x1_pred), p=float('inf'), dim=1)

        metrics.update(rec_loss=reconstruction_loss, pred_loss=avg_state_pred_err.mean(),
                       lin_loss=avg_obs_pred_err.mean(), linf_loss=linf_loss.mean())

        loss = self.pred_w * (reconstruction_loss + metrics["pred_loss"]) + metrics["lin_loss"]
        return loss, metrics

    def batch_unpack(self, batch):
        return self.state_ctrl_to_x(batch)

    def batch_pack(self, x):
        return self.x_to_state_crtl(x)

    def state_ctrl_to_x(self, batch):
        """
        Mapping from batch of ClosedLoopDynamics data points to NN model input-output data points
        """
        inputs = torch.concatenate([batch[STATES], batch[CTRLS]], dim=2)
        return inputs

    def x_to_state_crtl(self, batch):
        x = batch.pop('x_pred')
        state = x[:, :, :self.robot.nq + self.robot.nv]
        ctrl = x[:, :, self.robot.nq + self.robot.nv:]
        batch.update({STATES: state, CTRLS: ctrl})
        return batch
    
    def get_metric_labels(self) -> Iterable[str]:
        return ["rec_loss", "pred_loss", "lin_loss", "linf_loss", "rec_loss", "pred_loss", "lin_loss", "linf_loss"]
        
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

    def __init__(self, state_dim: int, obs_dim: int, dt: float, num_hidden_layers=2, n_hidden_neurons=32,
                 robot=None, activation: Module = torch.nn.Tanh, **kwargs):
        super(DynamicsAutoEncoder, self).__init__(state_dim=state_dim, dt=dt, robot=robot, **kwargs)

        self.encoder = MLP(d_in=state_dim, d_out=obs_dim * 2, ch=n_hidden_neurons, n_layers=num_hidden_layers,
                           activation=activation)
        self.decoder = MLP(d_in=obs_dim * 2, d_out=state_dim, ch=n_hidden_neurons, n_layers=num_hidden_layers,
                           activation=activation)

        self.observation_dynamics = EigenspaceDynamics(dim=obs_dim, **kwargs)

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

    def __init__(self, repX: SparseRep, obs_dim: int, dt: float, num_hidden_layers=2, n_hidden_neurons=32,
                 robot=None, activation: Module = torch.nn.Tanh, **kwargs):
        DynamicsModule.__init__(self, state_dim=repX.G.d, dt=dt, robot=robot, **kwargs)
        EquivariantModule.__init__(self, rep_in=repX, rep_out=repX)
        self.rep_in = repX
        self.rep_out = repX

        # Decompose the space of complex observations into blocks of spaces isomorphic to the regular rep space
        self.repZ = SparseRep(self.rep_in.G.isotypic_decomposition(obs_dim))

        # Encoder computes the complex observations
        self.encoder = EMLP(rep_in=self.rep_in, rep_out=self.repZ, ch=n_hidden_neurons, n_layers=num_hidden_layers,
                            activation=[activation] * num_hidden_layers)
        self.encoder2re_obs = BasisLinear(rep_in=self.repZ, rep_out=self.repZ, bias=False)
        self.encoder2imag_obs = BasisLinear(rep_in=self.repZ, rep_out=self.repZ, bias=False)
        # The input of the decoder is complex, having the real and img the same symmetry transformation
        self.decoder = EMLP(rep_in=self.repZ + self.repZ, rep_out=self.rep_in, ch=n_hidden_neurons,
                            n_layers=num_hidden_layers + 1,
                            activation=[activation] * num_hidden_layers + [torch.nn.Identity])
        self.observation_dynamics = EigenspaceDynamics(dim=obs_dim, **kwargs)

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

        x_pred = self.decoder(z_pred)
        return {"x_pred": x_pred, "z_pred": z_pred, "z": z}

    def get_hparams(self):
        return {'encoder': self.encoder.get_hparams(),
                'decoder': self.decoder.get_hparams()}


