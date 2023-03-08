import crocoddyl
import numpy as np


class CostModelPendulum(crocoddyl.CostModelAbstract):

    def __init__(self, state, activation, nu):
        activation = activation if activation is not None else crocoddyl.ActivationModelQuad(state.ndx)
        crocoddyl.CostModelAbstract.__init__(self, state, activation, nu=nu)

    def calc(self, data, x, u):
        c1 = np.cos(x[0])
        s1 = np.sin(x[0])
        data.residual.r[:] = np.array([s1, 1 - c1, x[2]])
        self.activation.calc(data.activation, data.residual.r)
        data.cost = data.activation.a_value

    def calcDiff(self, data, x, u):
        c1 = np.cos(x[0])
        s1 = np.sin(x[0])

        self.activation.calcDiff(data.activation, data.residual.r)

        data.residual.Rx[0, 0] = c1
        data.residual.Rx[1, 1] = s1
        data.residual.Rx[3, 3] = 1
        data.Lx[:] = np.dot(data.residual.Rx.T, data.activation.Ar)

        # data.Rxx[:2, :2] = np.diag([c1**2 - s1**2, c2**2 - s2**2])
        # data.Rxx[2:4, :2] = np.diag([s1**2 + (1 - c1) * c1, s2**2 + (1 - c2) * c2])
        # data.Rxx[4:6, 2:4] = np.diag([1, 1])
        # data.Lxx[:, :] = np.diag(np.dot(data.Rxx.T, np.diag(data.activation.Arr)))

    def createData(self, collector):
        data = CostDataPendulum(self, collector)
        return data


class CostDataPendulum(crocoddyl.CostDataAbstract):

    def __init__(self, model, collector):
        crocoddyl.CostDataAbstract.__init__(self, model, collector)
        self.Rxx = np.zeros((3, 2))


class ActuationModelPendulum(crocoddyl.ActuationModelAbstract):

    def __init__(self, state):
        crocoddyl.ActuationModelAbstract.__init__(self, state, 1)
        self.nv = state.nv

    def calc(self, data, x, u):
        data.tau[:] = data.dtau_du * u

    def calcDiff(self, data, x, u):
        pass

    def commands(self, data, x, tau):
        data.u[:] = tau[0]

    def torqueTransform(self, data, x, tau):
        pass

    def createData(self):
        data = ActuationDataPendulum(self)
        return data


class ActuationDataPendulum(crocoddyl.ActuationDataAbstract):

    def __init__(self, model):
        crocoddyl.ActuationDataAbstract.__init__(self, model)
        self.dtau_du[0] = 1.
        self.tau_set = [True]    # DoF with actuators.
        self.Mtau[0] = 1.
