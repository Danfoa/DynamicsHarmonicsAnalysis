import logging
from typing import List, Union

import escnn
import numpy as np
from escnn import gspaces
from escnn.nn import EquivariantModule, FieldType

log = logging.getLogger(__name__)


class EMLP(EquivariantModule):
    """Equivariant Multi-Layer Perceptron (EMLP) model."""

    def __init__(self,
                 in_type: FieldType,
                 out_type: FieldType,
                 num_hidden_units: int = 64,
                 num_layers: int = 3,
                 with_bias: bool = True,
                 activation: str = "ELU",
                 head_with_activation: bool = False,
                 batch_norm: bool = True,
                 init_mode="fan_in"):
        """Constructor of an Equivariant Multi-Layer Perceptron (EMLP) model.

        This utility class allows to easily instanciate a G-equivariant MLP architecture. As a convention, we assume
        every internal layer is a map between the input space: X and an output space: Y, denoted as
        z = σ(y) = σ(W x + b). Where x ∈ X, W: X -> Y, b ∈ Y. The group representation used for intermediate layer
        embeddings ρ_Y: G -> GL(Y) is defined as a sum of multiple regular representations:
        ρ_Y := ρ_reg ⊕ ρ_reg ⊕ ... ⊕ ρ_reg. Therefore, the number of `hidden layer's neurons` will be a multiple of |G|.
        Being the multiplicities of the regular representation: ceil(num_hidden_units/|G|)

        Args:
        ----
            in_type (escnn.nn.FieldType): Input field type containing the representation of the input space.
            out_type (escnn.nn.FieldType): Output field type containing the representation of the output space.
            num_hidden_units: Number of hidden units in the intermediate layers. The effective number of hidden units
            will be ceil(num_hidden_units/|G|). Since we assume intermediate embeddings are regular fields.
            num_layers: Number of layers in the MLP including input and output/head layers. That is, the number of
            hidden layers will be num_layers - 2.
            with_bias: Whether to include a bias term in the linear layers.
            activation (escnn.nn.EquivariantModule, list(escnn.nn.EquivariantModule)): If a single activation module is
            provided it will be used for all layers except the output layer. If a list of activation modules is provided
            then `num_layers` activation equivariant modules should be provided.
            head_with_activation: Whether to include an activation module in the output layer.
            init_mode: Not used until now. Will be used to initialize the weights of the MLP.
        """
        super(EMLP, self).__init__()
        logging.info("Instantiating EMLP (PyTorch)")
        self.in_type, self.out_type = in_type, out_type
        self.gspace = self.in_type.gspace
        self.group = self.gspace.fibergroup
        self.activation = activation.lower()

        self.num_layers = num_layers
        if self.num_layers == 1 and not head_with_activation:
            log.warning(f"{self} model with 1 layer and no activation. This is equivalent to a linear map")

        # Approximate the num of neurons as the num of signals in the space spawned by the irreps of the input type
        self.num_hidden_regular_fields = int(np.ceil(num_hidden_units // self.in_type.size))
        # To compute the signal over the group we use all elements for finite groups
        activation = self.get_activation(self.activation)
        hidden_type = activation.in_type

        input_irreps = set(in_type.representation.irreps)
        inner_irreps = set(out_type.irreps)
        diff = input_irreps.symmetric_difference(inner_irreps)
        if len(diff) > 0:
            log.warning(f"Irreps {list(diff)} of group {self.gspace.fibergroup} are not in the input/output types."
                        f"This represents an information bottleneck. Consider extracting invariant features.")

        layer_in_type = in_type

        self.net = escnn.nn.SequentialModule()
        for n in range(self.num_layers - 1):
            layer_out_type = hidden_type

            block = escnn.nn.SequentialModule()
            block.add_module(f"linear_{n}", escnn.nn.Linear(layer_in_type, layer_out_type, bias=with_bias))
            if batch_norm:
                block.add_module(f"batchnorm_{n}", escnn.nn.IIDBatchNorm1d(layer_out_type)),
            block.add_module(f"act_{n}", activation)

            self.net.add_module(f"block_{n}", block)
            layer_in_type = layer_out_type

        # Add final layer
        head_block = escnn.nn.SequentialModule()
        head_block.add_module(f"linear_{num_layers - 1}", escnn.nn.Linear(layer_in_type, out_type, bias=with_bias))
        if head_with_activation:
            if batch_norm:
                head_block.add_module(f"batchnorm_{num_layers - 1}", escnn.nn.IIDBatchNorm1d(out_type)),
            head_block.add_module(f"act_{num_layers - 1}", activation)
        # head_layer.check_equivariance()
        self.net.add_module("head", head_block)
        # Test the entire model is equivariant.
        # self.net.check_equivariance()

    def get_activation(self, activation):
        grid_length = self.group.order() if not self.group.continuous else 20 #self.group._maximum_frequency
        if "identity" in activation.lower():
            raise NotImplementedError("Identity activation not implemented yet")
            # return escnn.nn.IdentityModule()
        else:
            return escnn.nn.FourierPointwise(self.gspace,
                                             channels=self.num_hidden_regular_fields,
                                             irreps=self.in_type.irreps,
                                             function=f"p_{activation.lower()}",
                                             inplace=True,
                                             type='regular' if not self.group.continuous else 'rand',
                                             N=grid_length)

    def forward(self, x):
        """Forward pass of the EMLP model."""
        return self.net(x)

    def get_hparams(self):
        return {'num_layers': self.num_layers,
                'hidden_ch':  self.num_hidden_regular_fields,
                'activation': str(self.activation.__class__.__name__),
                }

    def reset_parameters(self, init_mode=None):
        """Initialize weights and biases of E-MLP model."""
        raise NotImplementedError()

    def evaluate_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        """Returns the output shape of the model given an input shape."""
        batch_size = input_shape[0]
        return batch_size, self.out_type.size


class SO2MLP(EquivariantModule):

    def __init__(self,
                 in_type: FieldType,
                 out_type: FieldType,
                 n_classes=10,
                 **kwargs):
        super(SO2MLP, self).__init__()

        self.G = in_type.gspace.fibergroup

        # since we are building an MLP, there is no base-space
        self.gspace = gspaces.no_base_space(self.G)

        # the input contains the coordinates of a point in the 2D space
        self.in_type = in_type  # self.gspace.type(self.G.standard_representation())
        in_irreps = in_type.irreps
        # Layer 1
        # We will use the regular representation of SO(2) acting on signals over SO(2) itself, bandlimited to
        # frequency 1
        # Most of the comments on the previous SO(3) network apply here as well

        activation1 = escnn.nn.FourierELU(
            self.gspace,
            channels=10,  # specify the number of signals in the output features
            irreps=in_irreps,  # self.G.bl_regular_representation(L=1).irreps,  # include all frequencies up to L=1
            inplace=True,
            # the following kwargs are used to build a discretization of the circle containing 6 equally distributed
            # points
            type='regular', N=20,
            )

        # map with an equivariant Linear layer to the input expected by the activation function, apply batchnorm and
        # finally the activation
        self.block1 = escnn.nn.SequentialModule(
            escnn.nn.Linear(self.in_type, activation1.in_type, bias=False),
            escnn.nn.IIDBatchNorm1d(activation1.in_type),
            activation1,
            )

        # Repeat a similar process for a few layers

        # 8 signals, bandlimited up to frequency 3
        activation2 = escnn.nn.FourierELU(
            self.gspace,
            channels=8,  # specify the number of signals in the output features
            irreps=in_irreps,  # self.G.bl_regular_representation(L=3).irreps,  # include all frequencies up to L=3
            inplace=True,
            # the following kwargs are used to build a discretization of the circle containing 16 equally distributed
            # points
            type='regular', N=16,
            )

        self.block2 = escnn.nn.SequentialModule(
            escnn.nn.Linear(self.block1.out_type, activation2.in_type, bias=False),
            escnn.nn.IIDBatchNorm1d(activation2.in_type),
            activation2,
            )

        # 8 signals, bandlimited up to frequency 3
        activation3 = escnn.nn.FourierELU(
            self.gspace,
            channels=8,  # specify the number of signals in the output features
            irreps=in_irreps,  # self.G.bl_regular_representation(L=3).irreps,  # include all frequencies up to L=3
            inplace=True,
            # the following kwargs are used to build a discretization of the circle containing 16 equally distributed
            # points
            type='regular', N=16,
            )
        self.block3 = escnn.nn.SequentialModule(
            escnn.nn.Linear(self.block2.out_type, activation3.in_type),
            escnn.nn.IIDBatchNorm1d(activation3.in_type),
            activation3,
            )

        # 5 signals, bandlimited up to frequency 2
        activation4 = escnn.nn.FourierELU(
            self.gspace,
            channels=5,  # specify the number of signals in the output features
            irreps=in_irreps,  # self.G.bl_regular_representation(L=2).irreps,  # include all frequencies up to L=2
            inplace=True,
            # the following kwargs are used to build a discretization of the circle containing 12 equally distributed
            # points
            type='regular', N=12,
            )
        self.block4 = escnn.nn.SequentialModule(
            escnn.nn.Linear(self.block3.out_type, activation4.in_type),
            escnn.nn.IIDBatchNorm1d(activation4.in_type),
            activation4,
            )

        # Final linear layer mapping to the output features
        # the output is a 2-dimensional vector rotating with frequency 2
        self.out_type = out_type  # self.gspace.type(self.G.irrep(2))
        self.block5 = escnn.nn.Linear(self.block4.out_type, self.out_type)

        print(self)
        print("")

    def forward(self, x: escnn.nn.GeometricTensor):
        # check the input has the right type
        assert x.type == self.in_type

        # apply each equivariant block

        # Each layer has an input and an output type
        # A layer takes a GeometricTensor in input.
        # This tensor needs to be associated with the same representation of the layer's input type
        #
        # The Layer outputs a new GeometricTensor, associated with the layer's output type.
        # As a result, consecutive layers need to have matching input/output types
        x = self.block1(x)
        a = x.tensor.detach().cpu().numpy()
        x = self.block2(x)
        b = x.tensor.detach().cpu().numpy()
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)

        return x

    def evaluate_output_shape(self, input_shape: tuple):
        shape = list(input_shape)
        assert len(shape) == 2, shape
        assert shape[1] == self.in_type.size, shape
        shape[1] = self.out_type.size
        return shape

    def get_hparams(self):
        return {}
