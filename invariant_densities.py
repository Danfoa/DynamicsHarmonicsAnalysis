import numpy as np
import torch
from escnn.nn import FieldType
from morpho_symm.nn.EMLP import EMLP
import escnn

import plotly.graph_objects as go
import numpy as np


if __name__ == "__main__":

    n = 50
    r = np.linspace(0, 1, n)
    x = np.linspace(-1, 1, n)
    y = np.linspace(-1, 1, n)
    x, y = np.meshgrid(x, y)

    G = escnn.group.DihedralGroup(3)

    plane_rep = G.irreps()[2]
    trivial_rep = G.trivial_representation

    gspace = escnn.gspaces.no_base_space(G)

    in_field_type = FieldType(gspace, [plane_rep])
    # Representation of y := [l, k] ∈ R3 x R3            =>    ρ_Y_js(g) := ρ_O3(g) ⊕ ρ_O3pseudo(g)  | g ∈ G
    out_field_type = FieldType(gspace, [trivial_rep])

    # Construct the equivariant MLP
    model = EMLP(in_type=in_field_type,
                 out_type=out_field_type,
                 num_layers=6,  # Input layer + 3 hidden layers + output/head layer
                 num_hidden_units=16,  # Number of hidden units per layer
                 activation=escnn.nn.ReLU,  # Activarions must be `EquivariantModules` instances
                 with_bias=True  # Use bias in the linear layers
                 )

    mesh = np.stack([x.flatten(), y.flatten()], axis=1)

    input = model.in_type(torch.Tensor(mesh))
    input2 = model.in_type(torch.Tensor(mesh**2))

    z = model(input)
    z2 = model(input)

    z = np.squeeze(z.tensor.detach().numpy())
    z2 = np.squeeze(z2.tensor.detach().numpy())

    # Create a 3D surface plot
    z = z.reshape((n, n))
    z /= np.sum(z)

    z2 = z2.reshape((n, n))
    z2 /= np.sum(z2)

    fig = go.Figure(data=[go.Surface(z=z, x=x, y=y, opacity=1.0,
                                     contours={
                                         "z": {"show": True, "start": np.min(z), "end": np.max(z),
                                               "size": (np.max(z) - np.min(z)) / 10, "color": "white"}
                                         },
                                     )])
    # fig.add_trace(go.Surface(z=z2+(np.max(z) * 5.5), x=x, y=y, opacity=0.5,
    #                          contours={
    #                              "z": {"show": True, "start": np.min(z2) + (np.max(z) * 5.5), "end": np.max(z2) + (np.max(z) * 5.5),
    #                                    "size": (np.max(z2) - np.min(z2)) / 20, "color": "white"}
    #                              },
    #                          ))
    # Add axis labels
    fig.update_layout(scene=dict(
        xaxis_title='X Axis',
        yaxis_title='Y Axis',
        zaxis_title='Z Axis',
        xaxis={"nticks": 20},
        zaxis={"nticks": 4},
        camera_eye= {"x": 0, "y": -1, "z": 0.5},
        aspectratio= {"x": 1, "y": 1, "z": 0.2}
        ))

    fig.write_html("invariant_density.html")

    # Show the plot
    fig.show()