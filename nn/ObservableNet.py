from typing import Optional

import torch.nn

from nn.mlp import MLP
from nn.emlp import EMLP


class TwinMLP(torch.nn.Module):
    """Auxiliary class to construct Twin MLPs with a potentially shared backbone."""

    def __init__(self, net_kwargs: dict, backbone_kwargs: Optional[dict] = None, equivariant=False, fake_aux_fn=False):
        super().__init__()
        self.fake_aux_fn = fake_aux_fn
        self.shared_backbone = backbone_kwargs is not None
        mlp_class = MLP if not equivariant else EMLP # SO2MLP

        if self.shared_backbone:
            self.backbone = mlp_class(**backbone_kwargs)

        self.fn1 = mlp_class(**net_kwargs)
        if not fake_aux_fn:
            self.fn2 = mlp_class(**net_kwargs)
        else:
            pass

    def forward(self, input):

        if self.shared_backbone:
            backbone_output = self.backbone(input)
            output1 = self.fn1(backbone_output)
            output2 = self.fn2(backbone_output)
        else:
            if self.fake_aux_fn:
                output1 = self.fn1(input)
                output2 = output1
            else:
                output1 = self.fn1(input)
                output2 = self.fn2(input)

        return output1, output2

    def get_hparams(self):
        return {}


