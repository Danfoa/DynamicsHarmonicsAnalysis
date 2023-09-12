from typing import Optional

import torch.nn

from nn.mlp import EMLP, MLP


class TwinMLP(torch.nn.Module):
    """Auxiliary class to construct Twin MLPs with a potentially shared backbone."""

    def __init__(self, net_kwargs: dict, backbone_kwargs: Optional[dict] = None, equivariant=False):
        super().__init__()

        self.shared_backbone = backbone_kwargs is not None
        mlp_class = MLP if not equivariant else EMLP

        if self.shared_backbone:
            self.backbone = mlp_class(**backbone_kwargs)

        self.fn1 = mlp_class(**net_kwargs)
        self.fn2 = mlp_class(**net_kwargs)

    def forward(self, input):

        if self.shared_backbone:
            backbone_output = self.backbone(input)
            output1 = self.fn1(backbone_output)
            output2 = self.fn2(backbone_output)
        else:
            output1 = self.fn1(input)
            output2 = self.fn2(input)

        return output1, output2

    def get_hparams(self):
        return dict(fn1=self.fn1.get_hparams(), fn2=self.fn2.get_hparams())


