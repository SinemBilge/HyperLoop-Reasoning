import torch
import torch.nn as nn
from typing import Union

from ..manifolds import PoincareBallCustomAutograd, PoincareBallStdGrad
import torch.nn.init as init
import math
from ..manifolds.lorentz import Lorentz



class LorentzLinear(nn.Module):
    """
    Hyperbolic Linear Layer

Parameters:
    manifold (Manifold): The manifold to use for the linear transformation.
    in_features (int): The size of each input sample.
    out_features (int): The size of each output sample.
    bias (bool, optional): If set to False, the layer will not learn an additive bias. Default is True.
    dropout (float, optional): The dropout probability. Default is 0.0.
    manifold_out (Manifold, optional): The output manifold. Default is None.
"""

    def __init__(self, manifold : Lorentz, in_features, out_features, bias=True, dropout=0.0, manifold_out=None):
        super().__init__()
        self.in_features = in_features + 1  # +1 for time dimension
        self.out_features = out_features
        self.bias = bias
        self.manifold = manifold
        self.manifold_out = manifold_out

        self.linear = nn.Linear(self.in_features, self.out_features, bias=bias)
        self.dropout_rate = dropout
        self.reset_parameters()

    def reset_parameters(self):
        """Reset layer parameters."""
        init.xavier_uniform_(self.linear.weight, gain=math.sqrt(2))
        if self.bias:
            init.constant_(self.linear.bias, 0)

    def forward(self, x, x_manifold='hyp'):
        """Forward pass for hyperbolic linear layer."""
        if x_manifold != 'hyp':
            x = torch.cat([torch.ones_like(x)[..., 0:1], x], dim=-1)
            x = self.manifold.expmap0(x)
        x_space = self.linear(x)

        x_time = ((x_space ** 2).sum(dim=-1, keepdims=True) + self.manifold.k).sqrt()
        x = torch.cat([x_time, x_space], dim=-1)
        if self.manifold_out is not None:
            x = x * (self.manifold_out.k / self.manifold.k).sqrt()
        return x


class PoincareLinear(nn.Module):
    """Poincare fully connected linear layer"""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        ball: Union[PoincareBallStdGrad, PoincareBallCustomAutograd],
        bias: bool = True,
        id_init: bool = True,
    ) -> None:
        super(PoincareLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.ball = ball
        self.has_bias = bias
        self.id_init = id_init

        self.z = nn.Parameter(torch.empty(in_features, out_features))
        if self.has_bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.bias = None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.id_init:
            self.z = nn.Parameter(
                1 / 2 * torch.eye(self.in_features, self.out_features)
            )
        else:
            nn.init.normal_(
                self.z, mean=0, std=(2 * self.in_features * self.out_features) ** -0.5
            )
        if self.has_bias:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ball.expmap0(x, dim=-1)
        y = self.ball.fully_connected(
            x=x,
            z=self.z,
            bias=self.bias,
        )
        y = self.ball.logmap0(y, dim=-1)
        return y