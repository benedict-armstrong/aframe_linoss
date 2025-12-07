"""
Minimal Wrapper of LinOSS
https://github.com/KasraMazaheri/TritonLinOSS
https://github.com/tk-rusch/linoss
"""

import math
from typing import Literal


from jaxtyping import Float
import torch.nn as nn
from torch import Tensor
from damped_linoss import LinOSS
# from einops import rearrange, repeat


class LinOSSModel(nn.Module):
    def __init__(
        self,
        layer_name: Literal["IMEX", "IM", "Damped"],
        input_dim: int,
        state_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_blocks: int,
        classification: bool = False,
        tanh_output: bool = False,
        output_step: int = 1,
        r_min: float = 0.9,
        r_max: float = 1.0,
        theta_max: float = math.pi,
        drop_rate: float = 0.05,
    ):
        """Minimal Wrapper of LinOSS

        Args:
            layer_name: Type of LinOSS layer to use. One of "IMEX", "
                "IM", "Damped".
            input_dim: Input dimension.
            state_dim: State dimension.
            hidden_dim: Hidden dimension.
            output_dim: Output dimension.
            num_blocks: Number of LinOSS blocks.
            classification: Whether to use classification head.
            tanh_output: Whether to apply tanh activation to output.
            output_step: Output step size.
            r_min: Minimum radius for eigenvalues.
            r_max: Maximum radius for eigenvalues.
            theta_max: Maximum angle for eigenvalues.
            drop_rate: Dropout rate.
        """
        super().__init__()

        self.linoss = LinOSS(
            layer_name=layer_name,
            input_dim=input_dim,
            state_dim=state_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_blocks=num_blocks,
            classification=classification,
            tanh_output=tanh_output,
            output_step=output_step,
            r_min=r_min,
            r_max=r_max,
            theta_max=theta_max,
            drop_rate=drop_rate,
        )

    def forward(self, X: Float[Tensor, "b t d"]) -> Float[Tensor, "b t o"]:
        """
        x: (batch, time_steps, input_size)
        returns: (batch, time_steps, output_size)
        """
        return self.linoss(X)
