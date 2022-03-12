from typing import Callable

import torch as th
import torch.nn as nn


class IntegralWReg(nn.Module):
    def __init__(self, sys: Callable[[th.Tensor, th.Tensor], th.Tensor], reg_coef: float = 1e-4):
        super().__init__()
        self.sys = sys
        self.reg_coef = reg_coef

    def forward(self, t: th.Tensor, x: th.Tensor) -> th.Tensor:
        loss = self.reg_coef * th.abs(self.sys(t, x)).sum(1)
        return loss
