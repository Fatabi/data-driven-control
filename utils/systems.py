from math import pi
from typing import Callable

import torch as th
import torch.nn as nn


class ControlledCartPole(nn.Module):
    """
    Cart pole with a pendulum attached on it.
    """

    STATE_DIM = 4
    CONTROL_DIM = 1
    MODULO = th.tensor([float("nan"), float("nan"), 2 * pi, float("nan")])

    def __init__(
        self,
        u: Callable[[th.Tensor, th.Tensor, th.Tensor], th.Tensor],
        M: float = 0.5,
        m: float = 0.2,
        b: float = 0.1,
        inertia: float = 0.006,
        g: float = 9.81,
        length: float = 0.3,
    ):
        super().__init__()
        self.u = u  # controller (nn.Module)

        p = inertia * (M + m) + M * m * length**2  # denominator for the A and B matrices

        self.register_buffer(
            "A",
            th.tensor(
                [
                    [0, 1, 0, 0],
                    [0, -(inertia + m * length**2) * b / p, (m**2 * g * length**2) / p, 0],
                    [0, 0, 0, 1],
                    [0, -(m * length * b) / p, m * g * length * (M + m) / p, 0],
                ]
            ),
        )
        self.A: th.Tensor
        self.register_buffer("B", th.tensor([[0, (inertia + m * length**2) / p, 0, m * length / p]]))
        self.B: th.Tensor

    def forward(self, t: th.Tensor, x: th.Tensor) -> th.Tensor:
        cur_u = self.u(t, x)
        cur_f = th.matmul(x, self.A.T).T.permute([1, 0]) + th.matmul(cur_u, self.B)
        return cur_f


class ControlledPendulum(nn.Module):
    """
    Inverted pendulum with torsional spring
    """

    STATE_DIM = 2
    CONTROL_DIM = 1
    MODULO = th.tensor([2 * pi, float("nan")])

    def __init__(
        self,
        u: Callable[[th.Tensor, th.Tensor, th.Tensor], th.Tensor],
        m: float = 1.0,
        k: float = 0.5,
        length: float = 1.0,
        qr: float = 0.0,
        β: float = 0.01,
        g: float = 9.81,
    ):
        super().__init__()
        self.u = u  # controller (nn.Module)
        self.m, self.k, self.l, self.qr, self.β, self.g = m, k, length, qr, β, g  # physics

    def forward(self, t: th.Tensor, x: th.Tensor) -> th.Tensor:
        q, p = x[..., :1], x[..., 1:]
        cur_u = self.u(t, x)
        dq = p / self.m
        dp = -self.k * (q - self.qr) - self.m * self.g * self.l * th.sin(q) - self.β * p / self.m + cur_u
        cur_f = th.cat([dq, dp], -1)
        return cur_f
