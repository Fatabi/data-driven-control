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

        # Taken from the link below:
        # https://ctms.engin.umich.edu/CTMS/index.php?example=InvertedPendulum&section=SystemModeling
        self.m2_l2_g = m ** 2 * length ** 2 * g
        self.inertia_plus_m_l2 = inertia + m * length ** 2
        self.m_l = m * length
        self.b = b
        self.M_plus_m = M + m
        self.M2_l2 = M ** 2 * length ** 2
        self.g = g

    def forward(self, t: th.Tensor, X: th.Tensor) -> th.Tensor:
        x, x_dot, theta, theta_dot = X[..., 0:1], X[..., 1:2], X[..., 2:3], X[..., 3:4]
        F = self.u(t, X)
        x_dot_dot = (
            self.m2_l2_g * th.sin(2 * theta) / 2
            + self.inertia_plus_m_l2 * (self.m_l * theta_dot ** 2 * th.sin(theta) - self.b * x + F)
        ) / (self.inertia_plus_m_l2 * self.M_plus_m - self.M2_l2 * th.cos(theta) ** 2)
        theta_dot_dot = -self.m_l * (self.g * th.sin(theta) + x_dot_dot * th.cos(theta)) / self.inertia_plus_m_l2
        X_dot = th.cat([x_dot, x_dot_dot, theta_dot, theta_dot_dot], dim=-1)
        return X_dot


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

    def forward(self, t: th.Tensor, X: th.Tensor) -> th.Tensor:
        q, p = X[..., :1], X[..., 1:]
        cur_u = self.u(t, X)
        dq = p / self.m
        dp = -self.k * (q - self.qr) - self.m * self.g * self.l * th.sin(q) - self.β * p / self.m + cur_u
        cur_f = th.cat([dq, dp], -1)
        return cur_f
