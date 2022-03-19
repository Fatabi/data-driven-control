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
        b: float = 0.0,
        inertia: float = 0.006,
        g: float = 9.81,
        length: float = 0.3,
    ):
        super().__init__()
        self.u = u  # controller (nn.Module)

        # Taken from the link below:
        # https://ctms.engin.umich.edu/CTMS/index.php?example=InvertedPendulum&section=SystemModeling
        self.m2_l2_g = m**2 * length**2 * g
        self.inertia_plus_m_l2 = inertia + m * length**2
        self.m_l = m * length
        self.b = b
        self.M_plus_m = M + m
        self.M2_l2 = M**2 * length**2
        self.g = g

    def forward(self, t: th.Tensor, X: th.Tensor) -> th.Tensor:
        x, x_dot, theta, theta_dot = X[..., 0:1], X[..., 1:2], X[..., 2:3], X[..., 3:4]  # noqa: F841
        F = self.u(t, X)
        x_dot_dot = (
            self.m2_l2_g * th.sin(2 * theta) / 2
            + self.inertia_plus_m_l2 * (self.m_l * theta_dot**2 * th.sin(theta) - self.b * x_dot + F)
        ) / (self.inertia_plus_m_l2 * self.M_plus_m - self.M2_l2 * th.cos(theta) ** 2)
        theta_dot_dot = -self.m_l * (self.g * th.sin(theta) + x_dot_dot * th.cos(theta)) / self.inertia_plus_m_l2
        X_dot = th.cat([x_dot, x_dot_dot, theta_dot, theta_dot_dot], dim=-1)
        return X_dot


class ControlledCartPoleV1(nn.Module):
    """
    Cart pole with a pendulum attached on it. Taken from:
    https://github.com/openai/gym/blob/6eec83cebb622b7e89d0465cd5bac57fd868c3d6/gym/envs/classic_control/cartpole.py
    """

    STATE_DIM = 4
    CONTROL_DIM = 1
    MODULO = th.tensor([float("nan"), float("nan"), 2 * pi, float("nan")])

    def __init__(
        self,
        u: Callable[[th.Tensor, th.Tensor, th.Tensor], th.Tensor],
        M: float = 0.5,
        m: float = 0.2,
        g: float = 9.81,
        length: float = 0.3,
    ):
        super().__init__()
        self.u = u  # controller (nn.Module)

        self.gravity = g
        self.masscart = M
        self.masspole = m
        self.total_mass = self.masspole + self.masscart
        self.length = length  # actually half the pole's length
        self.polemass_length = self.masspole * self.length

    def forward(self, t: th.Tensor, X: th.Tensor) -> th.Tensor:
        x, x_dot, theta, theta_dot = X[..., 0:1], X[..., 1:2], X[..., 2:3], X[..., 3:4]  # noqa: F841
        F = self.u(t, X)
        costheta = th.cos(theta)
        sintheta = th.sin(theta)
        # For the interested reader:
        # https://coneural.org/florian/papers/05_cart_pole.pdf
        temp = (F + self.polemass_length * theta_dot**2 * sintheta) / self.total_mass
        theta_dot_dot = (self.gravity * sintheta - costheta * temp) / (
            self.length * (4.0 / 3.0 - self.masspole * costheta**2 / self.total_mass)
        )
        x_dot_dot = temp - self.polemass_length * theta_dot_dot * costheta / self.total_mass
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


class Controlled3DOFAircraft(nn.Module):
    """
    Cart pole with a pendulum attached on it.
    """

    STATE_DIM = 3
    CONTROL_DIM = 1
    MODULO = th.tensor([2 * pi, float("nan"), 2 * pi])

    def __init__(
        self,
        u: Callable[[th.Tensor, th.Tensor, th.Tensor], th.Tensor],
    ):
        super().__init__()
        self.u = u  # controller (nn.Module)
        self.register_buffer("A", th.tensor([[-0.313, 56.7, 0], [-0.0139, -0.426, 0], [0, 56.7, 0]]), persistent=False)
        self.A: th.Tensor
        self.register_buffer("B", th.tensor([[0.232], [0.0203], [0]]), persistent=False)
        self.B: th.Tensor

    def forward(self, t: th.Tensor, X: th.Tensor) -> th.Tensor:
        F = self.u(t, X)
        X_dot = th.matmul(self.A, X.T).T + th.matmul(self.B, F.T).T
        return X_dot
