from typing import Callable, Optional

import torch as th
import torch.nn as nn

from utils.functions import get_modulo_parameters, tensor_encode_modulo_partial, tensor_modulo_partial_shuffle


class DiscreteCost(nn.Module):
    """Discrete cost function
    Args:
        x_star: th.tensor, target position
        P: float, terminal cost weights
        Q: float, state weights
        R: float, controller regulator weights
    """

    def __init__(
        self,
        x_star: th.Tensor,
        control_dim: int,
        t0: float,
        tf: float,
        u: Optional[Callable[[th.Tensor, th.Tensor, th.Tensor], th.Tensor]] = None,
        P: Optional[th.Tensor] = None,
        Q: Optional[th.Tensor] = None,
        R: Optional[th.Tensor] = None,
        modulo: Optional[th.Tensor] = None,
    ):
        super().__init__()
        self.u = u  # controller (nn.Module)
        self.t0 = t0
        self.tf = tf
        state_dim = x_star.nelement()

        if modulo is None:
            modulo = th.tensor([float("nan")] * state_dim)
        assert len(modulo) == state_dim, "`modulo` argument must be of size `state_dim`"
        self.register_buffer("modulo", modulo, persistent=False)
        self.modulo: th.Tensor

        # Pre-calculate some tensors for faster `forward()` calls
        not_mod, is_mod, mod_coef = get_modulo_parameters(modulo)
        self.register_buffer("not_mod", not_mod, persistent=False)
        self.not_mod: th.Tensor
        self.register_buffer("is_mod", is_mod, persistent=False)
        self.is_mod: th.Tensor
        self.register_buffer("mod_coef", mod_coef, persistent=False)
        self.mod_coef: th.Tensor

        if P is None:
            P = th.zeros(state_dim)
        if Q is None:
            Q = th.ones(state_dim)
        if R is None:
            R = th.zeros(control_dim)
        self.register_buffer("P", tensor_modulo_partial_shuffle(P, not_mod, is_mod), persistent=False)
        self.P: th.Tensor
        self.register_buffer("Q", tensor_modulo_partial_shuffle(Q, not_mod, is_mod), persistent=False)
        self.Q: th.Tensor
        self.register_buffer("R", R, persistent=False)
        self.R: th.Tensor

        x_star = tensor_encode_modulo_partial(x_star, not_mod, is_mod, mod_coef)
        self.register_buffer("x_star", x_star)
        self.x_star: th.Tensor

    def forward(self, t: th.Tensor, x: th.Tensor):
        """
        t: traversed timestamps (t_span)
        x: trajectory with shape (t_span, batch_size, state_dim)
        u: control input (t_span, batch_size, control_dim)
        """
        x = tensor_encode_modulo_partial(x, self.not_mod, self.is_mod, self.mod_coef)
        decay_factor = ((t - self.t0) / (self.tf - self.t0)).clamp(0.0, 1.0)
        decay_factor_normalized = decay_factor.mul(len(t) / decay_factor.sum()).unsqueeze(-1)
        cost = (x[-1] - self.x_star).mul(self.P).norm(p=2, dim=-1).mean()
        cost += (x - self.x_star).mul(self.Q).norm(p=2, dim=-1).mul(decay_factor_normalized).mean()
        if self.u is not None:
            cur_u = self.u(t, x)
            cost += (cur_u - 0).mul(self.R).norm(p=2, dim=-1).mul(decay_factor_normalized).mean()
        return cost
