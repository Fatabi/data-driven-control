from typing import List, Optional

import torch as th
import torch.nn as nn

from utils.functions import get_modulo_parameters, tensor_encode_modulo_partial
from utils.models import MLP


class NeuralController(nn.Module):
    def __init__(
        self,
        state_dim: int,
        control_dim: int,
        hidden_dims: List[int],
        gain: float = 100.0,
        modulo: Optional[th.Tensor] = None,
    ):
        super().__init__()
        self.gain = gain
        if modulo is None:
            modulo = th.tensor([float("nan")] * state_dim)
        assert len(modulo) == state_dim, "`modulo` argument must be of size `state_dim`"
        self.register_buffer("modulo", modulo)
        self.modulo: th.Tensor

        # Pre-calculate some tensors for faster `forward()` calls
        not_mod, is_mod, mod_coef = get_modulo_parameters(modulo)
        self.register_buffer("not_mod", not_mod, persistent=False)
        self.not_mod: th.Tensor
        self.register_buffer("is_mod", is_mod, persistent=False)
        self.is_mod: th.Tensor
        self.register_buffer("mod_coef", mod_coef, persistent=False)
        self.mod_coef: th.Tensor

        x_mod_dim = int(self.not_mod.sum()) + 2 * int(self.is_mod.sum())
        self.model = MLP(2 * x_mod_dim, control_dim, hidden_dims)
        self.normalizer = nn.Softsign()

    def forward(self, t: th.Tensor, x: th.Tensor, x_star: th.Tensor) -> th.Tensor:
        x_modulo = tensor_encode_modulo_partial(x, self.not_mod, self.is_mod, self.mod_coef)
        x_star_modulo = tensor_encode_modulo_partial(x_star, self.not_mod, self.is_mod, self.mod_coef).broadcast_to(
            x_modulo.shape
        )
        # x_modulo = th.where(self.modulo.isnan(), x, th.remainder(x, self.modulo))  # This is not smooth around 2*pi !!
        return self.normalizer(self.model(th.cat([x_modulo, x_star_modulo], dim=-1))) * self.gain


class ZeroController(nn.Module):
    def __init__(self, control_dim: int):
        super().__init__()
        self.control_dim = control_dim

    def forward(self, t: th.Tensor, x: th.Tensor) -> th.Tensor:
        return th.zeros(x.shape[0], self.control_dim, dtype=x.dtype, device=x.device)
