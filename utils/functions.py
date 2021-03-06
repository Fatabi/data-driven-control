from math import pi
from typing import Tuple

import torch as th


class GradMod(th.autograd.Function):
    """
    We can implement our own custom autograd Functions by subclassing
    th.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """

    @staticmethod
    def forward(ctx, input, other):
        """
        In the forward pass we receive a Tensor containing the input and return a
        Tensor containing the output. You can cache arbitrary Tensors for use in the
        backward pass using the save_for_backward method.
        """
        result = th.remainder(input, other)
        ctx.save_for_backward(input, other)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        x, y = ctx.saved_variables
        return grad_output * 1, grad_output * th.neg(th.div(x, y, rounding_mode="trunc"))


def get_modulo_parameters(modulo: th.Tensor) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
    not_mod = modulo.isnan()
    is_mod = not_mod.logical_not()
    mod_coef = (2 * pi) / modulo[is_mod]
    return not_mod, is_mod, mod_coef


def tensor_encode_modulo_partial(X: th.Tensor, not_mod: th.Tensor, is_mod: th.Tensor, mod_coef: th.Tensor) -> th.Tensor:
    X_not_mod, X_is_mod = X[..., not_mod], X[..., is_mod]
    X_modulo = th.cat([X_not_mod, th.sin(X_is_mod * mod_coef), th.cos(X_is_mod * mod_coef)], dim=-1)
    return X_modulo


def tensor_modulo_partial_shuffle(X: th.Tensor, not_mod: th.Tensor, is_mod: th.Tensor) -> th.Tensor:
    X_not_mod, X_is_mod = X[..., not_mod], X[..., is_mod]
    X_modulo = th.cat([X_not_mod, X_is_mod, X_is_mod], dim=-1)
    return X_modulo
