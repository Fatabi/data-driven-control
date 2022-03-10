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


def tensor_encode_modulo_partial(x: th.Tensor, not_mod: th.Tensor, is_mod: th.Tensor, mod_coef: th.Tensor) -> th.Tensor:
    x_not_mod, x_is_mod = x[..., not_mod], x[..., is_mod]
    x_modulo = th.cat([x_not_mod, th.sin(x_is_mod * mod_coef), th.cos(x_is_mod * mod_coef)], dim=-1)
    return x_modulo
