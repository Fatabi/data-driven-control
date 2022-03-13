from typing import List

import torch as th
import torch.nn as nn
import torch.nn.init as init


class Linear(nn.Linear):
    def reset_parameters(self):
        init.orthogonal_(self.weight)
        if self.bias is not None:
            init.zeros_(self.bias)


class MLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dims: List[int]):
        super().__init__()
        layers = []
        for hidden_dim in hidden_dims:
            layers.append(Linear(input_dim, hidden_dim))
            layers.append(nn.PReLU(hidden_dim, 1.0))
            input_dim = hidden_dim
        layers.append(Linear(input_dim, output_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, input: th.Tensor) -> th.Tensor:
        return self.layers(input)
