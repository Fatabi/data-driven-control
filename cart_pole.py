import os
from math import pi
from typing import Callable, List, Optional

import matplotlib.pyplot as plt
import pandas as pd
import pytorch_lightning as pl
import torch as th
import torch.nn as nn
from pytorch_lightning.callbacks.gradient_accumulation_scheduler import GradientAccumulationScheduler
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from torchdyn.core import ODEProblem

from models.mlp import MLP

th.set_default_dtype(th.float64)


class ControlledCartPole(nn.Module):
    """
    Cart pole with a pendulum attached on it.
    """

    def __init__(self, u: Callable[[th.Tensor, th.Tensor], th.Tensor]):
        super().__init__()
        self.u = u  # controller (nn.Module)

        M = 0.5
        m = 0.2
        b = 0.1
        inertia = 0.006
        g = 9.8
        length = 0.3

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


class IntegralCost(nn.Module):
    """Integral cost function
    Args:
        x_star: th.tensor, target position
        P: float, terminal cost weights
        Q: float, state weights
        R: float, controller regulator weights
    """

    def __init__(self, x_star: th.Tensor, P: float = 0.0, Q: float = 1.0, R: float = 0.0):
        super().__init__()
        self.register_buffer("x_star", x_star)
        self.x_star: th.Tensor
        self.P, self.Q, self.R, = (
            P,
            Q,
            R,
        )

    def forward(self, x: th.Tensor, u: Optional[th.Tensor] = None):
        """
        x: trajectory
        u: control input
        """
        cost = self.P * th.norm(x[-1] - self.x_star, p=2, dim=-1).mean()
        cost += self.Q * th.norm(x - self.x_star, p=2, dim=-1).mean()
        if u is not None:
            cost += self.R * th.norm(u - 0, p=2).mean()
        return cost


class NeuralController(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dims: List[int], gain: float = 100.0):
        super().__init__()
        self.model = MLP(input_dim, output_dim, hidden_dims)
        self.gain = gain
        self.normalizer = nn.Softsign()

    def forward(self, t: th.Tensor, x: th.Tensor) -> th.Tensor:
        return self.normalizer(self.model(x)) * self.gain


class ZeroController(nn.Module):
    def __init__(self, output_dim: int):
        super().__init__()
        self.output_dim = output_dim

    def forward(self, t: th.Tensor, x: th.Tensor) -> th.Tensor:
        return th.zeros(x.shape[0], self.output_dim, dtype=x.dtype, device=x.device)


class IntegralWReg(nn.Module):
    def __init__(self, sys: Callable[[th.Tensor, th.Tensor], th.Tensor], reg_coef: float = 1e-4):
        super().__init__()
        self.sys = sys
        self.reg_coef = reg_coef

    def forward(self, t: th.Tensor, x: th.Tensor) -> th.Tensor:
        loss = self.reg_coef * th.abs(self.sys(x)).sum(1)
        return loss


class CartPoleModel(pl.LightningModule):
    def __init__(
        self,
        sys: ControlledCartPole,
        x_star: th.Tensor,
        t_span: th.Tensor,
        max_epochs: int,
        lr: float = 1e-3,
        reg_coef: float = 1e-4,
    ):
        super().__init__()
        self.sys = ODEProblem(sys, solver="dopri5", sensitivity="autograd", integral_loss=IntegralWReg(sys, reg_coef))
        self.register_buffer("x_star", x_star)
        self.x_star: th.Tensor
        self.register_buffer("t_span", t_span)
        self.t_span: th.Tensor
        self.max_epochs = max_epochs
        self.lr = lr
        self.cost_func = IntegralCost(x_star)

    def configure_optimizers(self):
        optim = Adam(self.parameters(), lr=self.lr, betas=(0.8, 0.99))
        sched = CosineAnnealingLR(optim, T_max=self.max_epochs, eta_min=1e-8)
        return [optim], [sched]

    def forward(self, x0: th.Tensor, t_span: Optional[th.Tensor] = None) -> th.Tensor:
        if t_span is None:
            t_span = self.t_span
        _, trajectory = self.sys.odeint(x0, t_span)
        return trajectory

    def training_step(self, batch: th.Tensor, batch_idx: int) -> th.Tensor:
        x0 = batch
        _, trajectory = self.sys.odeint(x0, t_span)
        loss = self.cost_func(trajectory)

        self.log("train_loss", loss)
        return loss


class TrajDataset(Dataset):
    def __init__(self, lb: List[float], ub: List[float], batch_size: int):
        super().__init__()
        self.lb = lb
        self.ub = ub
        self.batch_size = batch_size
        self.init_dist = th.distributions.Uniform(th.Tensor(lb), th.Tensor(ub))

    def __len__(self) -> int:
        return self.batch_size * 6

    def __getitem__(self, _: int) -> th.Tensor:
        return self.init_dist.sample((1,)).flatten()


if __name__ == "__main__":
    pl.seed_everything(1234)
    u = NeuralController(input_dim=4, output_dim=1, hidden_dims=[128])
    # Controlled system
    sys = ControlledCartPole(u)
    # Loss function declaration
    x_star = th.Tensor([0.0, 0.0, pi, 0.0])

    # Time span
    t0, tf = 0, 2  # initial and final time for controlling the system
    steps = 10 * tf + 1  # so we have a time step of 0.1s
    t_span = th.linspace(t0, tf, steps)
    # Hyperparameters
    lr = 3e-3
    max_epochs = 300
    batch_size = 256
    # Initial distribution
    lb = [-1, -0.5, -pi, -pi / 2]
    ub = [1, 0.5, pi, pi / 2]

    model = CartPoleModel(sys, x_star, t_span, max_epochs, lr)
    trainer = pl.Trainer(
        gpus=[0], max_epochs=max_epochs, log_every_n_steps=2, callbacks=[GradientAccumulationScheduler({max_epochs // 2: 2})]
    )
    dataset = TrajDataset(lb, ub, batch_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=max(os.cpu_count() // 2, 1), persistent_workers=True)
    trainer.fit(model, dataloader)

    # Testing the controller on the real system
    x0 = dataset.init_dist.sample((100,)).to(model.device)
    t0, tf = 0, 10  # initial and final time for controlling the system
    steps = 10 * tf + 1  # so we have a time step of 0.1s
    t_span_fine = th.linspace(t0, tf, steps).to(model.device)
    traj = model.forward(x0, t_span=t_span_fine)
    t_span_fine = t_span_fine.detach().cpu()
    traj = traj.detach().cpu()

    # Plotting
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    for i in range(len(x0)):
        ax.plot(t_span_fine, traj[:, i, 0], "k", alpha=0.3, label="x" if i == 0 else None)
        ax.plot(t_span_fine, traj[:, i, 1], "b", alpha=0.3, label="x_dot" if i == 0 else None)
        ax.plot(t_span_fine, traj[:, i, 2], "r", alpha=0.3, label="theta" if i == 0 else None)
        ax.plot(t_span_fine, traj[:, i, 3], "g", alpha=0.3, label="theta_dot" if i == 0 else None)
    ax.legend()
    ax.set_title("Controlled trajectories")
    ax.set_xlabel(r"$t~[s]$")
    plt.savefig("all_trajectories.png", dpi=300)

    # Exporting Datas
    traj_0 = th.cat([t_span_fine.unsqueeze(-1), traj[:, 0]], dim=-1)
    df = pd.DataFrame(traj_0)
    df.to_csv("./trajectory_0.csv", index=False)
