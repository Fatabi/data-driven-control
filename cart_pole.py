from math import pi
from typing import List, Optional

import matplotlib.pyplot as plt
import pandas as pd
import pytorch_lightning as pl
import torch as th
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from torchdyn.numerics import odeint


class ControlledCartPole(nn.Module):
    """
    Inverted pendulum with torsional spring
    """

    def __init__(self, u):
        super().__init__()
        self.u = u  # controller (nn.Module)
        self.nfe = 0  # number of function evaluations
        self.cur_f = None  # current function evaluation
        self.cur_u = None  # current controller evaluation

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

    def forward(self, t, x):
        self.nfe += 1
        self.cur_u = self.u(t, x)
        self.cur_f = th.matmul(x, self.A.T).T.permute([1, 0]) + th.matmul(self.cur_u, self.B)
        return self.cur_f


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
    def __init__(self, model, u_min=-20, u_max=20):
        super().__init__()
        self.model = model
        self.u_min, self.u_max = u_min, u_max

    def forward(self, t, x):
        x = self.model(x)
        return th.clamp(x, self.u_min, self.u_max)


class CartPoleTrainer(pl.LightningModule):
    def __init__(self, sys: ControlledCartPole, x_star: th.Tensor, t_span: th.Tensor, max_epochs: int, lr: float = 1e-3):
        super().__init__()
        self.sys = sys
        self.x_star = x_star
        self.t_span = t_span
        self.max_epochs = max_epochs
        self.lr = lr
        self.cost_func = IntegralCost(x_star)

    def configure_optimizers(self):
        optim = Adam(self.parameters(), lr=self.lr)
        sched = CosineAnnealingLR(optim, T_max=self.max_epochs, eta_min=1e-8)
        return [optim], [sched]

    def forward(self, x0: th.Tensor, t_span: Optional[th.Tensor] = None) -> th.Tensor:
        if t_span is None:
            t_span = self.t_span
        _, trajectory = odeint(self.sys, x0, t_span, solver="dopri5", atol=1e-6, rtol=1e-6)
        return trajectory

    def training_step(self, batch: th.Tensor, batch_idx: int) -> th.Tensor:
        x0 = batch
        _, trajectory = odeint(self.sys, x0, self.t_span, solver="dopri5", atol=1e-6, rtol=1e-6)
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
        return self.batch_size * 5

    def __getitem__(self, _: int) -> th.Tensor:
        return self.init_dist.sample((1,)).flatten()


if __name__ == "__main__":
    control_model = nn.Sequential(nn.Linear(4, 32), nn.Tanh(), nn.Linear(32, 1))
    u = NeuralController(control_model)
    # Controlled system
    sys = ControlledCartPole(u)
    # Loss function declaration
    x_star = th.Tensor([0.0, 0.0, pi, 0.0])

    # Time span
    t0, tf = 0, 2  # initial and final time for controlling the system
    steps = 20 + 1  # so we have a time step of 0.1s
    t_span = th.linspace(t0, tf, steps)
    # Hyperparameters
    lr = 1e-3
    max_epochs = 300
    batch_size = 1024
    # Initial distribution
    lb = [-1, -2, -pi, -pi * 2]
    ub = [1, 2, pi, pi * 2]

    model = CartPoleTrainer(sys, x_star, t_span, max_epochs, lr)
    trainer = pl.Trainer(gpus=[0], max_epochs=max_epochs, log_every_n_steps=5)
    dataset = TrajDataset(lb, ub, batch_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=1, persistent_workers=True)
    trainer.fit(model, dataloader)

    # Testing the controller on the real system
    x0 = dataset.init_dist.sample((100,)).to(model.device)
    t0, tf, steps = 0, 2, 20 * 10 + 1
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

    # Exporting Datas
    traj_0 = th.cat([t_span_fine.unsqueeze(-1), traj[:, 0]], dim=-1)
    df = pd.DataFrame(traj_0)
    df.to_csv("./veriler.csv", index=False)
