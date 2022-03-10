from math import pi
from typing import Callable, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch as th
import torch.nn as nn
from pytorch_lightning.callbacks.gradient_accumulation_scheduler import GradientAccumulationScheduler
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torchdyn.core import ODEProblem

from utils.functions import tensor_encode_modulo_partial
from utils.models import MLP

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

    def __init__(
        self,
        x_star: th.Tensor,
        control_dim: int,
        t0: float,
        tf: float,
        P: Optional[th.Tensor] = None,
        Q: Optional[th.Tensor] = None,
        R: Optional[th.Tensor] = None,
    ):
        super().__init__()
        self.register_buffer("x_star", x_star)
        self.x_star: th.Tensor
        self.t0 = t0
        self.tf = tf
        state_dim = x_star.nelement()
        if P is None:
            P = th.zeros(state_dim)
        if Q is None:
            Q = th.ones(state_dim)
        if R is None:
            R = th.zeros(control_dim)
        self.register_buffer("P", P, persistent=False)
        self.P: th.Tensor
        self.register_buffer("Q", Q, persistent=False)
        self.Q: th.Tensor
        self.register_buffer("R", R, persistent=False)
        self.R: th.Tensor

    def forward(self, t: th.Tensor, x: th.Tensor, u: Optional[th.Tensor] = None):
        """
        t: traversed timestamps (t_span)
        x: trajectory with shape (t_span, batch_size, state_dim)
        u: control input (t_span, batch_size, control_dim)
        """
        decay_factor = ((t - self.t0) / (self.tf - self.t0)).clamp(0.0, 1.0)
        decay_factor_normalized = decay_factor.mul(len(t) / decay_factor.sum()).unsqueeze(-1)
        cost = (x[-1] - self.x_star).mul(self.P).norm(p=2, dim=-1).mean()
        cost += (x - self.x_star).mul(self.Q).norm(p=2, dim=-1).mul(decay_factor_normalized).mean()
        if u is not None:
            cost += (u - 0).mul(self.R).norm(p=2, dim=-1).mul(decay_factor_normalized).mean()
        return cost


class NeuralController(nn.Module):
    def __init__(
        self,
        x_star: th.Tensor,
        state_dim: int,
        control_dim: int,
        hidden_dims: List[int],
        gain: float = 100.0,
        modulo: Optional[th.Tensor] = None,
    ):
        super().__init__()
        self.register_buffer("x_star", x_star)
        self.x_star: th.Tensor
        self.gain = gain
        if modulo is None:
            modulo = th.tensor([float("nan")] * state_dim)
        assert len(modulo) == state_dim, "`modulo` argument must be of size `state_dim`"
        self.register_buffer("modulo", modulo)
        self.modulo: th.Tensor

        # Pre-calculate some tensors for faster `forward()` calls
        self.register_buffer("not_mod", self.modulo.isnan(), persistent=False)
        self.not_mod: th.Tensor
        self.register_buffer("is_mod", self.not_mod.logical_not(), persistent=False)
        self.is_mod: th.Tensor
        self.register_buffer("mod_coef", (2 * pi) / self.modulo[self.is_mod], persistent=False)
        self.mod_coef: th.Tensor

        self.model = MLP(2 * (int(self.not_mod.sum()) + 2 * int(self.is_mod.sum())), control_dim, hidden_dims)
        self.normalizer = nn.Softsign()

    def forward(self, t: th.Tensor, x: th.Tensor) -> th.Tensor:
        x_modulo = tensor_encode_modulo_partial(x, self.not_mod, self.is_mod, self.mod_coef)
        x_star_modulo = tensor_encode_modulo_partial(self.x_star, self.not_mod, self.is_mod, self.mod_coef).broadcast_to(
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


class IntegralWReg(nn.Module):
    def __init__(self, sys: Callable[[th.Tensor, th.Tensor], th.Tensor], reg_coef: float = 1e-4):
        super().__init__()
        self.sys = sys
        self.reg_coef = reg_coef

    def forward(self, t: th.Tensor, x: th.Tensor) -> th.Tensor:
        loss = self.reg_coef * th.abs(self.sys(x)).sum(1)
        return loss


class CartPoleModel(pl.LightningModule):
    STATE_DIM = 4
    CONTROL_DIM = 1
    MODULO = th.tensor([float("nan"), float("nan"), float("nan"), float("nan")])

    def __init__(
        self,
        x_star: th.Tensor,
        t_span: th.Tensor,
        max_epochs: int,
        lr: float = 1e-3,
        reg_coef: float = 1e-4,
        P: Optional[th.Tensor] = None,
        Q: Optional[th.Tensor] = None,
        R: Optional[th.Tensor] = None,
    ):
        super().__init__()
        u = NeuralController(
            x_star, state_dim=self.STATE_DIM, control_dim=self.CONTROL_DIM, hidden_dims=[128], modulo=self.MODULO
        )
        # Controlled system
        sys = ControlledCartPole(u)
        self.sys = ODEProblem(sys, solver="dopri5", sensitivity="autograd", integral_loss=IntegralWReg(sys, reg_coef))
        self.register_buffer("t_span", t_span, persistent=False)
        self.t_span: th.Tensor
        self.max_epochs = max_epochs
        self.lr = lr
        self.cost_func = IntegralCost(x_star, self.CONTROL_DIM, t_span[0], t_span[-1], P, Q, R)

    def configure_optimizers(self):
        optim = Adam(self.parameters(), lr=self.lr, betas=(0.8, 0.99))
        sched = CosineAnnealingLR(optim, T_max=self.max_epochs, eta_min=1e-8)
        return [optim], [sched]

    def forward(self, x0: th.Tensor, t_span: Optional[th.Tensor] = None) -> th.Tensor:
        if t_span is None:
            t_span = self.t_span
        _, traj = self.sys.odeint(x0, t_span)
        return traj

    def training_step(self, batch: th.Tensor, batch_idx: int) -> th.Tensor:
        x0 = batch
        traj_t, traj = self.sys.odeint(x0, t_span)
        loss = self.cost_func(traj_t, traj)

        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch: th.Tensor, batch_idx: int):
        # the logger you used (in this case tensorboard)
        tensorboard: SummaryWriter = self.logger.experiment
        x0 = batch
        t0, tf = self.t_span[0].item(), self.t_span[-1].item()  # initial and final time for controlling the system
        steps = 40 * int(np.round(tf - t0)) + 1  # so we have a time step of 0.1s
        t_span_fine = th.linspace(t0, tf, steps).to(model.device)
        traj = self.forward(x0, t_span=t_span_fine)
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
        fig.canvas.draw()
        data = np.asarray(fig.canvas.renderer._renderer).take([0, 1, 2], axis=2)
        tensorboard.add_image("val_trajectories", th.from_numpy(data).permute([2, 0, 1]), global_step=self.global_step)
        plt.close(fig)


class TrajDataset(Dataset):
    def __init__(self, lb: List[float], ub: List[float], traj_cnt: int):
        super().__init__()
        self.lb = lb
        self.ub = ub
        self.traj_cnt = traj_cnt
        self.init_dist = th.distributions.Uniform(th.Tensor(lb), th.Tensor(ub))

    def __len__(self) -> int:
        return self.traj_cnt

    def __getitem__(self, _: int) -> th.Tensor:
        return self.init_dist.sample((1,)).flatten()


if __name__ == "__main__":
    pl.seed_everything(1234)
    # Loss function declaration
    x_star = th.Tensor([0.0, 0.0, pi, 0.0])

    # Time span
    t0, tf = 0, 4  # initial and final time for controlling the system
    steps = 10 * (tf - t0) + 1  # so we have a time step of 0.1s
    t_span = th.linspace(t0, tf, steps)
    # Hyperparameters
    lr = 3e-3
    max_epochs = 300
    batch_size = 128
    Q = th.tensor([1.0, 1.0, 10.0, 20.0])
    # Initial distribution
    lb = [-1, -0.1, 3 * pi / 4, -pi / 16]
    ub = [1, 0.1, 5 * pi / 4, pi / 16]
    train_dataset = TrajDataset(lb, ub, batch_size * 6)
    val_dataset = TrajDataset(lb, ub, batch_size)

    model = CartPoleModel(x_star, t_span, max_epochs, lr, Q=Q)
    trainer = pl.Trainer(
        gpus=[0], max_epochs=max_epochs, log_every_n_steps=2, callbacks=[GradientAccumulationScheduler({max_epochs // 2: 2})]
    )
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=2, persistent_workers=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, num_workers=2, persistent_workers=True)
    trainer.fit(model, train_dataloader, val_dataloader)

    # Testing the controller on the real system
    x0 = train_dataset.init_dist.sample((100,)).to(model.device)
    t0, tf = 0, 4  # initial and final time for controlling the system
    steps = 40 * (tf - t0) + 1  # so we have a time step of 0.1s
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
