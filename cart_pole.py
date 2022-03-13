from math import pi
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch as th
from pytorch_lightning.callbacks.gradient_accumulation_scheduler import GradientAccumulationScheduler
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchdyn.core import ODEProblem

from utils.controllers import NeuralController
from utils.costs import DiscreteCost
from utils.datasets import IcDataset
from utils.regularizers import IntegralWReg
from utils.systems import ControlledCartPole

th.set_default_dtype(th.float64)
plt.ioff()


class CartPoleModel(pl.LightningModule):
    def __init__(
        self,
        X_star: th.Tensor,
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
            X_star=X_star,
            state_dim=ControlledCartPole.STATE_DIM,
            control_dim=ControlledCartPole.CONTROL_DIM,
            hidden_dims=[32],
            modulo=ControlledCartPole.MODULO,
        )
        # Controlled system
        sys = ControlledCartPole(u=u)
        self.sys = ODEProblem(sys, solver="dopri5", sensitivity="autograd", integral_loss=IntegralWReg(sys, reg_coef))
        self.register_buffer("t_span", t_span, persistent=False)
        self.t_span: th.Tensor
        self.max_epochs = max_epochs
        self.lr = lr
        self.cost_func = DiscreteCost(
            X_star=X_star,
            control_dim=ControlledCartPole.CONTROL_DIM,
            t0=t_span[0],
            tf=t_span[-1],
            P=P,
            Q=Q,
            R=R,
            # modulo=ControlledCartPole.MODULO,
        )

    def configure_optimizers(self):
        optim = Adam(self.parameters(), lr=self.lr, betas=(0.8, 0.99))
        sched = ReduceLROnPlateau(optim, "min", 0.25, 2)
        return {"optimizer": optim, "lr_scheduler": sched, "monitor": "train_loss"}

    def forward(self, x0: th.Tensor, t_span: Optional[th.Tensor] = None) -> Tuple[th.Tensor, th.Tensor]:
        if t_span is None:
            t_span = self.t_span
        traj_t, traj = self.sys.odeint(x0, t_span)
        return traj_t, traj

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
        steps = 100 * int(np.round(tf - t0)) + 1  # so we have a time step of 0.01s
        t_span_fine = th.linspace(t0, tf, steps).to(model.device)
        t_span_fine, traj = self.forward(x0, t_span=t_span_fine)
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


if __name__ == "__main__":
    pl.seed_everything(1234)
    # Loss function declaration
    X_star = th.Tensor([0.0, 0.0, pi, 0.0])

    # Time span
    t0, tf = 0, 2  # initial and final time for controlling the system
    steps = 100 * (tf - t0) + 1  # so we have a time step of 0.01s
    t_span = th.linspace(t0, tf, steps)
    # Hyperparameters
    lr = 1e-2
    reg_coef = 0.0
    max_epochs = 300
    batch_size = 128
    Q = th.tensor([1.0, 1.0, 2.0, 1.0])
    # Initial distribution
    lb = [-0.05, -0.05, pi - 0.05, -0.05]
    ub = [0.05, 0.05, pi + 0.05, 0.05]
    train_dataset = IcDataset(lb=lb, ub=ub, ic_cnt=batch_size * 6)
    val_dataset = IcDataset(lb=lb, ub=ub, ic_cnt=batch_size)

    model = CartPoleModel(X_star=X_star, t_span=t_span, max_epochs=max_epochs, lr=lr, reg_coef=reg_coef, Q=Q)
    trainer = pl.Trainer(
        gpus=[0], max_epochs=max_epochs, log_every_n_steps=2, callbacks=[GradientAccumulationScheduler({max_epochs // 2: 2})]
    )
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=1, persistent_workers=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, num_workers=1, persistent_workers=True)
    trainer.fit(model, train_dataloader, val_dataloader)

    # Testing the controller on the real system
    x0 = train_dataset.init_dist.sample((100,)).to(model.device)
    t0, tf = 0, 4  # initial and final time for controlling the system
    steps = 100 * (tf - t0) + 1  # so we have a time step of 0.01s
    t_span_fine = th.linspace(t0, tf, steps).to(model.device)
    t_span_fine, traj = model.forward(x0, t_span=t_span_fine)
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
