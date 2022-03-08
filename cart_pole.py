import time
from math import pi

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn

# from torchdyn.core import NeuralODE
# from torchdyn.numerics import Euler, HyperEuler, odeint
# from torchdyn.datasets import *
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
        I = 0.006
        g = 9.8
        l = 0.3

        p = I * (M + m) + M * m * l**2  # denominator for the A and B matrices

        self.A = torch.tensor(
            [
                [0, 1, 0, 0],
                [0, -(I + m * l**2) * b / p, (m**2 * g * l**2) / p, 0],
                [0, 0, 0, 1],
                [0, -(m * l * b) / p, m * g * l * (M + m) / p, 0],
            ]
        )
        self.B = torch.tensor([[0, (I + m * l**2) / p, 0, m * l / p]])

    def forward(self, t, x):
        self.nfe += 1
        self.cur_u = self.u(t, x)
        self.cur_f = torch.matmul(x, self.A.T).T.permute([1, 0]) + torch.matmul(self.cur_u, self.B)
        return self.cur_f


class IntegralCost(nn.Module):
    """Integral cost function
    Args:
        x_star: torch.tensor, target position
        P: float, terminal cost weights
        Q: float, state weights
        R: float, controller regulator weights
    """

    def __init__(self, x_star, P=0, Q=1, R=0):
        super().__init__()
        self.x_star = x_star
        self.P, self.Q, self.R, = (
            P,
            Q,
            R,
        )

    def forward(self, x, u=torch.Tensor([0.0])):
        """
        x: trajectory
        u: control input
        """
        cost = self.P * torch.norm(x[-1] - self.x_star, p=2, dim=-1).mean()
        cost += self.Q * torch.norm(x - self.x_star, p=2, dim=-1).mean()
        cost += self.R * torch.norm(u - 0, p=2).mean()
        return cost


device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
# The controller is a simple MLP with one hidden layer with bounded output


class NeuralController(nn.Module):
    def __init__(self, model, u_min=-20, u_max=20):
        super().__init__()
        self.model = model
        self.u_min, self.u_max = u_min, u_max

    def forward(self, t, x):
        x = self.model(x)
        return torch.clamp(x, self.u_min, self.u_max)


model = nn.Sequential(nn.Linear(4, 32), nn.Tanh(), nn.Linear(32, 1)).to(device)
u = NeuralController(model)
for p in u.model[-1].parameters():
    torch.nn.init.zeros_(p)

# Controlled system
sys = ControlledCartPole(u).to(device)


# Loss function declaration
x_star = torch.Tensor([0.0, 0.0, pi, 0.0]).to(device)
cost_func = IntegralCost(x_star)

# Time span
t0, tf = 0, 2  # initial and final time for controlling the system
steps = 20 + 1  # so we have a time step of 0.1s
t_span = torch.linspace(t0, tf, steps).to(device)

# Initial distribution
min_bound = [-1, -2, -pi, -pi * 2]
max_bound = [1, 2, pi, pi * 2]

init_dist = torch.distributions.Uniform(torch.Tensor(min_bound), torch.Tensor(max_bound))
# Hyperparameters
lr = 3e-3
epochs = 500
bs = 1024
opt = torch.optim.Adam(u.parameters(), lr=lr)

# Training loop
t0 = time.time()
losses = []
for e in range(epochs):
    x0 = init_dist.sample((bs,)).to(device)
    _, trajectory = odeint(sys, x0, t_span, solver="dopri5", atol=1e-5, rtol=1e-5)
    loss = cost_func(trajectory)
    losses.append(loss.detach().cpu().item())
    loss.backward()
    opt.step()
    opt.zero_grad()
    print("Loss {:.4f} , epoch {}".format(loss.item(), e), end="\r")
timing = time.time() - t0
print("\nTraining time: {:.4f} s".format(timing))

# Testing the controller on the real system
x0 = init_dist.sample((100,)).to(device)
t0, tf, steps = 0, 2, 20 * 10 + 1
t_span_fine = torch.linspace(t0, tf, steps).to(device)
_, traj = odeint(sys, x0, t_span_fine, solver="dopri5", atol=1e-5, rtol=1e-5)
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
traj_0 = torch.cat([t_span_fine.unsqueeze(-1), traj[:, 0]], dim=-1)
df = pd.DataFrame(traj_0)
df.to_csv("./veriler.csv", index=False)
