import time
from math import pi

import matplotlib.pyplot as plt
import pandas as pd
import torch as th

# from torchdyn.core import NeuralODE
# from torchdyn.numerics import Euler, HyperEuler, odeint
# from torchdyn.datasets import *
from torchdyn.numerics import odeint

from utils.controllers import NeuralController
from utils.costs import DiscreteCost
from utils.systems import ControlledPendulum

device = th.device("cuda:0") if th.cuda.is_available() else th.device("cpu")

# Loss function declaration
x_star = th.Tensor([0.0, 0.0]).to(device)
# Time span
t0, tf = 0, 2  # initial and final time for controlling the system
steps = 10 * (tf - t0) + 1  # so we have a time step of 0.1s
t_span = th.linspace(t0, tf, steps).to(device)
# Initial distribution
x0 = pi  # limit of the state distribution (in rads and rads/second)
init_dist = th.distributions.Uniform(th.Tensor([-x0, -x0]), th.Tensor([x0, x0]))
# Hyperparameters
lr = 3e-3
epochs = 500
bs = 1024

# The controller is a simple MLP with one hidden layer with bounded output
u = NeuralController(
    x_star=x_star,
    state_dim=ControlledPendulum.STATE_DIM,
    control_dim=ControlledPendulum.CONTROL_DIM,
    hidden_dims=[32],
    modulo=ControlledPendulum.MODULO,
)
# Controlled system
sys = ControlledPendulum(u=u).to(device)
opt = th.optim.Adam(u.parameters(), lr=lr)

cost_func = DiscreteCost(
    x_star=x_star, control_dim=ControlledPendulum.CONTROL_DIM, t0=t_span[0], tf=t_span[-1], modulo=ControlledPendulum.MODULO
)
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
t_span_fine = th.linspace(t0, tf, steps).to(device)
_, traj = odeint(sys, x0, t_span_fine, solver="dopri5", atol=1e-7, rtol=1e-7)
t_span_fine = t_span_fine.detach().cpu()
traj = traj.detach().cpu()

# Plotting
fig, ax = plt.subplots(1, 1, figsize=(12, 6))
for i in range(len(x0)):
    ax.plot(t_span_fine, traj[:, i, 0], "k-.", alpha=0.3, label="p" if i == 0 else None)
    ax.plot(t_span_fine, traj[:, i, 1], "b", alpha=0.3, label="q" if i == 0 else None)
ax.legend()
ax.set_title("Controlled trajectories")
ax.set_xlabel(r"$t~[s]$")

# Exporting Datas
traj_0 = th.cat([t_span_fine.unsqueeze(-1), traj[:, 0]], dim=-1)
df = pd.DataFrame(traj_0)
df.to_csv("./veriler.csv", index=False)
