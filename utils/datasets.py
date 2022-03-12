from typing import List

import torch as th
from torch.utils.data import Dataset


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
