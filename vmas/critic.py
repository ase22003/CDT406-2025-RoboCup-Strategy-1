import torch.nn as nn


class CentralCritic(nn.Module):
    def __init__(self, total_obs_dim, n_agents):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(total_obs_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, n_agents)
        )

    def forward(self, x):  # x: [T, total_obs_dim]
        return self.net(x)  # [T, n_agents]
