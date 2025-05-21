import torch
from torch import nn
from torch.distributions import Categorical, Normal


class HighLevelPolicy(nn.Module):
    def __init__(self, obs_dim, n_agents, hidden_dim=128, param_std=0.1):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mode_head = nn.Linear(hidden_dim, 4)
        self.param_head = nn.Linear(hidden_dim, 2 + 1 + n_agents)
        self.param_std = param_std  # fixed std dev for params

    def forward(self, obs):
        h = self.encoder(obs)  # [B, H]

        # Discrete mode distribution
        mode_logits = self.mode_head(h)  # [B, 4]
        mode_dist = Categorical(logits=mode_logits)
        mode = mode_dist.sample()  # [B]
        logp_mode = mode_dist.log_prob(mode)  # [B]

        # Continuous params distribution
        param_mean = self.param_head(h)  # [B, P]
        param_dist = Normal(param_mean, self.param_std)
        params = param_dist.rsample()  # [B, P]
        logp_params = param_dist.log_prob(params).sum(dim=-1)  # [B]

        # Combined log‐prob (discrete + continuous)
        logp = logp_mode + logp_params  # [B]

        return mode, params, logp.unsqueeze(-1)  # [B,1]
