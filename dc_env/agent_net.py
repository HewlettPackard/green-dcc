import torch
import torch.nn as nn
import torch.nn.functional as F

class ActorNet(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, act_dim)  # one output per DC
        )

    def forward(self, obs_batch):
        """
        obs_batch: [T, obs_dim]
        Returns: logits [T, act_dim]
        """
        return self.net(obs_batch)

    def sample_actions(self, obs_batch):
        logits = self.forward(obs_batch)  # [T, act_dim]
        probs = F.softmax(logits, dim=-1)  # [T, act_dim]
        dist = torch.distributions.Categorical(probs)
        actions = dist.sample()            # [T]
        log_probs = dist.log_prob(actions)  # [T]
        entropy = dist.entropy().mean()     # scalar, average over T
        return actions, log_probs, entropy


class CriticNet(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim=256):
        super().__init__()
        self.q1 = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, act_dim)
        )
        self.q2 = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, act_dim)
        )

    def forward(self, obs_batch, actions):
        """
        Compute Q-values for the given (obs, action) pairs.
        Returns the mean Q for q1 and q2 across tasks.
        """
        q1_all = self.q1(obs_batch)  # [T, act_dim]
        q2_all = self.q2(obs_batch)  # [T, act_dim]

        q1 = q1_all.gather(1, actions.unsqueeze(-1)).squeeze(-1)  # [T]
        q2 = q2_all.gather(1, actions.unsqueeze(-1)).squeeze(-1)  # [T]

        # q1_mean = q1.mean()
        # q2_mean = q2.mean()
        return q1, q2

    def forward_all(self, obs_batch):
        """
        Returns full Q-values for all actions. Used during target computation.
        """
        return self.q1(obs_batch), self.q2(obs_batch)  # [T, act_dim] each

