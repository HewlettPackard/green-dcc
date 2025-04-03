import torch
from rl_components.agent_net import ActorNet, CriticNet

def test_actor_critic_forward():
    obs_dim = 10
    act_dim = 5
    batch_size = 4

    obs = torch.randn(batch_size, obs_dim)
    actor = ActorNet(obs_dim, act_dim, hidden_dim=64)
    critic = CriticNet(obs_dim, act_dim, hidden_dim=64)

    logits = actor(obs)
    assert logits.shape == (batch_size, act_dim)

    actions = torch.randint(0, act_dim, (batch_size,))
    q1, q2 = critic(obs, actions)
    assert q1.shape == (batch_size,)
    assert q2.shape == (batch_size,)
