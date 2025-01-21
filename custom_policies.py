import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy
from torch.nn import TransformerEncoder, TransformerEncoderLayer

from typing import Callable, Tuple
from gymnasium import spaces
import torch as th
from torch import nn

class TransformerFeatureExtractor(BaseFeaturesExtractor):
    """
    Custom feature extractor that interprets the flattened observation as
    (N data centers) x (M features), applies a Transformer, and returns
    a single feature vector for the policy and value network.
    """
    def __init__(
        self,
        observation_space,
        n_data_centers=3,
        n_features_per_dc=2,  # e.g. [curr_workload, ci]
        d_model=32,
        nhead=4,
        num_encoder_layers=2,
        features_dim=64
    ):
        """
        :param n_data_centers: Number of data centers (N).
        :param n_features_per_dc: Number of features per data center (M).
        :param d_model: Transformer embedding dimension.
        :param nhead: Number of attention heads.
        :param num_encoder_layers: Number of encoder layers in the Transformer.
        :param features_dim: Final output dimension for the feature extractor.
        """
        super().__init__(observation_space, features_dim)
        
        self.n_data_centers = n_data_centers
        self.n_features_per_dc = n_features_per_dc

        # We expect the raw observation to have shape (N * M,)
        expected_shape = n_data_centers * n_features_per_dc
        assert observation_space.shape[0] == expected_shape, (
            f"Observation dimension mismatch. Got: {observation_space.shape[0]}, "
            f"Expected: {expected_shape} (N*M). Please adjust n_data_centers or n_features_per_dc."
        )

        # 1) Simple linear embedding from M -> d_model
        self.embedding = nn.Linear(n_features_per_dc, d_model)

        # 2) Define a standard TransformerEncoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        # 3) Final linear to produce the desired feature_dim for SB3
        self.projection = nn.Linear(d_model, features_dim)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # observations.shape: [batch_size, N*M]
        batch_size = observations.shape[0]

        # Reshape to [batch_size, N, M]
        x = observations.view(batch_size, self.n_data_centers, self.n_features_per_dc)

        # Embed each DC’s features -> shape [batch_size, N, d_model]
        x = self.embedding(x)

        # Pass through the Transformer Encoder
        # shape remains [batch_size, N, d_model] (because we set batch_first=True)
        x = self.transformer_encoder(x)

        # We can pool across the N dimension. Let's do an average pool across data centers:
        # shape -> [batch_size, d_model]
        x = x.mean(dim=1)

        # Project to final desired features_dim -> [batch_size, features_dim]
        x = self.projection(x)

        return x


class ScalableTransformerFeatureExtractor(BaseFeaturesExtractor):
    """
    Custom Feature Extractor that structures observations for the Transformer.
    Assumes observations are flattened as [N, F], where N is variable.
    """
    def __init__(self, observation_space, n_features_per_dc=2, d_model=64, **kwargs):
        super(ScalableTransformerFeatureExtractor, self).__init__(observation_space, features_dim=d_model)
        
        self.n_features_per_dc = n_features_per_dc
        
        # Linear layer to embed features
        self.embedding = nn.Linear(n_features_per_dc, d_model)
        
        # Positional Encoding
        self.positional_encoding = PositionalEncoding(d_model)
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the feature extractor.
        Expects observations to be of shape [batch_size, N * F]
        """
        batch_size = observations.shape[0]
        total_features = observations.shape[1]
        n_data_centers = total_features // self.n_features_per_dc
        
        # Reshape to [batch_size, N, F]
        x = observations.view(batch_size, n_data_centers, self.n_features_per_dc)
        
        # Embed features
        x = self.embedding(x)  # [batch_size, N, d_model]
        
        # Apply positional encoding
        x = self.positional_encoding(x)  # [batch_size, N, d_model]
        
        return x

class PositionalEncoding(nn.Module):
    """
    Adds positional information to the input embeddings.
    """
    def __init__(self, d_model, max_len=100):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        :param x: [batch_size, N, d_model]
        :return: [batch_size, N, d_model]
        """
        N = x.size(1)
        x = x + self.pe[:, :N]
        return x

class ScalableAttentionPolicy(ActorCriticPolicy):
    """
    Custom Actor-Critic Policy with an attention mechanism to handle variable N.
    """
    def __init__(
        self,
        observation_space,
        action_space,
        lr_schedule,
        net_arch={
            'pi': [64, 64],  # Policy network layers
            'vf': [64, 64]   # Value network layers
        },
        activation_fn=nn.ReLU,
        features_extractor_class=BaseFeaturesExtractor,
        features_extractor_kwargs=None,
        n_heads=2,
        transformer_layers=3,
        dim_feedforward=16,
        **kwargs
    ):

        super(ScalableAttentionPolicy, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch=net_arch,
            activation_fn=activation_fn,
            features_extractor_class=features_extractor_class,
            features_extractor_kwargs=features_extractor_kwargs,
            **kwargs)
        
        # Transformer Encoder Setup
        encoder_layer = TransformerEncoderLayer(
            d_model=self.features_dim,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
        )
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=transformer_layers)
        
        # Action Head
        # We'll generate a single scalar action per data center, handled dynamically
        self.action_head = nn.Linear(self.features_dim, 1)
        
        # Critic Head
        self.critic_head = nn.Linear(self.features_dim, 1)
    
    def forward(self, obs, deterministic=False):
        """
        Forward pass for the policy network.
        """
        # Extract features
        features = self.extract_features(obs)  # [batch_size, N, features_dim]
        
        # Apply Transformer Encoder
        transformer_output = self.transformer_encoder(features)  # [batch_size, N, features_dim]
        
        # Generate actions for each data center
        action_logits = self.action_head(transformer_output)  # [batch_size, N, 1]
        action_mean = action_logits.squeeze(-1)  # [batch_size, N]
        
        # Define standard deviation (fixed for simplicity)
        action_std = torch.ones_like(action_mean) * 0.1  # [batch_size, N]
        
        # Create action distribution
        dist = torch.distributions.Normal(action_mean, action_std)
        
        # Sample actions
        if deterministic:
            actions = action_mean
        else:
            actions = dist.rsample()  # [batch_size, N]
        
        # Compute log_probs
        log_probs = dist.log_prob(actions).sum(dim=1)  # [batch_size]
        
        # Generate critic value
        critic_value = self.critic_head(transformer_output).mean(dim=1)  # [batch_size, features_dim]
        critic_value = critic_value.mean(dim=1, keepdim=True)  # [batch_size, 1]
        
        return actions, critic_value, log_probs
    
    def predict_values(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Get the estimated values according to the current policy given the observations.
        :param obs: Observation
        :return: the estimated values.
        """
        # Extract features
        features = self.extract_features(obs)  # [batch_size, N, features_dim]
        
        # Apply Transformer Encoder
        transformer_output = self.transformer_encoder(features)  # [batch_size, N, features_dim]
        
        # Generate critic value
        critic_value = self.critic_head(transformer_output).mean(dim=1)  # [batch_size, features_dim]
        critic_value = critic_value.mean(dim=1, keepdim=True)  # [batch_size, 1]
        
        return critic_value  # [batch_size, 1]


    def _distribution(self, obs):
        action_mean, action_std, _ = self.forward(obs)
        return th.distributions.Normal(action_mean, action_std)
    
    def evaluate_actions(self, obs, actions):
        """
        Evaluate actions and compute log probabilities, critic value, and entropy.
        """
        # Extract features
        features = self.extract_features(obs)  # [batch_size, N, features_dim]
        
        # Apply Transformer Encoder
        transformer_output = self.transformer_encoder(features)  # [batch_size, N, features_dim]
        
        # Generate action distribution parameters
        action_logits = self.action_head(transformer_output)  # [batch_size, N, 1]
        action_mean = action_logits.squeeze(-1)  # [batch_size, N]
        action_std = torch.ones_like(action_mean) * 0.1  # [batch_size, N]
        
        # Create action distribution
        dist = torch.distributions.Normal(action_mean, action_std)
        
        # Compute log_prob of provided actions
        log_prob = dist.log_prob(actions).sum(dim=1)  # Shape: [batch_size]
        
        # Compute entropy
        entropy = dist.entropy().sum(dim=1).mean()
        
        # Generate critic value
        critic_value = self.critic_head(transformer_output).mean(dim=1)  # [batch_size, features_dim]
        critic_value = critic_value.mean(dim=1, keepdim=True)  # [batch_size, 1]
        
        return log_prob, critic_value, entropy


class TransformerActorCriticPolicy(ActorCriticPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, net_arch=None, **kwargs):
        super(TransformerActorCriticPolicy, self).__init__(
            observation_space, action_space, lr_schedule, net_arch=net_arch, **kwargs
        )

        # Parameters
        self.num_attention_heads = 2
        self.transformer_hidden_dim = 64
        self.num_transformer_layers = 2
        self.n_data_centers = 3
        self.n_features_per_dc = 2
        self.features_dim = 2  # features_dim

        # Transformer model
        self.transformer_encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.features_dim,
            nhead=self.num_attention_heads,
            dim_feedforward=self.transformer_hidden_dim,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            self.transformer_encoder_layer, num_layers=self.num_transformer_layers
        )

        # Actor network
        self.actor_net = nn.Sequential(
            nn.Linear(self.features_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # Output logits for each action (N actions)
        )

        # Critic network
        self.critic_net = nn.Sequential(
            nn.Linear(self.features_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # Output a single scalar value
        )

    def forward(self, obs, deterministic=False):
        """
        Forward pass through the policy.
        """
        # observations.shape: [batch_size, N*M]
        batch_size = obs.shape[0]

        # Reshape to [batch_size, N, M]
        x = obs.view(batch_size, self.n_data_centers, self.n_features_per_dc)
        
        # Apply the transformer
        transformer_output = self.transformer_encoder(x)  # Output: [batch_size, N_data_centers, features_dim]

        # Actor: Process each data center independently
        action_logits = self.actor_net(transformer_output)  # Output: [batch_size, N_data_centers, 1]

        # Flatten the logits to prepare for action sampling
        action_logits_flat = action_logits.view(batch_size, -1)  # [batch_size, N_data_centers]

        # Create a probability distribution for actions
        action_probs = F.softmax(action_logits_flat, dim=-1)
        dist = torch.distributions.Categorical(action_probs)

        # Sample actions
        if deterministic:
            actions = torch.argmax(action_probs, dim=-1)
        else:
            actions = dist.sample()  # Sample actions from the distribution

        # Compute log probabilities of the actions
        log_probs = dist.log_prob(actions)

        # Critic: Pool features to compute a single scalar value
        pooled_features = transformer_output.mean(dim=1)  # Global mean pooling
        value = self.critic_net(pooled_features).squeeze(-1)  # Output: [batch_size]

        return actions, value, log_probs


    def _build(self, *args, **kwargs):
        # Ensure any needed initialization is handled here
        pass

