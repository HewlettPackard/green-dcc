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
        logits = self.forward(obs_batch)  # [T, obs_dim]
        probs = F.softmax(logits, dim=-1)  # [T, obs_dim]
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
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, act_dim)
        )
        self.q2 = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
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
        
        assert ((actions >= 0) & (actions < q1_all.shape[1])).all(), "Action index out of bounds"

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
    
# TaskEncoder can remain the same if you use it
class TaskEncoder(nn.Module):
    def __init__(self, obs_dim_per_task, embed_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim_per_task, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, embed_dim)
        )
        self.norm = nn.LayerNorm(embed_dim)
    def forward(self, task_obs_list):
        encoded_tasks = self.encoder(task_obs_list)
        return self.norm(encoded_tasks)

class SimpleAttentionBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1, ff_dim_multiplier=4):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * ff_dim_multiplier),
            nn.ReLU(),
            nn.Linear(embed_dim * ff_dim_multiplier, embed_dim)
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, key_padding_mask=None):
        # x shape: [batch_size, num_tasks, embed_dim]
        # key_padding_mask shape: [batch_size, num_tasks] (True for padded)

        # Self-attention
        attn_output, _ = self.attention(x, x, x, key_padding_mask=key_padding_mask, need_weights=False)
        x = x + self.dropout(attn_output) # Add & Norm (Residual connection)
        x = self.norm1(x)

        # Feed-forward
        ffn_output = self.ffn(x)
        x = x + self.dropout(ffn_output) # Add & Norm
        x = self.norm2(x)
        return x
    
class AttentionActorNet(nn.Module):
    def __init__(self, obs_dim_per_task, act_dim, embed_dim=128, num_heads=4, num_attention_layers=2, dropout=0.1):
        super().__init__()
        self.act_dim = act_dim
        self.embed_dim = embed_dim # Store for processing batch function

        self.task_encoder = TaskEncoder(obs_dim_per_task, embed_dim)
        
        self.attention_layers = nn.ModuleList(
            [SimpleAttentionBlock(embed_dim, num_heads, dropout) for _ in range(num_attention_layers)]
        )
        self.output_head = nn.Linear(embed_dim, act_dim)

    def _process_embeddings_with_attention(self, task_embeddings_batch, key_padding_mask_batch):
        # task_embeddings_batch: [B, T_max, embed_dim]
        # key_padding_mask_batch: [B, T_max] (True for padded)
        x = task_embeddings_batch
        for layer in self.attention_layers:
            x = layer(x, key_padding_mask=key_padding_mask_batch)
        return x # [B, T_max, embed_dim]

    def forward(self, obs_input, key_padding_mask=None):
        """
        obs_input: Can be [k_t, obs_dim_per_task] for single step action selection
                   OR [B, T_max, obs_dim_per_task] for batch processing from buffer.
        key_padding_mask: [B, T_max] if obs_input is batched and padded.
                          None if obs_input is [k_t, obs_dim_per_task] (all valid).
        """
        is_batched = obs_input.ndim == 3
        
        if not is_batched: # Single set of k_t tasks
            if obs_input.numel() == 0: return torch.empty(0, self.act_dim, device=obs_input.device)
            task_embeddings = self.task_encoder(obs_input) # [k_t, embed_dim]
            # For single set, treat as batch_size=1 for attention layers
            processed_embeddings = self._process_embeddings_with_attention(
                task_embeddings.unsqueeze(0), # [1, k_t, embed_dim]
                key_padding_mask.unsqueeze(0) if key_padding_mask is not None else None # [1, k_t]
            ).squeeze(0) # Back to [k_t, embed_dim]
        else: # Batched input: [B, T_max, obs_dim_per_task]
            B, T_max, _ = obs_input.shape
            if T_max == 0 : return torch.empty(B, 0, self.act_dim, device=obs_input.device)

            obs_flat = obs_input.reshape(B * T_max, -1)
            task_embeddings_flat = self.task_encoder(obs_flat) # [B*T_max, embed_dim]
            task_embeddings_batch = task_embeddings_flat.view(B, T_max, self.embed_dim)
            
            processed_embeddings_batch = self._process_embeddings_with_attention(
                task_embeddings_batch, 
                key_padding_mask_batch=key_padding_mask
            ) # [B, T_max, embed_dim]
            # Flatten for output head to get per-task logits
            processed_embeddings = processed_embeddings_batch.reshape(B * T_max, self.embed_dim)

        logits = self.output_head(processed_embeddings) # [k_t or B*T_max, act_dim]
        return logits

    def get_dist(self, obs_input, key_padding_mask=None): # Renamed from sample_actions for consistency
        logits = self.forward(obs_input, key_padding_mask)
        if logits.numel() == 0: return None
        return torch.distributions.Categorical(logits=logits)

class AttentionCriticNet(nn.Module):
    def __init__(self, obs_dim_per_task, act_dim, embed_dim=128, num_heads=4, num_attention_layers=2, dropout=0.1):
        super().__init__()
        self.act_dim = act_dim
        self.embed_dim = embed_dim

        self.task_encoder = TaskEncoder(obs_dim_per_task, embed_dim)
        self.attention_layers = nn.ModuleList(
            [SimpleAttentionBlock(embed_dim, num_heads, dropout) for _ in range(num_attention_layers)]
        )
        # Q-heads operate on the processed embeddings
        self.q1_head = nn.Linear(embed_dim, act_dim)
        self.q2_head = nn.Linear(embed_dim, act_dim)

    def _process_embeddings_with_attention(self, task_embeddings_batch, key_padding_mask_batch):
        # (Identical to the one in AttentionActorNet)
        x = task_embeddings_batch
        for layer in self.attention_layers:
            x = layer(x, key_padding_mask=key_padding_mask_batch)
        return x

    def forward_all(self, obs_input, key_padding_mask=None):
        """ Returns all Q-values for all actions for each task """
        is_batched = obs_input.ndim == 3
        
        if not is_batched:
            if obs_input.numel() == 0:
                return (torch.empty(0, self.act_dim, device=obs_input.device),
                        torch.empty(0, self.act_dim, device=obs_input.device))
            task_embeddings = self.task_encoder(obs_input) # [k_t, embed_dim]
            processed_embeddings = self._process_embeddings_with_attention(
                task_embeddings.unsqueeze(0),
                key_padding_mask.unsqueeze(0) if key_padding_mask is not None else None
            ).squeeze(0) # [k_t, embed_dim]
        else: # Batched input
            B, T_max, _ = obs_input.shape
            if T_max == 0:
                return (torch.empty(B, 0, self.act_dim, device=obs_input.device).view(B*T_max, self.act_dim), # Ensure correct shape for empty
                        torch.empty(B, 0, self.act_dim, device=obs_input.device).view(B*T_max, self.act_dim))

            obs_flat = obs_input.reshape(B * T_max, -1)
            task_embeddings_flat = self.task_encoder(obs_flat)
            task_embeddings_batch = task_embeddings_flat.view(B, T_max, self.embed_dim)
            processed_embeddings_batch = self._process_embeddings_with_attention(
                task_embeddings_batch,
                key_padding_mask_batch=key_padding_mask
            ) # [B, T_max, embed_dim]
            processed_embeddings = processed_embeddings_batch.reshape(B * T_max, self.embed_dim)

        q1_all_actions = self.q1_head(processed_embeddings) # [k_t or B*T_max, act_dim]
        q2_all_actions = self.q2_head(processed_embeddings) # [k_t or B*T_max, act_dim]
        return q1_all_actions, q2_all_actions

    def forward(self, obs_input, actions_list_flat_or_single, key_padding_mask=None):
        """
        obs_input: [k_t, D_obs] or [B, T_max, D_obs]
        actions_list_flat_or_single: [k_t] or [B*T_max] (containing valid actions for unpadded items, -1 for padded)
        key_padding_mask: [B, T_max] if batched
        """
        q1_all_actions_flat, q2_all_actions_flat = self.forward_all(obs_input, key_padding_mask)
        # q1_all/q2_all will be shape [k_t, act_dim] or [B*T_max, act_dim]

        if q1_all_actions_flat.numel() == 0:
            return (torch.empty(0, device=obs_input.device), torch.empty(0, device=obs_input.device))

        # Gather expects actions to be indices. Padded actions are -1.
        # We must only gather for valid actions and mask out results for padded ones.
        # The actual selection for loss calculation is done in the training loop after masking.
        # Here, we just provide the Q-values for the taken actions (where actions are valid).
        # The caller (training loop) should handle masking based on valid_idx.
        
        # Clamp actions to be valid indices for gather, padded actions will be ignored later by mask
        actions_clamped = actions_list_flat_or_single.clamp(0, self.act_dim - 1)
        
        q1 = q1_all_actions_flat.gather(1, actions_clamped.unsqueeze(-1)).squeeze(-1)
        q2 = q2_all_actions_flat.gather(1, actions_clamped.unsqueeze(-1)).squeeze(-1)
        
        return q1, q2