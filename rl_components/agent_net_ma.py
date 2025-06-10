"""
• **ManagerNet**  attention scorer that picks **which data-center (DC)** should
  run a meta-task.  It supports an *automatic* **padding mask**: if your input
  tensor already reserves space for `num_clusters` DCs (e.g. 10) but the current
  scenario only has `active_clusters` (e.g. 3), simply pass that integer and the
  remaining padded DC slots will be hard-masked (logits = −∞).

• **WorkerNet**  classifier that decides **execute now** vs **defer** for the
  local queue.

Both networks are framework-agnostic and can be dropped into any PyTorch
training loop or higher-level library.
"""

import torch 
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple

class MLP(nn.Module):
    """Two layer MLP with optional LayerNorm."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, *, layer_norm: bool = False):
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(in_dim, hidden_dim)]
        if layer_norm:
            layers.append(nn.LayerNorm(hidden_dim))
        layers += [nn.ReLU(), nn.Linear(hidden_dim, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.net(x)
    

class AttentionModule(nn.Module):
    """Self-attention over option set (mask aware)."""

    def __init__(self, emb_dim: int, num_layers: int = 2, num_heads: int = 4, ff_dim: int = 256):
        super().__init__()
        layer = nn.TransformerEncoderLayer(d_model=emb_dim, nhead=num_heads,
                                               dim_feedforward=ff_dim, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)

    def forward(self, seq_emb_option_initial: torch.Tensor, 
                all_options_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.encoder(seq_emb_option_initial, src_key_padding_mask=all_options_padding_mask)
    
class ManagerActor(nn.Module):
    """
    Attention-based policy π-manager actor
    """
    def __init__(self, 
                 D_emb_meta_manager: int, 
                 D_global: int, 
                 D_option_feat: int, 
                 max_total_options, 
                 *, hidden_dim: int=128):
        
        super().__init__()
        self.max_total_options = max_total_options
        self.D_option_feat = D_option_feat

        self.attn = AttentionModule(D_option_feat)
        self.query = MLP(D_emb_meta_manager +  D_global, hidden_dim, D_option_feat)
        self.scorer = MLP(D_option_feat * 2, hidden_dim, 1)

    def forward(self, 
                emb_meta_task_mgr,              #(B, D_meta_manager)
                emb_global_context_mgr,         #(B, D_global)
                obs_all_options_set_padded,     #(B, max_total_options, D_option_feat)
                all_options_padding_mask ):     #(B, max_total_options)
        
        B, max_total_options, D_option_feat = obs_all_options_set_padded.shape

        seq_emb_options_contextual = self.attn(obs_all_options_set_padded, all_options_padding_mask)
        query = self.query(torch.cat([emb_meta_task_mgr, emb_global_context_mgr], dim = 1)) 
        query_expanded = query.unsqueeze(1).expand(-1, max_total_options, -1)  #(B, max_total_options, D_option_feat)
        fused = torch.cat([query_expanded, seq_emb_options_contextual], -1)
        logits = self.scorer(fused).squeeze(-1) #(B, max_total_options)

        if all_options_padding_mask is not None:
            logits = logits.masked_fill(all_options_padding_mask, float('-inf'))

        return logits
    
    def sample_action(self, emb_meta_task_mgr,             
                emb_global_context_mgr,         
                obs_all_options_set_padded,     
                all_options_padding_mask):
        """
        Samples actions, log_probs, and entropy from the policy distribution.
        This method is useful for on-policy algorithms or for evaluation.
        For SAC, typically only forward() is called and distribution handled externally.
        """

        logits = self.forward(emb_meta_task_mgr,             
                emb_global_context_mgr,         
                obs_all_options_set_padded,     
                all_options_padding_mask)
        
        probs = F.softmax(logits, dim=-1) # Softmax to get probabilities
        dist = torch.distributions.Categorical(probs=probs) # Use probs for Categorical
        actions = dist.sample()
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy().mean() # Average entropy over the batch of tasks
        return actions, log_probs, entropy
    
class ManagerCritic(nn.Module):
    """
    Twin-Q critic for π-Manager
      • q_values_all_options  = forward_q_values(...)
      • q1,q2 for chosen idx  = q_for_action(...)
    """
    def __init__(self,
                 D_emb_meta_manager: int, 
                 D_global: int, 
                 D_option_feat: int, 
                 max_total_options, 
                 *, hidden_dim: int=128):
        
        super().__init__()
        self.max_total_options = max_total_options
        self.D_option_feat = D_option_feat
        self.attn = AttentionModule(D_option_feat)
        self.query = MLP(D_emb_meta_manager +  D_global, hidden_dim, D_option_feat)
        self.Q1 = MLP(D_option_feat * 2, hidden_dim, 1)
        self.Q2 = MLP(D_option_feat * 2, hidden_dim, 1) 


    def forward_q_values(self,
                         emb_meta_task_mgr,             
                         emb_global_context_mgr,         
                         obs_all_options_set_padded,   
                         all_options_padding_mask):
        """
        return: q1_all, q2_all  →  (B, max_total_options)
        """
        seq_emb_options_contextual   = self.attn(obs_all_options_set_padded, all_options_padding_mask)                                   
        query = self.query(torch.cat([emb_meta_task_mgr, emb_global_context_mgr], -1))                
        query = query.unsqueeze(1).expand(-1, seq_emb_options_contextual.size(1), -1)          
        fused = torch.cat([query, seq_emb_options_contextual], -1)
        q1_all = self.Q1(fused).squeeze(-1)
        q2_all = self.Q2(fused).squeeze(-1)
        return q1_all, q2_all
    
    def q_for_action(self,
                     emb_meta_task_mgr,             
                     emb_global_context_mgr, 
                     obs_all_options_set_padded,
                     action_idx,                       # LongTensor (B,)
                     all_options_padding_mask=None):

        q1_all, q2_all = self.forward_q_values(
            emb_meta_task_mgr,             
            emb_global_context_mgr,
            obs_all_options_set_padded,
            all_options_padding_mask)

        q_selected = lambda q: q.gather(1, action_idx.unsqueeze(-1)).squeeze(-1)
        return q_selected(q1_all), q_selected(q2_all)  
    

class WorkerActor(nn.Module):
    """
    MLP-based policy π-worker actor
    """
    def __init__(self, D_meta_worker: int, D_local_worker: int, D_global: int,  *, hidden_dim: int=128):
        super().__init__()
        in_dim = D_meta_worker + D_local_worker + D_global
        self.mlp = MLP(in_dim, hidden_dim, 2)

    def forward(self, obs_worker_meta_task_i, obs_local_dc_i_for_worker, obs_global_context):
        combined_input = torch.cat([obs_worker_meta_task_i, obs_local_dc_i_for_worker,obs_global_context], -1)
        logits_worker_action = self.mlp(combined_input)

        return logits_worker_action
    
    def action_sampling(self, obs_worker_meta_task_i, obs_local_dc_i_for_worker, obs_global_context):
        logits = self.forward(obs_worker_meta_task_i, obs_local_dc_i_for_worker, obs_global_context)
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs=probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy().mean()

        return action, log_prob, entropy
    

class WorkerCritic(nn.Module):
    """
    Twin-Q critic for π-worker
    """

    def __init__(self, D_meta_worker: int, D_local_worker: int, D_global: int,  *, hidden_dim: int=128):
        super().__init__()
        in_dim = D_meta_worker + D_local_worker + D_global
        self.Q1 = MLP(in_dim, hidden_dim, 2)
        self.Q2 = MLP(in_dim, hidden_dim, 2)

    def forward_q_value(self, obs_worker_meta_task_i, obs_local_dc_i_for_worker, obs_global_context):
        combined_input = torch.cat([obs_worker_meta_task_i, obs_local_dc_i_for_worker,obs_global_context], -1)
        q1 = self.Q1(combined_input)
        q2 = self.Q2(combined_input)
        return q1, q2
    
    def q_for_action(self,obs_worker_meta_task_i, obs_local_dc_i_for_worker, obs_global_context, action_idx):
        q1, q2 = self.forward_q_value(obs_worker_meta_task_i, obs_local_dc_i_for_worker,obs_global_context)
        q_selected = lambda q: q.gather(1, action_idx.unsqueeze(-1)).squeeze(-1)
        return q_selected(q1), q_selected(q2)
       



        



