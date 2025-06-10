import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd # For Timedelta, if used in generating example obs

# --- Assumptions for this Exploration Script ---
# 1. We will define a simplified version of the DTA_Manager policy network here.
#    In your actual project, you would import it from `rl_components.agent_net`
#    or wherever DTA_Policy_Attention is defined.
# 2. We'll use example dimensions. Make sure these match your actual policy.
# 3. We'll focus on a single DTA_Manager instance (e.g., for DC_1).

# --- Hyperparameters for the DTA_Manager Policy (Match your actual config) ---
D_META_MANAGER = 7        # Dimension of meta-task descriptor
D_GLOBAL = 4              # Dimension of global context (time features)
D_OPTION_FEAT = 8         # Dimension of features for each destination option (local or remote)
D_EMB_OPTION = 64         # Embedding dimension for options after MLP_option_encoder
D_EMB_QUERY = 64          # Embedding dimension for the final query vector
MAX_TOTAL_OPTIONS = 5     # Max number of options (1 local + N-1 remote, padded).
                          # For N=3 DCs, M=3 options. If MAX_TOTAL_OPTIONS=5, we'll have 2 padding.
                          # For N=5 DCs, M=5 options. If MAX_TOTAL_OPTIONS=5, no padding.
                          # Let's use a scenario with actual_num_options <= MAX_TOTAL_OPTIONS
NUM_TF_HEADS = 2          # Number of heads in Transformer Encoder
NUM_TF_LAYERS = 1         # Number of layers in Transformer Encoder

# --- Simplified NN Modules (for this script, or import your actual ones) ---
class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=D_EMB_QUERY): # Use D_EMB_QUERY as a common hidden
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    def forward(self, x):
        return self.net(x)

# --- DTA_Manager Policy Definition (Simplified version of your pi_Manager) ---
class DTAManagerPolicyExplore(nn.Module):
    def __init__(self):
        super().__init__()
        self.max_total_options = MAX_TOTAL_OPTIONS
        self.d_emb_option = D_EMB_OPTION # Store for reshaping later

        # Encoders
        self.mlp_meta_mgr = MLP(D_META_MANAGER, D_EMB_QUERY)
        self.mlp_global_mgr = MLP(D_GLOBAL, D_EMB_QUERY)
        self.mlp_option_encoder = MLP(D_OPTION_FEAT, D_EMB_OPTION)

        # Attention Module (Transformer Encoder)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=D_EMB_OPTION,
            nhead=NUM_TF_HEADS,
            dim_feedforward=D_EMB_OPTION * 2, # Example feedforward dim
            batch_first=True,
            dropout=0.0 # No dropout for deterministic exploration
        )
        self.remote_transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=NUM_TF_LAYERS)

        # Query Formation
        self.mlp_task_query_mgr = MLP(D_EMB_QUERY * 2, D_EMB_QUERY) # input: concat(emb_meta, emb_global)

        # Scoring Head (Simple dot product for this example, or MLP scorer)
        self.scorer_mlp = nn.Sequential(
            nn.Linear(D_EMB_QUERY + D_EMB_OPTION, D_EMB_OPTION // 2), # Example hidden dim
            nn.ReLU(),
            nn.Linear(D_EMB_OPTION // 2, 1)
        )
        # To use the dot product directly, you might need a projection layer for the query
        # # if self.d_emb_query != self.d_emb_option:
        # self.query_projection_for_dot_product = nn.Linear(self.d_emb_query, self.d_emb_option)

    def forward(self, obs_manager_meta_task, obs_global_context,
                obs_all_options_set_padded, all_options_padding_mask):
        
        # --- Stage 1: Feature Encoding ---
        #print("\n--- Stage 1: Feature Encoding ---")
        emb_meta_task_mgr = self.mlp_meta_mgr(obs_manager_meta_task)
        #print(f"  emb_meta_task_mgr shape: {emb_meta_task_mgr.shape}")
        # #print(f"  emb_meta_task_mgr:\n{emb_meta_task_mgr.detach().numpy()}")

        emb_global_context_mgr = self.mlp_global_mgr(obs_global_context)
        #print(f"  emb_global_context_mgr shape: {emb_global_context_mgr.shape}")

        # Encode all options (local + remote + padding)
        # obs_all_options_set_padded shape: [B, max_total_options, D_option_feat]
        B = obs_all_options_set_padded.shape[0] # Batch size (will be 1 for this exploration)
        options_flat = obs_all_options_set_padded.view(B * self.max_total_options, -1)
        emb_options_initial_flat = self.mlp_option_encoder(options_flat)
        seq_emb_options_initial = emb_options_initial_flat.view(B, self.max_total_options, self.d_emb_option)
        #print(f"  seq_emb_options_initial shape: {seq_emb_options_initial.shape}")
        # #print(f"  seq_emb_options_initial (first option):\n{seq_emb_options_initial[0,0,:].detach().numpy()}")


        # --- Stage 2: Contextualization of All Options (Attention) ---
        #print("\n--- Stage 2: Contextualization of Options (Attention) ---")
        # TransformerEncoderLayer expects src_key_padding_mask where True means "ignore"
        # all_options_padding_mask is already in this format (True for padded)
        seq_emb_options_contextual = self.remote_transformer_encoder(
            src=seq_emb_options_initial,
            src_key_padding_mask=all_options_padding_mask
        )
        #print(f"  seq_emb_options_contextual shape: {seq_emb_options_contextual.shape}")
        # #print(f"  seq_emb_options_contextual (first valid option after attention):\n{seq_emb_options_contextual[0, all_options_padding_mask[0]==False][0].detach().numpy()}")


        # --- Stage 3: Task Context Query Formation ---
        #print("\n--- Stage 3: Task Context Query Formation ---")
        query_input_mgr = torch.cat([emb_meta_task_mgr, emb_global_context_mgr], dim=-1)
        task_context_query_mgr = self.mlp_task_query_mgr(query_input_mgr)
        #print(f"  task_context_query_mgr shape: {task_context_query_mgr.shape}")
        # #print(f"  task_context_query_mgr:\n{task_context_query_mgr.detach().numpy()}")


        # --- Stage 4: Scoring Head ---
        #print("\n--- Stage 4: Scoring Destination Options ---")
        # task_context_query_mgr shape: [B, D_EMB_QUERY]
        # seq_emb_options_contextual shape: [B, max_total_options, D_EMB_OPTION]

        # We need D_EMB_QUERY to match D_EMB_OPTION for a clean dot product or simple MLP scorer per option.
        # Let's assume D_EMB_QUERY == D_EMB_OPTION for this example. If not, one would need projection.
        # For this exploration script, we'll ensure they match by design of the MLPs or add a explicit projection for the query.
        
        # Option 1: Dot Product Scoring (if D_EMB_QUERY == D_EMB_OPTION)
        # Ensure query is suitable for broadcasting or batch matrix multiply (bmm)
        # query_for_scoring needs to be [B, D_EMB_OPTION]
        if not D_EMB_QUERY == D_EMB_OPTION:
            raise ValueError("D_EMB_QUERY must match D_EMB_OPTION for this exploration script to work correctly.")
        query_for_scoring = task_context_query_mgr # Assuming D_EMB_QUERY == D_EMB_OPTION (e.g., both are 64)

        # Reshape query for bmm: [B, D_EMB_OPTION] -> [B, 1, D_EMB_OPTION] to act as a single query vector per batch item
        # or [B, D_EMB_OPTION, 1] if you want to bmm with [B, max_total_options, D_EMB_OPTION]
        # Let's use bmm: query [B, 1, D_emb], options [B, D_emb, max_total_options] (after permute) -> [B, 1, max_total_options]
        
        # query_for_bmm = query_for_scoring.unsqueeze(1) # [B, 1, D_EMB_OPTION]
        # options_for_bmm = seq_emb_options_contextual.permute(0, 2, 1) # [B, D_EMB_OPTION, max_total_options]
        # logits_destination_options = torch.bmm(query_for_bmm, options_for_bmm).squeeze(1) # [B, max_total_options]

        # Alternative and often cleaner for scoring: element-wise product then sum, or a small MLP
        # For element-wise product and sum (cosine similarity style without normalization, or scaled dot product):
        # query_expanded_for_dot = query_for_scoring.unsqueeze(1) # [B, 1, D_EMB_OPTION]
        # logits_destination_options = torch.sum(query_expanded_for_dot * seq_emb_options_contextual, dim=2) # [B, max_total_options]

        # Option 2: MLP Scorer (more general, doesn't require D_EMB_QUERY == D_EMB_OPTION initially)
        # This is what was in my previous refined diagram explanation using self.pointer_score_mlp
        # Input to scorer MLP: concat(query_expanded, option_embedding)
        query_expanded_for_mlp = query_for_scoring.unsqueeze(1).repeat(1, self.max_total_options, 1) # [B, max_total_options, D_EMB_QUERY]
        
        # Concatenate each option's embedding with the (repeated) query vector
        scoring_input_features = torch.cat([query_expanded_for_mlp, seq_emb_options_contextual], dim=-1) # [B, max_total_options, D_EMB_QUERY + D_EMB_OPTION]
        scoring_input_flat = scoring_input_features.view(B * self.max_total_options, D_EMB_QUERY + D_EMB_OPTION)
        
        # Define self.scorer_mlp in __init__ if using this:
        # self.scorer_mlp = nn.Sequential(
        #     nn.Linear(D_EMB_QUERY + D_EMB_OPTION, D_EMB_OPTION // 2), # Example hidden dim
        #     nn.ReLU(),
        #     nn.Linear(D_EMB_OPTION // 2, 1)
        # )
        # For this script, let's define it ad-hoc for clarity, or assume it exists.
        # If self.scorer_mlp is not defined, this part will error.
        # Let's make a dummy scorer for this script if not already part of the class:
        if not hasattr(self, 'scorer_mlp'): # Check if scorer_mlp exists
            #print("  INFO: Scorer MLP not explicitly defined in DTAManagerPolicyExplore __init__. Using a temporary one for exploration.")
            # Define a temporary scorer for the exploration script if it wasn't in the class init
            # This should ideally be part of the class __init__ for a real policy
            temp_scorer_mlp = nn.Sequential(
                nn.Linear(D_EMB_QUERY + D_EMB_OPTION, max(1, (D_EMB_QUERY + D_EMB_OPTION) // 4) ), nn.ReLU(), # make hidden dim smaller
                nn.Linear(max(1, (D_EMB_QUERY + D_EMB_OPTION) // 4), 1)
            ).to(obs_manager_meta_task.device) # Move to same device
            destination_scores_flat = temp_scorer_mlp(scoring_input_flat)
        else: # If self.scorer_mlp was defined in __init__
            destination_scores_flat = self.scorer_mlp(scoring_input_flat) # [B * max_total_options, 1]
        
        logits_destination_options = destination_scores_flat.view(B, self.max_total_options) # [B, max_total_options]

        #print(f"  Raw logits_destination_options shape: {logits_destination_options.shape}")
        #print(f"  Raw logits_destination_options (first batch item if B > 1):\n{logits_destination_options[0].detach().numpy()}")

        # Apply padding mask to logits
        masked_logits = logits_destination_options.masked_fill(all_options_padding_mask, -float('inf'))
        #print(f"  Masked logits_destination_options (first batch item if B > 1):\n{masked_logits[0].detach().numpy()}")

        return masked_logits

# --- Main Exploration Function ---
def explore_policy():
    #print("--- Initializing DTA_Manager Policy for Exploration ---")
    policy = DTAManagerPolicyExplore()
    policy.eval() # Set to evaluation mode (disables dropout if any)

    # --- Create Example Inputs (Batch Size B=1) ---
    #print("\n--- Creating Example Inputs (Batch Size B=1) ---")
    
    # Scenario: 3 Datacenters (DC0=Local, DC1=Remote1, DC2=Remote2)
    # MAX_TOTAL_OPTIONS is 5, so 2 will be padding.
    actual_num_options = 3 # DC0 (local), DC1 (remote), DC2 (remote)

    # 1. obs_manager_meta_task_i
    example_meta_task = torch.randn(1, D_META_MANAGER) # Batch of 1
    #print(f"obs_manager_meta_task_i shape: {example_meta_task.shape}")

    # 2. obs_global_context
    example_global_context = torch.randn(1, D_GLOBAL)
    #print(f"obs_global_context shape: {example_global_context.shape}")

    # 3. obs_all_options_set_padded
    example_all_options_set = torch.zeros(1, MAX_TOTAL_OPTIONS, D_OPTION_FEAT)
    # Fill in features for the actual options
    for i in range(actual_num_options):
        example_all_options_set[0, i, :] = torch.rand(D_OPTION_FEAT) * (i + 1) # Make them somewhat distinct
    #print(f"obs_all_options_set_padded shape: {example_all_options_set.shape}")
    # #print("Example features for first 3 options (Local, Remote1, Remote2):")
    # for i in range(actual_num_options):
    #     #print(f"  Option {i}: {example_all_options_set[0, i, :].numpy()}")

    # 4. all_options_padding_mask
    example_padding_mask = torch.ones(1, MAX_TOTAL_OPTIONS, dtype=torch.bool)
    example_padding_mask[0, :actual_num_options] = False # First `actual_num_options` are NOT padded
    #print(f"all_options_padding_mask shape: {example_padding_mask.shape}")
    #print(f"all_options_padding_mask: {example_padding_mask.numpy()}")


    # --- Forward Pass through the Policy ---
    #print("\n--- Executing Forward Pass ---")
    with torch.no_grad(): # No need to compute gradients
        masked_logits = policy(
            example_meta_task,
            example_global_context,
            example_all_options_set,
            example_padding_mask
        )

    # --- Analyze Outputs ---
    #print("\n--- Analyzing Final Outputs ---")
    #print(f"Final Masked Logits for Destination Options (shape {masked_logits.shape}):")
    #print(masked_logits.numpy())

    probabilities = F.softmax(masked_logits, dim=-1)
    #print(f"Probabilities over Options (shape {probabilities.shape}):")
    #print(probabilities.numpy())

    chosen_action_index = torch.argmax(probabilities, dim=-1).item()
    #print(f"Chosen Action (Index of highest probability option): {chosen_action_index}")

    # if chosen_action_index == 0:
        #print("Decision: Commit to Local Worker (Option 0)")
    # else:
        #print(f"Decision: Transfer to Remote DC (Option {chosen_action_index} - which is Remote DC #{chosen_action_index})")

if __name__ == "__main__":
    explore_policy()