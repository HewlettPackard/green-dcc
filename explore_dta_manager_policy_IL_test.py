import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

# --- Determine Device ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# --- Hyperparameters (same as explore_dta_manager_policy.py) ---
D_META_MANAGER = 7
D_GLOBAL = 4
D_OPTION_FEAT = 8
D_EMB_OPTION = 64
D_EMB_QUERY = 64
MAX_TOTAL_OPTIONS = 5 # 1 local + up to 4 remotes for this example
NUM_TF_HEADS = 2
NUM_TF_LAYERS = 1

# --- Feature indices for the expert rule ---
FEAT_IDX_1 = 2
FEAT_IDX_2 = 7

from explore_dta_manager_policy import DTAManagerPolicyExplore as DTAManagerPolicyIL

# --- Helper to generate one data sample ---
def generate_sample(num_actual_options):
    # Ensure num_actual_options <= MAX_TOTAL_OPTIONS
    num_actual_options = min(num_actual_options, MAX_TOTAL_OPTIONS)

    obs_meta = torch.randn(1, D_META_MANAGER)
    obs_global = torch.randn(1, D_GLOBAL)
    
    obs_options_set = torch.zeros(1, MAX_TOTAL_OPTIONS, D_OPTION_FEAT)
    # Populate features for actual options
    for i in range(num_actual_options):
        # Make features somewhat diverse, ensure positive for product rule if needed
        obs_options_set[0, i, :] = torch.rand(D_OPTION_FEAT) + 0.1 
        # Specifically control features for expert rule
        # obs_options_set[0, i, FEAT_IDX_1] = torch.rand(1) * 10 + 1 # Values between 1 and 11
        # obs_options_set[0, i, FEAT_IDX_2] = torch.rand(1) * 10 + 1 # Values between 1 and 11

    padding_mask = torch.ones(1, MAX_TOTAL_OPTIONS, dtype=torch.bool)
    padding_mask[0, :num_actual_options] = False

    # Expert action
    min_product_score = float('inf')
    expert_action_idx = -1
    for i in range(num_actual_options): # Only consider valid options
        feat_vec = obs_options_set[0, i, :]
        product_score = feat_vec[FEAT_IDX_1] * feat_vec[FEAT_IDX_2] - feat_vec[FEAT_IDX_1] - feat_vec[FEAT_IDX_2]
        if product_score < min_product_score:
            min_product_score = product_score
            expert_action_idx = i
    
    if expert_action_idx == -1 and num_actual_options > 0: # Should not happen if num_actual_options > 0
        expert_action_idx = 0 # Default to first valid if error
    elif num_actual_options == 0: # Handle case with no valid options (shouldn't occur if 1 local always there)
        expert_action_idx = 0 # Or a special "no-op" index if your setup has one

    return {
        "meta": obs_meta, "global": obs_global, 
        "options_set": obs_options_set, "mask": padding_mask
    }, torch.tensor([expert_action_idx], dtype=torch.long)


# --- Imitation Learning Training ---
def train_imitation(policy, num_epochs=100, batch_size=32, dataset_size=10000):
    print(f"\n--- Starting Imitation Learning (Supervised Training) ---")
    optimizer = optim.Adam(policy.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss() # Expects raw logits from policy
    policy.to(DEVICE)
    policy.train() # Set policy to training mode
    print(f"Training on {dataset_size} samples with batch size {batch_size} for {num_epochs} epochs...")

    # Generate dataset
    dataset = []
    for _ in range(dataset_size):
        # Vary the number of actual options to test generalization
        num_options = np.random.randint(1, MAX_TOTAL_OPTIONS + 1) # At least 1 option (local)
        obs_dict, expert_action = generate_sample(num_options)
        if expert_action.item() != -1 : # Only add valid samples
            dataset.append((obs_dict, expert_action))
    
    if not dataset:
        print("ERROR: No valid data generated for training.")
        return

    for epoch in range(num_epochs):
        policy.train()
        total_loss = 0
        correct_preds = 0
        total_preds = 0
        
        # Simple batching
        random.shuffle(dataset)
        for i in range(0, len(dataset), batch_size):
            batch = dataset[i : i+batch_size]
            if not batch: continue

            obs_metas, obs_globals, obs_options_sets, obs_masks, expert_actions = [], [], [], [], []
            for obs_d, act_e in batch:
                obs_metas.append(obs_d["meta"])
                obs_globals.append(obs_d["global"])
                obs_options_sets.append(obs_d["options_set"])
                obs_masks.append(obs_d["mask"])
                expert_actions.append(act_e)

            b_meta = torch.cat(obs_metas).to(DEVICE)
            b_global = torch.cat(obs_globals).to(DEVICE)
            b_options = torch.cat(obs_options_sets).to(DEVICE)
            b_mask = torch.cat(obs_masks).to(DEVICE)
            b_expert_actions = torch.cat(expert_actions).squeeze().to(DEVICE) # Ensure it's [BatchSize]

            optimizer.zero_grad()
            logits_destination_options = policy(b_meta, b_global, b_options, b_mask)
            
            loss = criterion(logits_destination_options, b_expert_actions)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            # Accuracy for logging
            with torch.no_grad():
                preds = torch.argmax(logits_destination_options, dim=-1)
                correct_preds += (preds == b_expert_actions).sum().item()
                total_preds += b_expert_actions.size(0)
        
        avg_loss = total_loss / (len(dataset) / batch_size)
        accuracy = correct_preds / total_preds if total_preds > 0 else 0
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")

        if accuracy > 0.995: # Early stopping if good enough
            print("Achieved high accuracy, stopping early.")
            break

# --- Test the Trained Policy ---
def test_trained_policy(policy):
    print("\n--- Testing Trained Policy on New Samples ---")
    policy.eval()
    for _ in range(5): # Test on 5 new samples
        num_options = np.random.randint(1, MAX_TOTAL_OPTIONS + 1)
        obs_dict, expert_action_tensor = generate_sample(num_options)
        expert_action_idx = expert_action_tensor.item()

        print(f"\nTest Sample (Num Actual Options: {num_options}):")
        # print(f"  Expert choice index: {expert_action_idx}")
        expert_score_val = float('inf')
        if expert_action_idx != -1 and expert_action_idx < num_options: # Check if expert action is valid
            expert_feat = obs_dict["options_set"][0, expert_action_idx]
            expert_score_val = expert_feat[FEAT_IDX_1] * expert_feat[FEAT_IDX_2]
            print(f"  Expert choice index: {expert_action_idx}, Expert score (feat[{FEAT_IDX_1}]*feat[{FEAT_IDX_2}]): {expert_score_val:.2f}")
        else:
            print(f"  Expert choice index: {expert_action_idx} (No valid options or error)")


        with torch.no_grad():
            masked_logits = policy(
                obs_dict["meta"].to(DEVICE), obs_dict["global"].to(DEVICE),
                obs_dict["options_set"].to(DEVICE), obs_dict["mask"].to(DEVICE)
            )
        masked_logits = masked_logits.cpu()  # Move back to CPU for printing
        probabilities = F.softmax(masked_logits, dim=-1)
        policy_choice_idx = torch.argmax(probabilities, dim=-1).item()
        
        print(f"  Policy Masked Logits:\n{masked_logits[0].numpy()}")
        print(f"  Policy Probabilities:\n{probabilities[0].numpy()}")
        print(f"  Policy choice index: {policy_choice_idx}")

        if policy_choice_idx < num_options: # Check if policy choice is a valid option
            policy_feat = obs_dict["options_set"][0, policy_choice_idx]
            policy_score_val = policy_feat[FEAT_IDX_1] * policy_feat[FEAT_IDX_2]
            print(f"  Policy choice score (feat[{FEAT_IDX_1}]*feat[{FEAT_IDX_2}]): {policy_score_val:.2f}")
        else:
            print(f"  Policy choice index {policy_choice_idx} is out of bounds for actual options ({num_options}).")


        if policy_choice_idx == expert_action_idx:
            print("  SUCCESS: Policy matched expert.")
        else:
            print("  FAILURE: Policy did NOT match expert.")

if __name__ == "__main__":
    # 1. Initialize policy
    dta_policy = DTAManagerPolicyIL()

    # 2. Train with imitation learning
    train_imitation(dta_policy, num_epochs=50, dataset_size=20000, batch_size=64) # Adjust params

    # 3. Test the trained policy
    test_trained_policy(dta_policy)