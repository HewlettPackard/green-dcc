import torch
import numpy as np

class RolloutStorage:
    """
    Stores transitions collected during rollouts for on-policy algorithms like PPO.
    Handles variable numbers of tasks per step for actor-related data.
    """
    def __init__(self, n_steps, obs_dim_actor, obs_dim_critic, max_tasks_per_step, device):
        self.n_steps = n_steps
        self.obs_dim_actor = obs_dim_actor
        self.obs_dim_critic = obs_dim_critic
        self.max_tasks_per_step = max_tasks_per_step # Needed for pre-allocation if desired, or use lists
        self.device = device

        # Use lists to handle variable lengths easily first
        self.actor_obs = []      # List[np.ndarray(k_t, obs_dim_actor)]
        self.critic_obs = []     # List[np.ndarray(obs_dim_critic,)]
        self.actions = []        # List[np.ndarray(k_t,)]
        self.log_probs = []      # List[np.ndarray(k_t,)]
        self.rewards = []        # List[float]
        self.dones = []          # List[bool]
        self.values = []         # List[float] - Critic predictions V(s_t)

        self.step = 0
        self.is_full = False

    def add(self, actor_obs_t, critic_obs_t, actions_t, log_probs_t, reward_t, done_t, value_t):
        """Adds one step of transition data."""
        # Convert lists/arrays from rollout to numpy for storage if needed
        # Store critic obs directly (already numpy)
        self.critic_obs.append(critic_obs_t)
        # Store rewards, dones, values
        self.rewards.append(reward_t)
        self.dones.append(done_t)
        self.values.append(value_t) # Store V(s_t) predicted during rollout

        # Handle potentially empty actor data
        if actor_obs_t.size > 0: # Check if the array contains any elements
            self.actor_obs.append(np.array(actor_obs_t, dtype=np.float32))
            self.actions.append(np.array(actions_t, dtype=np.int64))
            self.log_probs.append(np.array(log_probs_t, dtype=np.float32))
        else: # Store placeholders for empty steps
             self.actor_obs.append(np.array([], dtype=np.float32).reshape(0, self.obs_dim_actor))
             self.actions.append(np.array([], dtype=np.int64))
             self.log_probs.append(np.array([], dtype=np.float32))


        self.step += 1
        if self.step == self.n_steps:
            self.is_full = True
            self.step = 0 # Ready for next cycle if needed, but usually clear after update

    def compute_returns_and_advantages(self, last_value, gamma, gae_lambda):
        """
        Computes returns and Generalized Advantage Estimation (GAE)
        after a rollout is complete. last_value is V(s_{t+n}).
        """
        num_transitions = len(self.rewards)
        self.advantages = np.zeros(num_transitions, dtype=np.float32)
        last_gae_lam = 0
        next_value = last_value # V(s_{t+n})

        for step in reversed(range(num_transitions)):
            if self.dones[step]:
                next_non_terminal = 0.0
                next_value = 0.0 # Value of terminal state is 0
            else:
                next_non_terminal = 1.0
                # next_value is V(s_{t+1}) which comes from self.values[step+1] or last_value if at end
                # For GAE calculation we use V(s_{t+1}) directly from collected values
                # But wait, the loop runs from n-1 down to 0.
                # next_value should be the value of the *next* state in the sequence

            # Get V(s_{t+1})
            if step == num_transitions - 1:
                next_step_value = last_value # Bootstrap with V(s_{t+n})
            else:
                next_step_value = self.values[step + 1]

            delta = self.rewards[step] + gamma * next_step_value * next_non_terminal - self.values[step]
            last_gae_lam = delta + gamma * gae_lambda * next_non_terminal * last_gae_lam
            self.advantages[step] = last_gae_lam

        # Returns are advantages + values V(s_t)
        self.returns = self.advantages + np.array(self.values, dtype=np.float32)

        # Convert advantages to tensor and normalize (standard practice)
        self.advantages = torch.tensor(self.advantages, dtype=torch.float32).to(self.device)
        self.returns = torch.tensor(self.returns, dtype=torch.float32).to(self.device)
        self.advantages = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-8)

    def get_data(self):
        """
        Returns the collected rollout data, flattening actor-related lists.
        Handles variable numbers of tasks per step.
        """
        # Critic data
        critic_obs_batch = torch.tensor(np.array(self.critic_obs), dtype=torch.float32).to(self.device)
        old_values_batch = torch.tensor(np.array(self.values), dtype=torch.float32).to(self.device) # Return stored V(s_t)

        # Actor data flattening
        actor_step_indices = []
        flat_actor_obs = []
        flat_actions = []
        flat_log_probs = []
        valid_step_indices = []

        for i in range(len(self.actor_obs)):
            num_tasks_in_step = self.actor_obs[i].shape[0]
            if num_tasks_in_step > 0:
                actor_step_indices.extend([i] * num_tasks_in_step)
                flat_actor_obs.append(self.actor_obs[i]) # List of numpy arrays
                flat_actions.append(self.actions[i])
                flat_log_probs.append(self.log_probs[i])
                valid_step_indices.append(i)

        if not flat_actor_obs:
            return (critic_obs_batch, self.returns, self.advantages,
                    None, None, None, None, old_values_batch) # Added old_values_batch

        actor_obs_batch = torch.from_numpy(np.concatenate(flat_actor_obs, axis=0)).to(self.device)
        actions_batch = torch.from_numpy(np.concatenate(flat_actions, axis=0)).to(self.device)
        log_probs_batch = torch.from_numpy(np.concatenate(flat_log_probs, axis=0)).to(self.device)
        actor_step_indices = torch.tensor(actor_step_indices, dtype=torch.long).to(self.device)

        advantages_per_task = self.advantages[actor_step_indices]

        return (critic_obs_batch, self.returns, self.advantages,
                actor_obs_batch, actions_batch, log_probs_batch,
                advantages_per_task, old_values_batch) # Added old_values_batch


    def after_update(self):
        """Resets the storage after updates are performed."""
        self.actor_obs.clear()
        self.critic_obs.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.dones.clear()
        self.values.clear()
        self.step = 0
        self.is_full = False
        if hasattr(self, 'returns'):
             del self.returns
             del self.advantages