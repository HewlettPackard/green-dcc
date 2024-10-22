# pylint: disable=line-too-long,missing-function-docstring,missing-class-docstring,missing-module-docstring, superfluos-parens
# pylint: disable=C0303,C0301,C0116,C0103,C0209,W1514,W0311,C0235,C0114
import random 
import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical
import torch.utils.data as data
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.optim.lr_scheduler import CyclicLR

################################## set device ##################################
# print("============================================================================================")
# set device to cpu or cuda
device = torch.device('cpu')

################################## PPO Policy ##################################
class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.state_values = []
        self.is_terminals = []
    
    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.is_terminals[:]


class RunningStats:
    def __init__(self):
        self.n = 0
        self.mean = None
        self.var = None

    def update(self, batch):
        if not isinstance(batch, torch.Tensor):
            batch = torch.tensor(batch, dtype=torch.float32).to(device)
        
        # Compute mean and variance over batch dimension (dim=0)
        batch_mean = batch.mean(dim=0)  # Shape: (num_features,)
        batch_var = batch.var(dim=0, unbiased=False)  # Shape: (num_features,)

        batch_size = batch.size(0)
        if self.n == 0:
            self.mean = batch_mean
            self.var = batch_var
            self.n = batch_size
        else:
            delta = batch_mean - self.mean
            total_n = self.n + batch_size
            new_mean = self.mean + delta * batch_size / total_n

            m_a = self.var * self.n
            m_b = batch_var * batch_size
            m2 = m_a + m_b + delta.pow(2) * self.n * batch_size / total_n
            new_var = m2 / total_n

            self.mean = new_mean
            self.var = new_var
            self.n = total_n

    def normalize(self, batch):
        if not isinstance(batch, torch.Tensor):
            batch = torch.tensor(batch, dtype=torch.float32).to(device)
        
        std = torch.sqrt(self.var + 1e-6)
        normalized_batch = (batch - self.mean) / std
        return normalized_batch



class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, has_continuous_action_space, action_std_init, init_output_bias=0.0):
        super(Actor, self).__init__()
        self.has_continuous_action_space = has_continuous_action_space
        # self.layer_norm = nn.LayerNorm(state_dim)
        self.action_dim = action_dim
        self.init_output_bias = init_output_bias  # New parameter for output bias initialization

        if has_continuous_action_space:
            self.action_var = torch.full((action_dim,), action_std_init * action_std_init).to(device)

            self.actor = nn.Sequential(
                nn.Linear(state_dim, 64),
                # nn.LayerNorm(64),     # LayerNorm after Linear
                nn.Tanh(),
                nn.Linear(64, 64),
                # nn.LayerNorm(64),     # LayerNorm after Linear
                nn.Tanh(),
                nn.Linear(64, action_dim)
                # nn.Tanh()
            )
            self.apply(self.init_weights)

        else:
            self.actor = nn.Sequential(
                nn.Linear(state_dim, 64),
                # nn.LayerNorm(64),     # LayerNorm after Linear
                nn.Tanh(),
                nn.Linear(64, 64),
                # nn.LayerNorm(64),     # LayerNorm after Linear
                nn.Tanh(),
                nn.Linear(64, action_dim),
                nn.Softmax(dim=-1)
            )
            # Initialize the output layer biases to favor the last action
            self.initialize_biases()

        
        # Apply weight initialization
        

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            # Orthogonal initialization for weights
            nn.init.orthogonal_(m.weight)
            nn.init.zeros_(m.bias)  # Initialize biases to zero

            # For the final layer, adjust initialization
            if m.out_features == self.action_dim:
                nn.init.orthogonal_(m.weight, gain=0.01)  # Small gain for action mean initialization
                # Set bias to a positive value to bias outputs towards +1.0
                nn.init.constant_(m.bias, self.init_output_bias)

    def initialize_biases(self):
        if not self.has_continuous_action_space:
            # Access the last Linear layer before Softmax
            output_layer = self.actor[-2]  # The second to last layer is Linear
            if isinstance(output_layer, nn.Linear):
                with torch.no_grad():
                    # Initialize biases
                    output_layer.bias.data.fill_(0.0)  # Set all biases to 0.0
                    output_layer.bias.data[-1] = 0.2    # Set bias for the last action to a higher value

    def set_action_std(self, new_action_std):
        if self.has_continuous_action_space:
            self.action_var = torch.full((self.action_dim,), new_action_std * new_action_std).to(device)

    def forward(self, state):
        if self.has_continuous_action_space:
            action_mean = self.actor(state)
            return action_mean
        else:
            action_probs = self.actor(state)
            return action_probs

    def evaluate(self, state, action):
        if self.has_continuous_action_space:
            action_mean = self.actor(state)
            action_var = self.action_var.expand_as(action_mean)
            cov_mat = torch.diag_embed(action_var).to(device)
            dist = MultivariateNormal(action_mean, cov_mat)
        else:
            action_probs = self.actor(state)
            dist = Categorical(action_probs)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()

        return action_logprobs, dist_entropy
    

class Critic(nn.Module):
    def __init__(self, state_dim):
        super(Critic, self).__init__()
        # self.layer_norm = nn.LayerNorm(state_dim)
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 256),
            # nn.LayerNorm(256),     # LayerNorm after Linear
            nn.Tanh(),
            nn.Linear(256, 256),
            # nn.LayerNorm(256),     # LayerNorm after Linear
            nn.Tanh(),
            nn.Linear(256, 1)
        )

        self.apply(self._init_weights)  # Apply the initialization method

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # Use Xavier initialization with gain appropriate for 'tanh'
            gain = nn.init.calculate_gain('tanh')
            nn.init.xavier_uniform_(m.weight, gain=gain)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            # Optionally, you can initialize LayerNorm parameters here
            # By default, weight is initialized to ones and bias to zeros
            pass  # Default initialization is usually sufficient
        
    def forward(self, state):
        # state = self.layer_norm(state)
        return self.critic(state)

class RolloutDataset(data.Dataset):
    def __init__(self, states, actions, logprobs, returns, advantages, old_state_values):
        self.states = states
        self.actions = actions
        self.logprobs = logprobs
        self.returns = returns
        self.advantages = advantages
        self.old_state_values = old_state_values

    def __len__(self):
        return len(self.returns)

    def __getitem__(self, idx):
        return (self.states[idx], self.actions[idx], self.logprobs[idx], 
                self.returns[idx], self.advantages[idx], self.old_state_values[idx])

        
class PPO:
    def __init__(self, state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space, action_std_init=0.6, init_output_bias=0.0):

        self.has_continuous_action_space = has_continuous_action_space

        if has_continuous_action_space:
            self.action_std = action_std_init

        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        
        self.buffer = RolloutBuffer()
        
        # Separate actor and critic networks
        self.actor = Actor(state_dim, action_dim, has_continuous_action_space, action_std_init, init_output_bias).to(device)
        self.critic = Critic(state_dim).to(device)

        self.actor_old = Actor(state_dim, action_dim, has_continuous_action_space, action_std_init, init_output_bias).to(device)
        self.actor_old.load_state_dict(self.actor.state_dict())

        self.critic_old = Critic(state_dim).to(device)
        self.critic_old.load_state_dict(self.critic.state_dict())

        # Separate optimizers
        self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)

        # Set up CosineAnnealingWarmRestarts scheduler
        # self.scheduler_actor = CosineAnnealingWarmRestarts(self.optimizer_actor, T_0=100, T_mult=1, eta_min=1e-7)
        # self.scheduler_critic = CosineAnnealingWarmRestarts(self.optimizer_critic, T_0=100, T_mult=1, eta_min=1e-7)
        # Initialize CyclicLR schedulers
        self.scheduler_actor = CyclicLR(
            self.optimizer_actor,
            base_lr=lr_actor / 100,  # Set base_lr to a fraction of lr_actor
            max_lr=lr_actor,        # Use initial lr_actor as max_lr
            step_size_up=50000,       # Adjust step_size_up as needed
            mode='triangular2',
            cycle_momentum=False    # Set to True if using optimizers with momentum
        )

        self.scheduler_critic = CyclicLR(
            self.optimizer_critic,
            base_lr=lr_critic / 100,
            max_lr=lr_critic,
            step_size_up=50000,
            mode='triangular2',
            cycle_momentum=False
        )

        self.MseLoss = nn.MSELoss()
        self.batch_size = 128
        
        # Running statistics for state, advantages, and rewards
        self.state_stats = RunningStats()
        self.advantage_stats = RunningStats()
        self.reward_stats = RunningStats()
        
        self.loss_fn = nn.SmoothL1Loss()

    def set_action_std(self, new_action_std):
        if self.has_continuous_action_space:
            self.action_std = new_action_std
            self.actor.set_action_std(new_action_std)
            self.actor_old.set_action_std(new_action_std)
        else:
            print("--------------------------------------------------------------------------------------------")
            print("WARNING : Calling PPO::set_action_std() on discrete action space policy")
            print("--------------------------------------------------------------------------------------------")


    def decay_action_std(self, action_std_decay_rate, min_action_std):
        print("--------------------------------------------------------------------------------------------")
        if self.has_continuous_action_space:
            self.action_std = self.action_std - action_std_decay_rate
            self.action_std = round(self.action_std, 4)
            if (self.action_std <= min_action_std):
                self.action_std = min_action_std
                print("setting actor output action_std to min_action_std : ", self.action_std)
            else:
                print("setting actor output action_std to : ", self.action_std)
            self.set_action_std(self.action_std)

        else:
            print("WARNING : Calling PPO::decay_action_std() on discrete action space policy")
        print("--------------------------------------------------------------------------------------------")


    def select_action(self, state, evaluate=False):
        # Convert state to tensor and move to the correct device
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        
        # Update running statistics and normalize the state
        self.state_stats.update(state)
        state = self.state_stats.normalize(state)

        if evaluate:
            with torch.no_grad():
                if self.has_continuous_action_space:
                    action_mean = self.actor(state)
                    action = action_mean  # Deterministic action
                else:
                    action_probs = self.actor(state)
                    action = torch.argmax(action_probs, dim=-1)
                state_val = self.critic(state).item()
        else:
            if self.has_continuous_action_space:
                action_mean = self.actor(state)
                cov_mat = torch.diag(self.actor.action_var).unsqueeze(dim=0).to(device)
                dist = MultivariateNormal(action_mean, cov_mat)
                action = dist.sample()
                action_logprob = dist.log_prob(action)
            else:
                action_probs = self.actor(state)
                dist = Categorical(action_probs)
                
                epsilon = 0.1  # Exploration rate; you can adjust or anneal this over time
                if random.random() < epsilon:
                    # Select a random action
                    action = torch.tensor([random.randrange(self.actor.action_dim)]).to(device)
                    action_logprob = dist.log_prob(action)
                else:
                    # Sample from the policy
                    action = dist.sample()
                    action_logprob = dist.log_prob(action)
                    
                # action = dist.sample()
                # action_logprob = dist.log_prob(action)
            state_val = self.critic(state).item()

        if evaluate:
            if self.has_continuous_action_space:
                return action.detach().cpu().numpy().flatten()
            else:
                return action.item()
        else:
            if self.has_continuous_action_space:
                return action.detach().cpu().numpy().flatten(), action_logprob.item(), state_val
            else:
                return action.item(), action_logprob.item(), state_val




    def compute_advantages(self, returns, state_values, is_terminals, gamma, lam=0.95):
        """
        If 'rewards' already contains the discounted return, we just subtract the 
        state values to compute the advantages.
        """
        # advantages = []
        
        # for t in range(len(rewards)):
            # advantage = rewards[t] - state_values[t]  # Subtract state values from returns
            # advantages.append(advantage)
        # Ensure rewards and state_values are tensors of the same shape
        returns = torch.tensor(returns, dtype=torch.float32).to(device)
        state_values = torch.tensor(state_values, dtype=torch.float32).to(device)

        # Vectorized advantage calculation
        advantages = returns - state_values
        return advantages

    def update(self):
        # Convert buffer lists to tensors
        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(device)
        if self.has_continuous_action_space:
            old_actions = torch.stack(self.buffer.actions, dim=0).detach().to(device)  # Shape: (N, 1)
        else:
            old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(device)
        old_returns = torch.tensor(self.buffer.rewards, dtype=torch.float32).detach().to(device)
        old_is_terminals = torch.tensor(self.buffer.is_terminals, dtype=torch.float32).detach().to(device)
        old_state_values = torch.squeeze(torch.stack(self.buffer.state_values, dim=0)).detach().to(device)

        # Normalize states once
        self.state_stats.update(old_states)
        old_states = self.state_stats.normalize(old_states)

        # Prepare data as a list of transitions
        dataset = data.TensorDataset(old_states, old_actions, old_logprobs, old_returns, old_state_values)
        # No need to compute advantages yet

        total_actor_loss = 0
        total_critic_loss = 0

        # Optimize policy for K epochs
        for _ in range(self.K_epochs):
            # Recompute state values and advantages using the updated critic
            with torch.no_grad():
                state_values = self.critic(old_states).squeeze()
            advantages = old_returns - state_values.detach()

            # Normalize advantages per epoch
            self.advantage_stats.update(advantages)
            advantages = self.advantage_stats.normalize(advantages)
            
            # Check that the lengths of the tensors are correct
            if not (old_states.size(0) == old_actions.size(0) == old_logprobs.size(0) == old_returns.size(0) == advantages.size(0) == old_state_values.size(0)):
                print("Lengths of states, actions, logprobs, returns, advantages, and state values do not match!")
                print("Length of states:", old_states.size(0))
                print("Length of actions:", old_actions.size(0))
                print("Length of logprobs:", old_logprobs.size(0))
                print("Length of returns:", old_returns.size(0))
                print("Length of advantages:", advantages.size(0))
                print("Length of state values:", old_state_values.size(0))
                

            # Create dataset with recomputed advantages
            epoch_dataset = data.TensorDataset(old_states, old_actions, old_logprobs, old_returns, advantages, old_state_values)


            # Shuffle transitions before creating minibatches
            loader = data.DataLoader(epoch_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0)

            for batch in loader:
                batch_states, batch_actions, batch_logprobs, batch_returns, batch_advantages, batch_old_state_values = batch

                # Move tensors to device if not already
                batch_states = batch_states.to(device)
                batch_actions = batch_actions.to(device)
                batch_logprobs = batch_logprobs.to(device)
                batch_returns = batch_returns.to(device)
                batch_advantages = batch_advantages.to(device)
                batch_old_state_values = batch_old_state_values.to(device)

                # Evaluate actions and compute loss
                batch_logprobs, dist_entropy = self.actor.evaluate(batch_states, batch_actions)

                state_values = self.critic(batch_states).squeeze()

                # Recompute old action probabilities using actor_old
                with torch.no_grad():
                    batch_old_logprobs, _ = self.actor_old.evaluate(batch_states, batch_actions)
        
                # Find ratio (pi_theta / pi_theta__old)
                ratios = torch.exp(batch_logprobs - batch_old_logprobs)

                # Surrogate loss
                surr1 = ratios * batch_advantages
                surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * batch_advantages
                actor_loss = -torch.min(surr1, surr2) - 0.01 * dist_entropy

                # Value function clipping with Huber loss (or MSE loss)
                value_pred_clipped = batch_old_state_values + torch.clamp(
                    state_values - batch_old_state_values, -self.eps_clip, self.eps_clip
                )
                value_loss_unclipped = (state_values - batch_returns).pow(2)
                value_loss_clipped = (value_pred_clipped - batch_returns).pow(2)
                critic_loss = 0.5 * torch.max(value_loss_unclipped, value_loss_clipped).mean()

                # Update actor
                self.optimizer_actor.zero_grad()
                actor_loss.mean().backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=5.0)
                self.optimizer_actor.step()
                self.scheduler_actor.step()

                # Update critic
                self.optimizer_critic.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=5.0)
                self.optimizer_critic.step()
                self.scheduler_critic.step()

                total_actor_loss += actor_loss.mean().item()
                total_critic_loss += critic_loss.item()

        # Copy new weights into old policy
        self.actor_old.load_state_dict(self.actor.state_dict())
        self.critic_old.load_state_dict(self.critic.state_dict())

        # Average the losses over the batches
        avg_actor_loss = total_actor_loss / (self.K_epochs)
        avg_critic_loss = total_critic_loss / (self.K_epochs)

        # Clear buffer
        self.buffer.clear()

        # Return loss for logging
        return avg_actor_loss, avg_critic_loss

    
    def save(self, checkpoint_path):
        torch.save(self.actor_old.state_dict(), checkpoint_path + "_actor.pth")
        torch.save(self.critic_old.state_dict(), checkpoint_path + "_critic.pth")
        
        # Save the running statistics for states, advantages, and rewards
        stats = {
            'state_stats': self.state_stats,
            'advantage_stats': self.advantage_stats,
            'reward_stats': self.reward_stats
        }
        torch.save(stats, checkpoint_path + "_stats.pth")

    def load(self, checkpoint_path):
        self.actor_old.load_state_dict(torch.load(checkpoint_path + "_actor.pth", map_location=device))
        self.actor.load_state_dict(torch.load(checkpoint_path + "_actor.pth", map_location=device))
        self.critic_old.load_state_dict(torch.load(checkpoint_path + "_critic.pth", map_location=device))
        self.critic.load_state_dict(torch.load(checkpoint_path + "_critic.pth", map_location=device))

        # Load the running statistics
        stats = torch.load(checkpoint_path + "_stats.pth", map_location=device)
        self.state_stats = stats['state_stats']
        self.advantage_stats = stats['advantage_stats']
        self.reward_stats = stats['reward_stats']


