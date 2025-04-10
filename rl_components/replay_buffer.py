import random
import numpy as np
import torch

class ReplayBuffer:
    """
    Replay buffer for variable-length tasks per timestep.
    Each entry is a transition:
        (obs, actions, reward, next_obs, done)
    but both obs and next_obs are lists of different possible lengths.
    We pad them to (max_tasks, obs_dim).

    We store two separate masks:
        mask_obs: which tasks are valid in 'obs'
        mask_next: which tasks are valid in 'next_obs'
    That allows us to handle the case N != M (different #tasks in consecutive steps).

    Then at sample() time, we return batches of shape:
        obs_b: [B, max_tasks, obs_dim]
        act_b: [B, max_tasks]
        rew_b: [B]
        next_obs_b: [B, max_tasks, obs_dim]
        done_b: [B]
        mask_obs_b: [B, max_tasks]
        mask_next_b: [B, max_tasks]
    """
    def __init__(self, capacity=100_000, max_tasks=32, obs_dim=7):
        self.capacity = capacity
        self.max_tasks = max_tasks
        self.obs_dim = obs_dim

        self.buffer = []
        self.pos = 0

    def add(self, obs, actions, reward, next_obs, done):
        """
        obs: list of shape (N, obs_dim)
        actions: list of length N (one int per task)
        reward: float
        next_obs: list of shape (M, obs_dim)
        done: bool

        We'll create zero-padded arrays of shape (max_tasks, obs_dim) for both obs and next_obs,
        and int arrays of shape (max_tasks,) for actions. Then produce mask arrays for each.
        """
        N = len(obs)   # number of tasks in current obs
        M = len(next_obs)  # number of tasks in next obs

        # Prepare padded observation arrays
        obs_padded = np.zeros((self.max_tasks, self.obs_dim), dtype=np.float32)
        next_obs_padded = np.zeros((self.max_tasks, self.obs_dim), dtype=np.float32)

        # Padded actions
        act_padded = np.full(self.max_tasks, -1, dtype=np.int64)

        # Masks
        mask_obs = np.zeros(self.max_tasks, dtype=np.float32)
        mask_next = np.zeros(self.max_tasks, dtype=np.float32)

        # Fill obs/act
        if N > 0:
            obs_arr = np.array(obs, dtype=np.float32)  # shape (N, obs_dim)
            act_arr = np.array(actions, dtype=np.int64)  # shape (N,)
            obs_padded[:N] = obs_arr
            act_padded[:N] = act_arr
            mask_obs[:N] = 1.0

        # Fill next_obs
        if M > 0:
            next_obs_arr = np.array(next_obs, dtype=np.float32)  # shape (M, obs_dim)
            next_obs_padded[:M] = next_obs_arr
            mask_next[:M] = 1.0

        data = (
            obs_padded,       # shape (max_tasks, obs_dim)
            act_padded,       # shape (max_tasks,)
            reward,           # float
            next_obs_padded,  # shape (max_tasks, obs_dim)
            done,             # bool
            mask_obs,         # shape (max_tasks,)
            mask_next         # shape (max_tasks,)
        )

        if len(self.buffer) < self.capacity:
            self.buffer.append(data)
        else:
            self.buffer[self.pos] = data
            self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size):
        """
        Returns a batch of transitions, each shaped for a standard RL update.
        Specifically:
            obs_b: [B, max_tasks, obs_dim]
            act_b: [B, max_tasks]
            rew_b: [B]
            next_obs_b: [B, max_tasks, obs_dim]
            done_b: [B]
            mask_obs_b: [B, max_tasks]
            mask_next_b: [B, max_tasks]
        """
        idx = np.random.choice(len(self.buffer), size=batch_size, replace=False)
        batch = [self.buffer[i] for i in idx]

        # Unpack: each element is (obs_padded, act_padded, rew, next_obs_padded, done, mask_obs, mask_next)
        obs_list, act_list, rew_list, next_obs_list, done_list, mask_obs_list, mask_next_list = zip(*batch)

        obs_b = np.stack(obs_list, axis=0)         # [B, max_tasks, obs_dim]
        act_b = np.stack(act_list, axis=0)         # [B, max_tasks]
        rew_b = np.array(rew_list, dtype=np.float32)  # [B]
        next_obs_b = np.stack(next_obs_list, axis=0) # [B, max_tasks, obs_dim]
        done_b = np.array(done_list, dtype=np.float32) # [B]
        mask_obs_b = np.stack(mask_obs_list, axis=0)   # [B, max_tasks]
        mask_next_b = np.stack(mask_next_list, axis=0) # [B, max_tasks]

        return (
            torch.from_numpy(obs_b),
            torch.from_numpy(act_b),
            torch.from_numpy(rew_b),
            torch.from_numpy(next_obs_b),
            torch.from_numpy(done_b),
            torch.from_numpy(mask_obs_b),
            torch.from_numpy(mask_next_b),
        )

    def __len__(self):
        return len(self.buffer)

class FastReplayBuffer:
    def __init__(self, capacity=100_000, max_tasks=32, obs_dim=7):
        self.capacity = capacity
        self.max_tasks = max_tasks
        self.obs_dim = obs_dim

        self.obs_buf = np.zeros((capacity, max_tasks, obs_dim), dtype=np.float32)
        self.next_obs_buf = np.zeros((capacity, max_tasks, obs_dim), dtype=np.float32)
        self.act_buf = np.full((capacity, max_tasks), -1, dtype=np.int64)
        self.rew_buf = np.zeros((capacity,), dtype=np.float32)
        self.done_buf = np.zeros((capacity,), dtype=np.float32)
        self.mask_obs_buf = np.zeros((capacity, max_tasks), dtype=np.float32)
        self.mask_next_buf = np.zeros((capacity, max_tasks), dtype=np.float32)

        self.pos = 0
        self.size = 0

    def add(self, obs, actions, reward, next_obs, done):
        N = len(obs)
        M = len(next_obs)

        obs_padded = np.zeros((self.max_tasks, self.obs_dim), dtype=np.float32)
        next_obs_padded = np.zeros((self.max_tasks, self.obs_dim), dtype=np.float32)
        act_padded = np.full((self.max_tasks,), -1, dtype=np.int64)
        mask_obs = np.zeros((self.max_tasks,), dtype=np.float32)
        mask_next = np.zeros((self.max_tasks,), dtype=np.float32)

        if N > 0:
            obs_arr = np.array(obs, dtype=np.float32)
            act_arr = np.array(actions, dtype=np.int64)
            obs_padded[:N] = obs_arr
            act_padded[:N] = act_arr
            mask_obs[:N] = 1.0

        if M > 0:
            next_obs_arr = np.array(next_obs, dtype=np.float32)
            next_obs_padded[:M] = next_obs_arr
            mask_next[:M] = 1.0

        self.obs_buf[self.pos] = obs_padded
        self.next_obs_buf[self.pos] = next_obs_padded
        self.act_buf[self.pos] = act_padded
        self.rew_buf[self.pos] = reward
        self.done_buf[self.pos] = float(done)
        self.mask_obs_buf[self.pos] = mask_obs
        self.mask_next_buf[self.pos] = mask_next

        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        idx = np.random.randint(0, self.size, size=batch_size)
        obs_b = torch.from_numpy(self.obs_buf[idx])
        act_b = torch.from_numpy(self.act_buf[idx])
        rew_b = torch.from_numpy(self.rew_buf[idx])
        next_obs_b = torch.from_numpy(self.next_obs_buf[idx])
        done_b = torch.from_numpy(self.done_buf[idx])
        mask_obs_b = torch.from_numpy(self.mask_obs_buf[idx])
        mask_next_b = torch.from_numpy(self.mask_next_buf[idx])

        return obs_b, act_b, rew_b, next_obs_b, done_b, mask_obs_b, mask_next_b

    def __len__(self):
        return self.size
    
class PrioritizedReplayBuffer:
    """
    Simple proportional Prioritized Experience Replay (PER) buffer.
    Each transition has a priority. We sample with probability proportional to it.
    Supports variable-length task observations, same as original ReplayBuffer.
    """
    def __init__(self, capacity=100_000, max_tasks=32, obs_dim=7, alpha=0.6):
        self.capacity = capacity
        self.max_tasks = max_tasks
        self.obs_dim = obs_dim
        self.alpha = alpha  # prioritization exponent

        self.buffer = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.pos = 0

    def add(self, obs, actions, reward, next_obs, done):
        N = len(obs)
        M = len(next_obs)

        obs_padded = np.zeros((self.max_tasks, self.obs_dim), dtype=np.float32)
        next_obs_padded = np.zeros((self.max_tasks, self.obs_dim), dtype=np.float32)
        act_padded = np.full(self.max_tasks, -1, dtype=np.int64)
        mask_obs = np.zeros(self.max_tasks, dtype=np.float32)
        mask_next = np.zeros(self.max_tasks, dtype=np.float32)

        if N > 0:
            obs_arr = np.array(obs, dtype=np.float32)
            act_arr = np.array(actions, dtype=np.int64)
            obs_padded[:N] = obs_arr
            act_padded[:N] = act_arr
            mask_obs[:N] = 1.0

        if M > 0:
            next_obs_arr = np.array(next_obs, dtype=np.float32)
            next_obs_padded[:M] = next_obs_arr
            mask_next[:M] = 1.0

        data = (obs_padded, act_padded, reward, next_obs_padded, done, mask_obs, mask_next)

        if len(self.buffer) < self.capacity:
            self.buffer.append(data)
        else:
            self.buffer[self.pos] = data

        # Assign max priority to new sample
        max_prio = self.priorities.max() if len(self.buffer) > 0 else 1.0
        max_prio = self.priorities.max()
        if max_prio == 0:
            max_prio = 1.0
        self.priorities[self.pos] = max_prio

        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        if len(self.buffer) == 0:
            raise ValueError("Buffer is empty")

        prios = self.priorities[:len(self.buffer)]
        probs = (prios + 1e-6) ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        batch = [self.buffer[i] for i in indices]

        obs_list, act_list, rew_list, next_obs_list, done_list, mask_obs_list, mask_next_list = zip(*batch)

        obs_b = torch.tensor(np.stack(obs_list), dtype=torch.float32)
        act_b = torch.tensor(np.stack(act_list), dtype=torch.int64)
        rew_b = torch.tensor(rew_list, dtype=torch.float32)
        next_obs_b = torch.tensor(np.stack(next_obs_list), dtype=torch.float32)
        done_b = torch.tensor(done_list, dtype=torch.float32)
        mask_obs_b = torch.tensor(np.stack(mask_obs_list), dtype=torch.float32)
        mask_next_b = torch.tensor(np.stack(mask_next_list), dtype=torch.float32)

        # Importance-sampling weights
        weights = (len(self.buffer) * probs[indices]) ** (-beta)
        weights /= weights.max()
        weights = torch.tensor(weights, dtype=torch.float32)

        return (
            obs_b, act_b, rew_b, next_obs_b, done_b,
            mask_obs_b, mask_next_b, weights, indices
        )

    def update_priorities(self, indices, priorities):
        for idx, prio in zip(indices, priorities):
            self.priorities[idx] = prio + 1e-6  # small epsilon to avoid 0 priority

    def __len__(self):
        return len(self.buffer)
