# manager_worker_buffers.py  (sample() no .to() / no device arg)
import numpy as np
import torch


class ManagerReplayBuffer:
    """Replay buffer for Manager transitions (embedding format)."""

    def __init__(self,
                 capacity: int,
                 D_emb_meta_manager: int,
                 D_global: int,
                 D_option_feat: int,
                 max_total_options: int):
        self.capacity = capacity
        self.pos = 0
        self.size = 0

        opt_shape = (capacity, max_total_options, D_option_feat)
        mask_shape = (capacity, max_total_options)

        self.meta   = np.zeros((capacity, D_emb_meta_manager), dtype=np.float32)
        self.global_ = np.zeros((capacity, D_global), dtype=np.float32)
        self.opt    = np.zeros(opt_shape, dtype=np.float32)
        self.mask   = np.zeros(mask_shape, dtype=np.bool_)
        self.act    = np.zeros((capacity,), dtype=np.int64)
        self.rew    = np.zeros((capacity,), dtype=np.float32)
        self.done   = np.zeros((capacity,), dtype=np.float32)

        # next state
        self.meta_n  = np.zeros_like(self.meta)
        self.global_n= np.zeros_like(self.global_)
        self.opt_n   = np.zeros_like(self.opt)
        self.mask_n  = np.zeros_like(self.mask)

    def add(self, meta, global_, opt, mask,
            act, rew, done,
            meta_n, global_n, opt_n, mask_n):
        i = self.pos
        self.meta[i]  = meta
        self.global_[i] = global_
        self.opt[i]   = opt
        self.mask[i]  = mask
        self.act[i]   = act
        self.rew[i]   = rew
        self.done[i]  = done
        self.meta_n[i]= meta_n
        self.global_n[i]= global_n
        self.opt_n[i] = opt_n
        self.mask_n[i]= mask_n
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        idx = np.random.randint(0, self.size, size=batch_size)
        f = lambda arr, dtype=None: torch.from_numpy(arr[idx]).type(dtype) \
            if dtype is not None else torch.from_numpy(arr[idx])

        return (
            f(self.meta),
            f(self.global_),
            f(self.opt),
            f(self.mask, torch.bool),
            f(self.act, torch.long),
            f(self.rew).unsqueeze(1),
            f(self.done).unsqueeze(1),
            f(self.meta_n),
            f(self.global_n),
            f(self.opt_n),
            f(self.mask_n, torch.bool),
        )

    def __len__(self):
        return self.size


class WorkerReplayBuffer:
    """Replay buffer for πWorker transitions (embedding format)."""

    def __init__(self,
                 capacity: int,
                 D_emb_meta_worker: int,
                 D_emb_local_worker: int,
                 D_global: int):
        self.capacity = capacity
        self.pos = 0
        self.size = 0

        self.meta   = np.zeros((capacity, D_emb_meta_worker), dtype=np.float32)
        self.local  = np.zeros((capacity, D_emb_local_worker), dtype=np.float32)
        self.global_= np.zeros((capacity, D_global), dtype=np.float32)
        self.act    = np.zeros((capacity,), dtype=np.int64)
        self.rew    = np.zeros((capacity,), dtype=np.float32)
        self.done   = np.zeros((capacity,), dtype=np.float32)

        # next state
        self.meta_n  = np.zeros_like(self.meta)
        self.local_n = np.zeros_like(self.local)
        self.global_n= np.zeros_like(self.global_)

    def add(self, meta, local, global_,
            act, rew, done,
            meta_n, local_n, global_n):
        i = self.pos
        self.meta[i]   = meta
        self.local[i]  = local
        self.global_[i]= global_
        self.act[i]    = act
        self.rew[i]    = rew
        self.done[i]   = done
        self.meta_n[i] = meta_n
        self.local_n[i]= local_n
        self.global_n[i]= global_n
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        idx = np.random.randint(0, self.size, size=batch_size)
        f = lambda arr, dtype=None: torch.from_numpy(arr[idx]).type(dtype) \
            if dtype is not None else torch.from_numpy(arr[idx])

        return (
            f(self.meta),
            f(self.local),
            f(self.global_),
            f(self.act, torch.long),
            f(self.rew).unsqueeze(1),
            f(self.done).unsqueeze(1),
            f(self.meta_n),
            f(self.local_n),
            f(self.global_n),
        )

    def __len__(self):
        return self.size
