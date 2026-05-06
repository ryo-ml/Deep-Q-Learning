from typing import Any
import torch
import numpy as np

class ReplayBuffer:
    def __init__(
        self,
        capacity: int,
        batch_size: int,
        device: torch.device,
    ) -> None:
        self.replay_buffer = {
            'obs': torch.zeros(capacity, 4, 84, 84, dtype=torch.uint8),
            'action': torch.zeros(capacity, dtype=torch.uint8),
            'reward': torch.zeros(capacity, dtype=torch.int8),
            'next_obs': torch.zeros(capacity, 4, 84, 84, dtype=torch.uint8),
            'done': torch.zeros(capacity, dtype=torch.bool),
        }

        self.capacity = capacity
        self.idx = 0
        self.size = 0
        self.batch_size = batch_size
        self.device = device

    def sample(self) -> dict[str, torch.Tensor]:
        idxs = np.random.choice(self.size, self.batch_size, replace=False)
        return {
            'obs': self.replay_buffer['obs'][idxs].to(self.device),
            'action': self.replay_buffer['action'][idxs].long().to(self.device),
            'reward': self.replay_buffer['reward'][idxs].float().to(self.device),
            'next_obs': self.replay_buffer['next_obs'][idxs].to(self.device),
            'done': self.replay_buffer['done'][idxs].float().to(self.device),
        }

    def store(
        self,
        obs: torch.Tensor,
        action: int,
        reward: float,
        next_obs: torch.Tensor,
        done: bool,
    ) -> None:
        if not isinstance(obs, torch.Tensor):
            obs = torch.from_numpy(np.array(obs))
        if not isinstance(next_obs, torch.Tensor):
            next_obs = torch.from_numpy(np.array(next_obs))

        if reward > 0:
            reward = 1
        elif reward < 0:
            reward = -1

        self.replay_buffer['obs'][self.idx] = obs.squeeze().to(torch.uint8)
        self.replay_buffer['action'][self.idx] = action
        self.replay_buffer['reward'][self.idx] = reward
        self.replay_buffer['next_obs'][self.idx] = next_obs.squeeze().to(torch.uint8)
        self.replay_buffer['done'][self.idx] = done

        self.idx = (self.idx + 1) % self.capacity
        if self.size < self.capacity:
            self.size += 1
