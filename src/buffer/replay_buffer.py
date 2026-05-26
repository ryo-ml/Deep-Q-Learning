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
            'obs': torch.zeros(capacity, 84, 84, dtype=torch.uint8),
            'action': torch.zeros(capacity, dtype=torch.uint8),
            'reward': torch.zeros(capacity, dtype=torch.int8),
            'done': torch.zeros(capacity, dtype=torch.bool),
        }

        self.capacity = capacity
        self.idx = 0
        self.size = 0
        self.batch_size = batch_size
        self.device = device

    def sample(self) -> dict[str, torch.Tensor]:
        if self.size < self.capacity:
            low = 3
            high = self.size - 1
        else:
            low = self.idx + 1
            high = self.idx + self.capacity - 4

        idxs = torch.from_numpy(
            np.random.choice(np.arange(high - low) % self.capacity, self.batch_size, replace=False),
        ).view(-1, 1)

        def get_valid_indices(idxs: torch.Tensor) -> torch.Tensor:
            """
            obs: [2, 3, 4, 5], done: [0, 1, 0, 0] -> obs: [4, 4, 4, 5]
            """
            offsets = torch.arange(-3, 1) # (4,)
            all_idxs = (idxs + offsets) % self.capacity # (batch_size, 4)

            dones = self.replay_buffer['done'][all_idxs] # (batch_size, 4)

            # get indices for done == 1, to the right by 1
            weights = torch.tensor([1, 2, 3, 0]).view(1, -1)
            done_positions = dones * weights

            # get rightmost indices for done == 1, to the right by 1
            max_pos = torch.max(done_positions, dim=1, keepdim=True).values # (batch_size, 1)

            # compare with original indices
            original_idxs = torch.tensor([0, 1, 2, 3]) # (4,)
            # when done_positions == [0, 0, ..., 0], original indices should be used
            final_offsets = torch.max(max_pos, original_idxs) # (batch_size, 4)

            valid_idxs = torch.gather(all_idxs, dim=1, index=final_offsets)

            return valid_idxs

        valid_obs_idxs = get_valid_indices(idxs)
        valid_next_obs_idxs = get_valid_indices(idxs + 1)

        return {
            'obs': self.replay_buffer['obs'][valid_obs_idxs].to(self.device),
            'action': self.replay_buffer['action'][idxs.squeeze(1)].long().to(self.device),
            'reward': self.replay_buffer['reward'][idxs.squeeze(1)].float().to(self.device),
            'next_obs': self.replay_buffer['obs'][valid_next_obs_idxs].to(self.device),
            'done': self.replay_buffer['done'][idxs.squeeze(1)].float().to(self.device),
        }

    def store(
        self,
        obs: np.ndarray, # (84, 84)
        action: int,
        reward: float,
        done: bool,
    ) -> None:
        obs = torch.from_numpy(np.array(obs))

        if reward > 0:
            reward = 1
        elif reward < 0:
            reward = -1

        self.replay_buffer['obs'][self.idx] = obs.unsqueeze(0).to(torch.uint8) # store (1, 84, 84)
        self.replay_buffer['action'][self.idx] = action
        self.replay_buffer['reward'][self.idx] = reward
        self.replay_buffer['done'][self.idx] = done

        self.idx = (self.idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
