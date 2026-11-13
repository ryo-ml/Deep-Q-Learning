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

        valid_obs_idxs = self._stack_indices(idxs)          # frames t-3 .. t
        valid_next_obs_idxs = self._stack_indices(idxs + 1) # frames t-2 .. t+1

        return {
            'obs': self.replay_buffer['obs'][valid_obs_idxs].to(self.device),
            'action': self.replay_buffer['action'][idxs.squeeze(1)].long().to(self.device),
            'reward': self.replay_buffer['reward'][idxs.squeeze(1)].float().to(self.device),
            'next_obs': self.replay_buffer['obs'][valid_next_obs_idxs].to(self.device),
            'done': self.replay_buffer['done'][idxs.squeeze(1)].float().to(self.device),
        }

    def _stack_indices(self, idxs: torch.Tensor) -> torch.Tensor:
        """
        Returns buffer indices of the 4 frames stacked into the observation at each t.

        Frames t-3 .. t are used, but frames that belong to a previous episode are
        replaced by the first frame of the episode containing t (the same padding the
        training loop uses after env.reset()).

        Example (t = 5):
            slots: [2, 3, 4, 5], done: [0, 1, 0, 0] -> [4, 4, 4, 5]  (episode starts at 4)
            slots: [2, 3, 4, 5], done: [0, 0, 0, 1] -> [2, 3, 4, 5]  (done at t is irrelevant)

        Args:
            idxs: (batch_size, 1) indices t
        Returns:
            (batch_size, 4) indices, oldest frame first
        """
        offsets = torch.arange(-3, 1)                                # (4,) = [-3, -2, -1, 0]
        all_idxs = (idxs + offsets) % self.capacity                  # (batch_size, 4)

        # A done at t-3 .. t-1 means the episode containing t starts right after it.
        # A done at t itself only says the episode ends after t, so it is ignored.
        dones = self.replay_buffer['done'][all_idxs[:, :3]]          # (batch_size, 3)
        has_done = dones.any(dim=1)                                  # (batch_size,)

        # position (0..2) of the rightmost done: flip, take the first True, map back
        last_done = 2 - dones.flip(dims=[1]).int().argmax(dim=1)     # (batch_size,)
        # position (0..3) of the first frame of t's episode; 0 when there is no done
        start = torch.where(has_done, last_done + 1, 0)              # (batch_size,)

        # replace frames before `start` with the frame at `start`
        start_idxs = all_idxs.gather(1, start.unsqueeze(1))          # (batch_size, 1)
        before_start = torch.arange(4) < start.unsqueeze(1)          # (batch_size, 4)
        return torch.where(before_start, start_idxs, all_idxs)       # (batch_size, 4)

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
