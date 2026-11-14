import torch
import numpy as np

class ReplayBuffer:
    # Number of frames stacked into one observation. Must match the frame stack in the
    # training loop / run_episode (deque maxlen) and DQN's input channels; not a tunable setting.
    FRAME_STACK = 4

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
        # Sample t such that frames t-(k-1) .. t+1 are all stored and in time order,
        # where k = FRAME_STACK (frames t-3 .. t+1 for k = 4). high is exclusive in randint.
        k = self.FRAME_STACK
        if self.size < self.capacity:
            low = k - 1                          # t-(k-1) >= 0        (oldest frame)
            high = self.size - 1                 # t+1     <= size - 1 (newest frame)
        else:
            low = self.idx + k - 1               # t-(k-1) >= idx      (oldest frame)
            high = self.capacity + self.idx - 1  # t+1     <= idx - 1  (newest frame), wraps around

        idxs = torch.randint(low, high, (self.batch_size, 1)) % self.capacity

        valid_obs_idxs = self._stack_indices(idxs)          # frames t-(k-1) .. t
        valid_next_obs_idxs = self._stack_indices(idxs + 1) # frames t-(k-2) .. t+1

        return {
            'obs': self.replay_buffer['obs'][valid_obs_idxs].to(self.device),
            'action': self.replay_buffer['action'][idxs.squeeze(1)].long().to(self.device),
            'reward': self.replay_buffer['reward'][idxs.squeeze(1)].float().to(self.device),
            'next_obs': self.replay_buffer['obs'][valid_next_obs_idxs].to(self.device),
            'done': self.replay_buffer['done'][idxs.squeeze(1)].float().to(self.device),
        }

    def _stack_indices(self, idxs: torch.Tensor) -> torch.Tensor:
        """
        Returns buffer indices of the FRAME_STACK (= k) frames stacked into the observation at each t.

        Frames t-(k-1) .. t are used, but frames that belong to a previous episode are
        replaced by the first frame of the episode containing t (the same padding the
        training loop uses after env.reset()).

        Example (k = 4, t = 5):
            slots: [2, 3, 4, 5], done: [0, 1, 0, 0] -> [4, 4, 4, 5]  (episode starts at 4)
            slots: [2, 3, 4, 5], done: [0, 0, 0, 1] -> [2, 3, 4, 5]  (done at t is irrelevant)

        Args:
            idxs: (batch_size, 1) indices t
        Returns:
            (batch_size, k) indices, oldest frame first
        """
        k = self.FRAME_STACK
        offsets = torch.arange(-(k - 1), 1)                          # (k,) = [-3, -2, -1, 0] for k = 4
        all_idxs = (idxs + offsets) % self.capacity                  # (batch_size, k)

        # A done at t-(k-1) .. t-1 means the episode containing t starts right after it.
        # A done at t itself only says the episode ends after t, so it is ignored.
        dones = self.replay_buffer['done'][all_idxs[:, :-1]]         # (batch_size, k-1)
        has_done = dones.any(dim=1)                                  # (batch_size,)

        # position (0 .. k-2) of the rightmost done: flip, take the first True, map back
        last_done = (k - 2) - dones.flip(dims=[1]).int().argmax(dim=1)  # (batch_size,)
        # position (0 .. k-1) of the first frame of t's episode; 0 when there is no done
        start = torch.where(has_done, last_done + 1, 0)              # (batch_size,)

        # replace frames before `start` with the frame at `start`
        start_idxs = all_idxs.gather(1, start.unsqueeze(1))          # (batch_size, 1)
        before_start = torch.arange(k) < start.unsqueeze(1)          # (batch_size, k)
        return torch.where(before_start, start_idxs, all_idxs)       # (batch_size, k)

    def store(
        self,
        obs: np.ndarray, # (84, 84)
        action: int,
        reward: float,
        done: bool,
    ) -> None:
        obs = torch.from_numpy(obs)
        reward = int(np.sign(reward))

        self.replay_buffer['obs'][self.idx] = obs.to(torch.uint8) # (84, 84)
        self.replay_buffer['action'][self.idx] = action
        self.replay_buffer['reward'][self.idx] = reward
        self.replay_buffer['done'][self.idx] = done

        self.idx = (self.idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
