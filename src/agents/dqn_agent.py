import numpy as np
import torch
import gymnasium as gym
from hydra.utils import instantiate
from omegaconf import DictConfig

from ..models.dqn import DQN
from ..buffer.replay_buffer import ReplayBuffer

class DQNAgent:
    """
    DQN without a target network.
    TD target: y = r + γ max_a Q_online(s', a)

    Subclasses change the TD target by overriding `compute_next_q_values` and
    add a target network sync by overriding `_update_target_params`.
    """
    def __init__(
        self,
        cfg: DictConfig,
        action_space: gym.spaces.Discrete,
    ) -> None:
        """
        Args:
            action_space: the training env's action space; random actions are sampled from it
                (seeded in make_env)
        """
        self.action_space = action_space
        self.online_model = DQN(action_space.n).to(cfg.device)

        self.replay_buffer = ReplayBuffer(
            cfg.replay_buffer.capacity,
            cfg.replay_buffer.batch_size,
            cfg.device,
        )

        self.optim = instantiate(cfg.optimizer, params=self.online_model.parameters())
        self.loss_func = instantiate(cfg.loss)

        self.gamma = cfg.agent.gamma
        self.epsilon_start = cfg.agent.epsilon_start
        self.epsilon_end = cfg.agent.epsilon_end
        self.epsilon_annealing = cfg.agent.epsilon_annealing

        self.env_steps = 0     # number of select_action calls (warmup steps are not counted)
        self.update_steps = 0  # number of update calls

    def select_action(self, obs: torch.Tensor) -> int:
        """
        Epsilon-greedy action. Epsilon is annealed linearly from epsilon_start to
        epsilon_end over epsilon_annealing env steps.

        Args:
            obs: (1, 4, 84, 84) uint8 stacked frames
        """
        self.epsilon = max(
            self.epsilon_end,
            self.epsilon_start - (self.epsilon_start - self.epsilon_end) * self.env_steps / self.epsilon_annealing
        )

        if np.random.uniform() < self.epsilon:
            action = int(self.action_space.sample())
        else:
            with torch.no_grad():
                q_values = self.online_model(obs)  # (1, n_actions)
                action = int(torch.argmax(q_values))

        self.env_steps += 1

        return action

    def update(self) -> float:
        """
        One gradient step on a mini-batch from the replay buffer. Returns the loss.
        """
        mini_batch = self.replay_buffer.sample()

        obs = mini_batch['obs']            # (B, 4, 84, 84)
        action = mini_batch['action']      # (B,)
        reward = mini_batch['reward']      # (B,)
        next_obs = mini_batch['next_obs']  # (B, 4, 84, 84)
        done = mini_batch['done']          # (B,)

        with torch.no_grad():
            next_q_values = self.compute_next_q_values(next_obs)  # (B,)
            y = reward + (1 - done) * self.gamma * next_q_values  # (B,)

        preds = self.online_model(obs)  # (B, n_actions)
        q_values = torch.gather(preds, dim=1, index=action.unsqueeze(1)).squeeze(1)  # (B,)

        self.optim.zero_grad()
        loss = self.loss_func(q_values, y)
        loss.backward()
        self.optim.step()

        self.update_steps += 1
        self._update_target_params()

        return loss.item()

    def compute_next_q_values(self, next_obs: torch.Tensor) -> torch.Tensor:
        """
        max_a Q_online(s', a)

        Args:
            next_obs: (B, 4, 84, 84)
        Returns:
            (B,)
        """
        return torch.max(self.online_model(next_obs), dim=1).values

    def _update_target_params(self) -> None:
        """Called after every update; does nothing since there is no target network."""
        pass

    def state_dict(self) -> dict:
        return {
            'env_steps': self.env_steps,
            'update_steps': self.update_steps,
            'online_model': self.online_model.state_dict(),
            'optim': self.optim.state_dict(),
        }

    def load_state_dict(self, state_dict: dict) -> None:
        self.env_steps = state_dict['env_steps']
        self.update_steps = state_dict['update_steps']
        self.online_model.load_state_dict(state_dict['online_model'])
        self.optim.load_state_dict(state_dict['optim'])


class TargetDQNAgent(DQNAgent):
    """
    DQN with a target network (Nature DQN).
    TD target: y = r + γ max_a Q_target(s', a)

    The target network is a copy of the online network, synced every
    target_update_interval updates.
    """
    def __init__(
        self,
        cfg: DictConfig,
        action_space: gym.spaces.Discrete,
    ) -> None:
        super().__init__(cfg, action_space)
        self.target_model = DQN(action_space.n).to(cfg.device)
        self.target_model.load_state_dict(self.online_model.state_dict())
        self.target_update_interval = cfg.train.target_update_interval

    def compute_next_q_values(self, next_obs: torch.Tensor) -> torch.Tensor:
        """
        max_a Q_target(s', a)

        Args:
            next_obs: (B, 4, 84, 84)
        Returns:
            (B,)
        """
        return torch.max(self.target_model(next_obs), dim=1).values

    def _update_target_params(self) -> None:
        """Copies the online network's parameters to the target network every target_update_interval updates."""
        if self.update_steps % self.target_update_interval == 0:
            self.target_model.load_state_dict(self.online_model.state_dict())

    def state_dict(self) -> dict:
        return {**super().state_dict(), 'target_model': self.target_model.state_dict()}

    def load_state_dict(self, state_dict: dict) -> None:
        super().load_state_dict(state_dict)
        self.target_model.load_state_dict(state_dict['target_model'])


class DoubleDQNAgent(TargetDQNAgent):
    """
    Double DQN: same as TargetDQNAgent (the target network and its sync are inherited)
    except that the next action is selected with the online network and evaluated
    with the target network.
    TD target: y = r + γ Q_target(s', argmax_a Q_online(s', a))
    """
    def compute_next_q_values(self, next_obs: torch.Tensor) -> torch.Tensor:
        """
        Q_target(s', argmax_a Q_online(s', a))

        Args:
            next_obs: (B, 4, 84, 84)
        Returns:
            (B,)
        """
        next_actions = torch.argmax(self.online_model(next_obs), dim=1)  # (B,)
        next_preds = self.target_model(next_obs)  # (B, n_actions)
        return torch.gather(next_preds, dim=1, index=next_actions.unsqueeze(1)).squeeze(1)  # (B,)
