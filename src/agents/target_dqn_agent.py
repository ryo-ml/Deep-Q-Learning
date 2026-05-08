from typing import Any
import numpy as np
import torch

from .agent import BaseAgent
from ..models.dqn import DQN
from ..buffer.replay_buffer import ReplayBuffer

class TargetDQNAgent(BaseAgent):
    """
    Single Q network WITH target network
    """
    def __init__(
        self,
        cfg: dict[str, Any],
        n_actions: int,
    ) -> None:
        self.cfg = cfg
        self.n_actions = n_actions
        self.online_model = DQN(n_actions).to(cfg['device'])
        self.target_model = DQN(n_actions).to(cfg['device'])
        self.target_model.load_state_dict(self.online_model.state_dict())

        self.replay_buffer = ReplayBuffer(
            cfg['replay_buffer']['capacity'],
            cfg['replay_buffer']['batch_size'],
            cfg['device'],
        )

        optim_type = {
            'adam': torch.optim.Adam,
            'rmsprop': torch.optim.RMSprop,
        }
        optim_cfg = {k: v for k, v in cfg['optimizer'].items() if k != 'type'}
        self.optim = optim_type[cfg['optimizer']['type']](
            self.online_model.parameters(),
            **optim_cfg,
        )

        loss_type = {
            'mse': torch.nn.MSELoss,
        }
        self.loss_func = loss_type[cfg['model']['loss']]()

        self.gamma = cfg['agent']['gamma']
        self.epsilon_start = cfg['agent']['epsilon_start']
        self.epsilon_end = cfg['agent']['epsilon_end']
        self.epsilon_annealing = cfg['agent']['epsilon_annealing']

        self.env_steps = 0
        self.update_steps = 0

    @property
    def model(self) -> DQN:
        return self.online_model

    def select_action(self, obs: torch.Tensor) -> int:
        self.epsilon = max(
            self.epsilon_end,
            self.epsilon_start - (self.epsilon_start - self.epsilon_end) * self.update_steps / self.epsilon_annealing
        )

        if np.random.uniform() < self.epsilon:
            action = int(np.random.choice(self.n_actions, size=1).squeeze(0))
        else:
            with torch.no_grad():
                q_values = self.online_model(obs)
                action = int(torch.argmax(q_values))

        self.env_steps += 1

        return action

    def update(self) -> float:
        mini_batch = self.replay_buffer.sample()

        obs = mini_batch['obs']
        action = mini_batch['action']
        reward = mini_batch['reward']
        next_obs = mini_batch['next_obs']
        done = mini_batch['done']

        with torch.no_grad():
            next_q_values = torch.max(self.target_model(next_obs), dim=1).values # (B,)
            y = reward + (1 - done) * self.gamma * next_q_values # (B, )

        preds = self.online_model(obs) # (B, n_actions)
        q_values = torch.gather(preds, dim=1, index=action.unsqueeze(1)).squeeze(1) # (B,)

        self.optim.zero_grad()
        loss = self.loss_func(q_values, y)
        loss.backward()
        # update online network
        self.optim.step()

        self.update_steps += 1

        # copy parameters of online network to those of target network
        if self.update_steps % self.cfg['train']['target_update_interval'] == 0:
            self.target_model.load_state_dict(self.online_model.state_dict())

        return loss.item()

    def state_dict(self) -> dict:
        return {
            'env_steps': self.env_steps,
            'update_steps': self.update_steps,
            'online_model': self.online_model.state_dict(),
            'target_model': self.target_model.state_dict(),
            'optim': self.optim.state_dict(),
        }

    def load_state_dict(self, state_dict: dict):
        self.env_steps = state_dict['env_steps']
        self.update_steps = state_dict['update_steps']
        self.online_model.load_state_dict(state_dict['online_model'])
        self.target_model.load_state_dict(state_dict['target_model'])
        self.optim.load_state_dict(state_dict['optim'])
