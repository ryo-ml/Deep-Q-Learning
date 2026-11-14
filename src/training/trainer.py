from pathlib import Path
from collections import deque
from tqdm import tqdm
import numpy as np
import torch
from omegaconf import DictConfig
import gymnasium as gym

from ..env.breakout import make_env
from ..agents.dqn_agent import BaseAgent
from .rollout import run_episode
from ..utils.visualize import plot_loss, plot_q_value

from ..utils.logger import get_logger

logger = get_logger(__name__)

class Trainer:
    def __init__(
        self,
        train_env: gym.Env,
        agent: BaseAgent,
        cfg: DictConfig,
        output_dir: Path,
        verbose: bool = True,
    ) -> None:
        self.train_env = train_env
        self.agent = agent
        self.cfg = cfg
        self.output_dir = Path(output_dir)

        self.steps = 0
        self.history = []
        # History structure:
        # loss and q_value can be omitted
        # self.history = [
        #     {'steps': 1000, 'loss': 0.1234, 'q_value': 0.5678},
        #     ...
        # ]

        # logging
        logger.info(' Initialization '.center(40, '='))

        logger.info(f"Seed: {cfg.seed}")
        logger.info(f"Agent: {agent.__class__.__name__}")
        logger.info(f"Device: {cfg.device}")
        logger.info(f"Output dir: {self.output_dir}")

        if verbose:
            logger.info(f"Total steps: {cfg.train.total_steps:,} ({cfg.train.total_steps / 1e6:.1f}M)")
            logger.info(f"Warmup steps: {cfg.train.warmup_steps}")
            logger.info(f"Replay buffer capacity : {cfg.replay_buffer.capacity} ({cfg.replay_buffer.capacity / 1e6:.1f}M)")
            logger.info(f"Batch size: {cfg.replay_buffer.batch_size}")
            logger.info(f"lr: {cfg.optimizer.lr} ({cfg.optimizer.lr:.2e})")
            logger.info(f"Update interval: {cfg.train.update_interval}")
            logger.info(f"Epsilon: {cfg.agent.epsilon_start} -> {cfg.agent.epsilon_end} over {cfg.agent.epsilon_annealing:,} ({cfg.agent.epsilon_annealing / 1e6:.1f}M) steps")

        logger.info(' Initialization done '.center(40, '='))

    def _log_metric(self, steps: int, **metrics):
        """
        Updates `self.history`
        """
        log_dict = {'steps': steps}
        log_dict.update(metrics)
        self.history.append(log_dict)

    def train(self) -> None:
        """
        Fills the replay buffer with random actions for `warmup_steps` steps, then trains
        for `total_steps` steps (`self.steps` counts only these). The episode in progress at
        the end of warmup continues into training (no reset at the boundary).
        """
        obs, _ = self.train_env.reset(seed=self.cfg.seed)
        frames = deque([obs] * 4, maxlen=4)

        # warmup: random actions, only fills the replay buffer
        logger.info('Warmup started')
        fixed_states = []
        for _ in range(self.cfg.train.warmup_steps):
            if len(fixed_states) < self.cfg.train.n_fixed_states:
                fixed_states.append(np.stack(frames))  # (4, 84, 84)

            action = self.train_env.action_space.sample()
            # step / store / reset: keep in sync with the train loop below
            next_obs, reward, terminated, truncated, _ = self.train_env.step(action)
            done = terminated or truncated
            self.agent.replay_buffer.store(obs, action, reward, done)
            if done:
                obs, _ = self.train_env.reset()
                frames = deque([obs] * 4, maxlen=4)
            else:
                obs = next_obs
                frames.append(obs)

        self.fixed_states = torch.from_numpy(np.stack(fixed_states)).to(self.cfg.device)
        logger.info('Warmup done!')

        # train: continues the episode left by warmup (obs / frames are carried over)
        logger.info('Training started')
        loss_interval = []
        for _ in tqdm(range(self.cfg.train.total_steps)):
            state = torch.from_numpy(np.stack(frames)).unsqueeze(0).to(self.cfg.device)  # (1, 4, 84, 84)
            action = self.agent.select_action(state)
            # step / store / reset: keep in sync with the warmup loop above
            next_obs, reward, terminated, truncated, _ = self.train_env.step(action)
            done = terminated or truncated
            self.agent.replay_buffer.store(obs, action, reward, done)
            if done:
                obs, _ = self.train_env.reset()
                frames = deque([obs] * 4, maxlen=4)
            else:
                obs = next_obs
                frames.append(obs)

            self.steps += 1
            if self.steps % self.cfg.train.update_interval == 0:
                loss_interval.append(self.agent.update())

            # log
            if self.steps % self.cfg.train.log_interval == 0:
                loss_mean = np.mean(loss_interval)
                loss_interval = []

                # Q values for fixed set
                with torch.no_grad():
                    q_values = torch.max(self.agent.model(self.fixed_states), dim=-1).values # (n_fixed_states,)
                    q_values_mean = q_values.mean().item()

                self._log_metric(self.steps, loss=loss_mean, q_value=q_values_mean) # update history

                logger.info(f'Step {self.steps} | Q: {q_values_mean:.4f} | Loss: {loss_mean:.4f} | Epsilon: {self.agent.epsilon:.4f}')

            # valid
            if self.steps % self.cfg.train.valid_interval == 0:
                self.valid()

            # save
            if self.steps % self.cfg.train.save_interval == 0:
                self.save()

        logger.info('Training has ended')

        self.plot_results()

    def plot_results(self):
        fig_dir = self.output_dir / 'figures'
        fig_dir.mkdir(parents=True, exist_ok=True)

        # plot loss and q values
        loss_steps = [h['steps'] for h in self.history if 'loss' in h]
        loss = [h['loss'] for h in self.history if 'loss' in h]
        q_value_steps = [h['steps'] for h in self.history if 'q_value' in h]
        q_values = [h['q_value'] for h in self.history if 'q_value' in h]
        plot_loss(loss_steps, loss, fig_dir / f'loss_{self.steps}.svg')
        plot_q_value(q_value_steps, q_values, fig_dir / f'q_value_{self.steps}.svg')

        logger.info(f'Figures saved under {fig_dir}')

    def valid(self, render_mode: str = 'rgb_array') -> None:
        video_dir = self.output_dir / 'videos'
        val_env = make_env(
            self.cfg.env,
            render_mode=render_mode,
            record_video=render_mode == 'rgb_array',
            video_folder=video_dir,
            name_prefix=f'{self.steps}',
        )

        try:
            score = run_episode(self.agent.model, val_env, self.cfg.device)
        finally:
            val_env.close()
        logger.info(f'Valid score: {score}')

    def save(self) -> None:
        d = {
            'steps': self.steps,
            'history': self.history,
            'agent': self.agent.state_dict(),
        }

        save_dir = self.output_dir / 'checkpoints'
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / f'steps_{self.steps}.pth'
        torch.save(d, save_path)
        logger.info(f'Checkpoint saved: {save_path}')

    def load(self, path: Path):
        d = torch.load(path, map_location=self.cfg.device, weights_only=False)
        self.steps = d['steps']
        self.history = d['history']
        self.agent.load_state_dict(d['agent'])

        logger.info(f'Checkpoint loaded from {path} at step {self.steps}')
