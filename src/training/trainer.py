from pathlib import Path
from collections import deque
from tqdm import tqdm
import numpy as np
import torch
from omegaconf import DictConfig
import gymnasium as gym
import wandb

from ..env.breakout import make_env
from ..agents.dqn_agent import DQNAgent
from .rollout import run_episode

from ..utils.logger import get_logger

logger = get_logger(__name__)

class Trainer:
    def __init__(
        self,
        train_env: gym.Env,
        agent: DQNAgent,
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
        # each entry has 'steps' and some of the metrics (plotted by plot.py):
        # [{'steps': 1000, 'loss': 0.12, 'q_value': 0.56}, {'steps': 1043, 'episode_return': 3.0}, ...]

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
        episode_return = 0.0  # raw score (the replay buffer stores the clipped reward)

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
            episode_return += reward
            if done:
                episode_return = 0.0  # warmup episodes are not logged
                obs, _ = self.train_env.reset()
                frames = deque([obs] * 4, maxlen=4)
            else:
                obs = next_obs
                frames.append(obs)

        self.fixed_states = torch.from_numpy(np.stack(fixed_states)).to(self.cfg.device)
        logger.info('Warmup done!')

        # train: continues the episode left by warmup (obs / frames are carried over)
        logger.info('Training started')
        loss_interval, return_interval = [], []
        for _ in tqdm(range(self.cfg.train.total_steps)):
            state = torch.from_numpy(np.stack(frames)).unsqueeze(0).to(self.cfg.device)  # (1, 4, 84, 84)
            action = self.agent.select_action(state)
            # step / store / reset: keep in sync with the warmup loop above
            next_obs, reward, terminated, truncated, _ = self.train_env.step(action)
            done = terminated or truncated
            self.agent.replay_buffer.store(obs, action, reward, done)
            episode_return += reward
            self.steps += 1
            if done:
                # the first episode started in warmup: its return includes the rewards earned there
                self._log_metric(self.steps, episode_return=episode_return)
                wandb.log({'train/episode_return': episode_return}, step=self.steps)
                return_interval.append(episode_return)
                episode_return = 0.0
                obs, _ = self.train_env.reset()
                frames = deque([obs] * 4, maxlen=4)
            else:
                obs = next_obs
                frames.append(obs)

            if self.steps % self.cfg.train.update_interval == 0:
                loss_interval.append(self.agent.update())

            # log
            if self.steps % self.cfg.train.log_interval == 0:
                loss_mean = np.mean(loss_interval)
                loss_interval = []

                # Q values for fixed set
                with torch.no_grad():
                    q_values = torch.max(self.agent.online_model(self.fixed_states), dim=-1).values # (n_fixed_states,)
                    q_values_mean = q_values.mean().item()

                self._log_metric(self.steps, loss=loss_mean, q_value=q_values_mean) # update history
                wandb.log({
                    'train/loss': loss_mean,
                    'train/q_value': q_values_mean,
                    'train/epsilon': self.agent.epsilon,
                }, step=self.steps)

                return_mean = f'{np.mean(return_interval):.1f}' if return_interval else '-'  # no episode ended in this interval
                return_interval = []
                logger.info(f'Step {self.steps} | Q: {q_values_mean:.4f} | Loss: {loss_mean:.4f} | Return: {return_mean} | Epsilon: {self.agent.epsilon:.4f}')

            # valid
            if self.steps % self.cfg.train.valid_interval == 0:
                self.valid()

            # save
            if self.steps % self.cfg.train.save_interval == 0:
                self.save()

        logger.info('Training has ended')

    def valid(self, render_mode: str = 'rgb_array') -> None:
        video_dir = self.output_dir / 'videos'
        val_env = make_env(
            self.cfg.env,
            seed=self.cfg.seed,
            render_mode=render_mode,
            record_video=(render_mode == 'rgb_array'),
            video_folder=video_dir,
            name_prefix=f'{self.steps}',
        )

        try:
            # one env for all episodes: each reset starts with a different number of no-ops
            scores = [
                run_episode(self.agent.online_model, val_env, self.cfg.device)
                for _ in range(self.cfg.train.valid_episodes)
            ]
        finally:
            val_env.close()  # RecordVideo writes the video file on close
        score_mean, score_std = float(np.mean(scores)), float(np.std(scores))
        logger.info(f'Valid score: {score_mean:.1f} ± {score_std:.1f} over {len(scores)} episodes')
        self._log_metric(self.steps, valid_score_mean=score_mean, valid_score_std=score_std)

        metrics = {'valid/score_mean': score_mean, 'valid/score_std': score_std}
        video = video_dir / f'{self.steps}-episode-0.mp4'
        if video.exists():
            metrics['valid/video'] = wandb.Video(str(video), format='mp4')
        wandb.log(metrics, step=self.steps)

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
