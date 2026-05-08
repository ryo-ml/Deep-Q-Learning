from typing import Any
from pathlib import Path
from tqdm import tqdm
import numpy as np
import torch
import gymnasium as gym

from ..env.breakout import make_env
from ..agents.dqn_agent import BaseAgent
from ..utils.logger import get_logger

logger = get_logger(__name__)

class Trainer:
    def __init__(
        self,
        train_env: gym.Env,
        agent: BaseAgent,
        cfg: dict[str, Any],
        verbose: bool = True,
    ) -> None:
        self.train_env = train_env
        self.agent = agent
        self.cfg = cfg

        self.steps = 0
        self.q_values = []
        self.loss = []

        # logging
        logger.info(' Initialization '.center(40, '='))

        logger.info(f"Seed: {cfg['seed']}")
        logger.info(f"Agent: {agent.__class__.__name__}")
        logger.info(f"Device: {cfg['device']}")
        
        if verbose:
            logger.info(f"Total steps: {cfg['train']['total_steps']:,} ({cfg['train']['total_steps'] / 1e6:.1f}M)")
            logger.info(f"Warmup steps: {cfg['train']['warmup_steps']}")
            logger.info(f"Replay buffer capacity : {cfg['replay_buffer']['capacity']} ({cfg['replay_buffer']['capacity'] / 1e6:.1f}M)")
            logger.info(f"Batch size: {cfg['replay_buffer']['batch_size']}")
            logger.info(f"lr: {cfg['optimizer']['lr']} ({cfg['optimizer']['lr']:.2e})")
            logger.info(f"Update interval: {cfg['train']['update_interval']}")
            logger.info(f"Epsilon: {cfg['agent']['epsilon_start']} -> {cfg['agent']['epsilon_end']} over {cfg['agent']['epsilon_annealing']:,} ({cfg['agent']['epsilon_annealing'] / 1e6:.1f}M) steps")

        logger.info(' Initialization done '.center(40, '='))

    def warmup(self) -> None:
        """
        Fills replay buffer with transitions using random actions.
        """
        logger.info('Warmup started')

        fixed_states = []
        obs, _ = self.train_env.reset(seed=self.cfg['seed'])
        while self.agent.replay_buffer.size < self.cfg['train']['warmup_steps']:
            action = self.train_env.action_space.sample() # random action
            next_obs, reward, terminated, truncated, _ = self.train_env.step(action)
            done = terminated or truncated

            if len(fixed_states) < self.cfg['train']['n_fixed_states']:
                fixed_states.append(obs)

            self.agent.replay_buffer.store(obs, action, reward, next_obs, done)

            if done:
                obs, _ = self.train_env.reset()
            else:
                obs = next_obs

        self.fixed_states = torch.tensor(np.array(fixed_states), dtype=torch.uint8).to(self.cfg['device'])

        logger.info('Warmup done!')
 
    def train(self) -> None:
        self.warmup()

        pbar = tqdm(total=self.cfg['train']['total_steps'])
        while self.steps < self.cfg['train']['total_steps']:
            obs, _ = self.train_env.reset()
            done = False
            while not done and self.steps < self.cfg['train']['total_steps']:
                obs_tensor = torch.tensor(obs, dtype=torch.uint8).unsqueeze(0).to(self.cfg['device'])
                action = self.agent.select_action(obs_tensor)
                next_obs, reward, terminated, truncated, _ = self.train_env.step(action)
                done = terminated or truncated
                self.agent.replay_buffer.store(obs, action, reward, next_obs, done)
                obs = next_obs

                assert self.agent.replay_buffer.size >= self.cfg['replay_buffer']['batch_size'], \
                    f'Replay buffer size {self.agent.replay_buffer.size} is less than batch size {self.cfg["replay_buffer"]["batch_size"]}'

                self.steps += 1
                if self.steps % self.cfg['train']['update_interval'] == 0:
                    loss_batch = self.agent.update()
                    self.loss.append(loss_batch)

                pbar.update(1)
                pbar.set_description(f'Steps: {self.steps}')

                # log
                if self.steps % self.cfg['train']['log_interval'] == 0:
                    _n_loss = self.cfg['train']['log_interval'] // self.cfg['train']['update_interval']
                    loss_mean = np.mean(self.loss[-_n_loss:])
                    # Q values for fixed set
                    with torch.no_grad():
                        q_values = torch.max(self.agent.model(self.fixed_states), dim=-1).values # (n_fixed_states,)
                        q_values_mean = q_values.mean().item()
                        self.q_values.append(q_values_mean)

                    logger.info(f'Step {self.steps} | Q: {q_values_mean:.4f} | Loss: {loss_mean:.4f} | Epsilon: {self.agent.epsilon:.4f}')

                # valid
                if self.steps % self.cfg['train']['valid_interval'] == 0:
                    self.valid()

                # save
                if self.steps % self.cfg['train']['save_interval'] == 0:
                    self.save()

    def valid(self, render_mode: str = 'rgb_array') -> None:
        val_env = make_env(
            render_mode=render_mode,
            record_video=render_mode == 'rgb_array',
            video_folder=Path(__file__).resolve().parent.parent.parent / 'videos',
            name_prefix=f'{self.steps}',
        )
        try:
            obs, _ = val_env.reset()
            done = False
            while not done:
                obs_tensor = torch.tensor(obs, dtype=torch.uint8).unsqueeze(0).to(self.cfg['device'])

                if np.random.uniform() < 0.05:
                    action = int(val_env.action_space.sample())
                else:
                    with torch.no_grad():
                        q_values = self.agent.model(obs_tensor)
                        action = int(torch.argmax(q_values).item())

                next_obs, _, terminated, truncated, _ = val_env.step(action)
                done = terminated or truncated
                obs = next_obs
        finally:
            val_env.close()

    def save(self) -> None:
        d = {
            'steps': self.steps,
            'q_values': self.q_values,
            'loss': self.loss,
            'agent': self.agent.state_dict(),
        }

        save_dir = Path(__file__).resolve().parent.parent.parent / 'checkpoints'
        save_dir.mkdir(exist_ok=True)
        save_path = save_dir / f'steps_{self.steps}.pth'
        torch.save(d, save_path)
        logger.info(f'Checkpoint saved: {save_path}')

    def load(self, path: Path):
        d = torch.load(path, map_location=self.cfg['device'])
        self.steps = d['steps']
        self.q_values = d['q_values']
        self.loss = d['loss']
        self.agent.load_state_dict(d['agent'])

        logger.info(f'Checkpoint loaded from {path} at step {self.steps}')
