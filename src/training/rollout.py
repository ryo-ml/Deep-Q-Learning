from collections import deque

import numpy as np
import torch
import gymnasium as gym

from ..models.dqn import DQN

@torch.inference_mode()
def run_episode(
    model: DQN,
    env: gym.Env,
    device: str,
    epsilon: float = 0.05,
) -> float:
    """
    Plays one episode with an epsilon-greedy policy and returns the total reward.
    Used for validation during training and for evaluation; nothing is stored or learned.
    """
    obs, _ = env.reset()
    frames = deque([obs.copy() for _ in range(4)], maxlen=4)
    total_reward, done = 0.0, False

    while not done:
        if np.random.uniform() < epsilon:
            action = int(env.action_space.sample())
        else:
            x = torch.tensor(np.stack(frames), dtype=torch.uint8).unsqueeze(0).to(device)
            action = int(model(x).argmax().item())

        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        done = terminated or truncated
        frames.append(obs)

    return total_reward
