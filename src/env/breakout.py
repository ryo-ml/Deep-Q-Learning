from pathlib import Path
import gymnasium as gym
from gymnasium.wrappers import GrayscaleObservation, ResizeObservation, TransformObservation, RecordVideo
import ale_py

import numpy as np
from omegaconf import DictConfig

def make_env(
    env_cfg: DictConfig,
    seed: int | None = None,
    render_mode: str = 'rgb_array',
    record_video: bool = False,
    video_folder: Path | None = None,
    name_prefix: str | None = None,
) -> gym.Env:
    """
    Args:
        env_cfg: `cfg.env` (name, frameskip, repeat_action_probability)
        seed: seeds `env.action_space`, which has its own RNG that `set_seed` does not cover.
            All random actions (warmup, epsilon-greedy) are drawn from it.
    """
    gym.register_envs(ale_py)

    env = gym.make(
        id=env_cfg.name,
        render_mode=render_mode,
        frameskip=env_cfg.frameskip,
        repeat_action_probability=env_cfg.repeat_action_probability,
    )
    env = GrayscaleObservation(env)
    env = ResizeObservation(env, (110, 84))
    env = TransformObservation(
        env,
        lambda obs: obs[18:102, :],
        gym.spaces.Box(low=0, high=255, shape=(84, 84), dtype=np.uint8)
    )

    if render_mode == 'rgb_array' and record_video:
        if video_folder is None:
            raise ValueError("video_folder must be specified when record_video=True")

        env = RecordVideo(
            env,
            video_folder=video_folder,
            episode_trigger=lambda e: True,
            name_prefix=name_prefix,
        )

    if seed is not None:
        env.action_space.seed(seed)

    return env
