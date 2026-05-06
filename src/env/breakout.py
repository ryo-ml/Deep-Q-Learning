from pathlib import Path
import gymnasium as gym
from gymnasium.wrappers import GrayscaleObservation, ResizeObservation, TransformObservation, FrameStackObservation, RecordVideo
import ale_py

import numpy as np

def make_env(
    render_mode: str = 'rgb_array',
    record_video: bool = False,
    video_folder: Path | None = None,
    name_prefix: str | None = None,
) -> gym.Env:
    gym.register_envs(ale_py)

    env = gym.make("ALE/Breakout-v5", render_mode=render_mode)
    env = GrayscaleObservation(env)
    env = ResizeObservation(env, (110, 84))
    env = TransformObservation(
        env, 
        lambda obs: obs[18:102, :],
        gym.spaces.Box(low=0, high=255, shape=(84, 84), dtype=np.uint8)
    )
    env = FrameStackObservation(env, 4)

    if render_mode == 'rgb_array' and record_video:
        if video_folder is None:
            raise ValueError("video_folder must be specified when record_video=True")

        env = RecordVideo(
            env,
            video_folder=video_folder,
            episode_trigger=lambda e: True,
            name_prefix=name_prefix,
        )

    return env
