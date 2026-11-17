from pathlib import Path
import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing, RecordVideo
import ale_py

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
    Creates the env with the Nature DQN preprocessing: each step repeats the action
    for `frameskip` frames, max-pools the last two frames (removes Atari flicker),
    converts to grayscale and resizes to 84x84 (uint8).

    Args:
        env_cfg: `cfg.env` (name, frameskip, repeat_action_probability, noop_max)
        seed: seeds the RNGs that `set_seed` does not cover: `env.action_space` (all random
            actions in warmup and epsilon-greedy are drawn from it) and the env itself
            (the number of no-op actions at reset).
    """
    gym.register_envs(ale_py)

    env = gym.make(
        id=env_cfg.name,
        render_mode=render_mode,
        frameskip=1,  # frame skipping is done by AtariPreprocessing so it can max-pool frames
        repeat_action_probability=env_cfg.repeat_action_probability,
    )
    env = AtariPreprocessing(
        env,
        noop_max=env_cfg.noop_max,     # random number (0 .. noop_max) of no-op actions at reset
        frame_skip=env_cfg.frameskip,
        screen_size=84,
        terminal_on_life_loss=False,
        grayscale_obs=True,
        scale_obs=False,               # keep uint8; DQN.forward scales to [0, 1]
    )  # obs: (84, 84) uint8

    # Seed before RecordVideo so this reset is not recorded as an (empty) episode
    if seed is not None:
        env.action_space.seed(seed)
        env.reset(seed=seed)  # seeds the env's own RNG; later reset() calls continue from it

    if render_mode == 'rgb_array' and record_video:
        if video_folder is None:
            raise ValueError("video_folder must be specified when record_video=True")

        env = RecordVideo(
            env,
            video_folder=video_folder,
            episode_trigger=lambda e: e == 0,  # record only the first episode
            name_prefix=name_prefix,
        )

    return env
