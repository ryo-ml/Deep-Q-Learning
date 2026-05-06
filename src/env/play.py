import gymnasium as gym
from gymnasium.utils.play import play
import ale_py

gym.register_envs(ale_py)

env = gym.make("ALE/Breakout-v5", render_mode='rgb_array')
obs, info = env.reset()

mapping = {
    (ord(" "),): 1,  # スペースキーで発射
    (ord("d"),): 2,  # Dキーで右
    (ord("a"),): 3,  # Aキーで左
}

play(env, keys_to_action=mapping, fps=15)

env.close()
