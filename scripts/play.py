import gymnasium as gym
from gymnasium.utils.play import play
import ale_py

gym.register_envs(ale_py)

env = gym.make("ALE/Breakout-v5", render_mode='rgb_array')
obs, info = env.reset()

mapping = {
    (ord(" "),): 1,  # shoot with Space
    (ord("d"),): 2,  # move right with D
    (ord("a"),): 3,  # move left with A
}

play(env, keys_to_action=mapping, fps=15)

env.close()
