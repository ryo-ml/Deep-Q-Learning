import re
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig

from src.utils.seed import set_seed
from src.env.breakout import make_env
from src.models.dqn import DQN
from src.training.rollout import run_episode
from src.utils.logger import get_logger

logger = get_logger(__name__)

def resolve_ckpt(path: Path) -> Path:
    """
    Returns the checkpoint file to evaluate.
    `path` is either a checkpoint file or a run directory; for a run directory,
    the checkpoint with the largest step count is used.
    """
    if path.is_file():
        return path

    ckpt_dir = path / 'checkpoints'
    ckpts = list(ckpt_dir.glob('steps_*.pth'))
    if not ckpts:
        raise FileNotFoundError(f'No checkpoints found in {ckpt_dir}')
    # compare step counts as numbers (string order puts steps_9900000 after steps_50000000)
    return max(ckpts, key=lambda p: int(re.search(r'steps_(\d+)', p.stem).group(1)))

def load_model(ckpt: Path, n_actions: int, device: str) -> DQN:
    """
    Builds the Q network and loads only its weights from a training checkpoint
    (the replay buffer, optimizer and target network are not created).
    """
    state = torch.load(ckpt, map_location=device, weights_only=False)
    model = DQN(n_actions).to(device)
    model.load_state_dict(state['agent']['online_model'])
    model.eval()
    logger.info(f'Loaded {ckpt} (step {state["steps"]:,})')
    return model

@hydra.main(version_base='1.3', config_path='config', config_name='eval')
def main(cfg: DictConfig) -> None:
    set_seed(cfg.seed)

    ckpt = resolve_ckpt(Path(cfg.eval.ckpt))
    env = make_env(render_mode='human')
    model = load_model(ckpt, env.action_space.n, cfg.device)

    try:
        for i in range(cfg.eval.episode):
            score = run_episode(model, env, cfg.device)
            logger.info(f'Episode {i}: score {score}')
    finally:
        env.close()

if __name__ == '__main__':
    main()
