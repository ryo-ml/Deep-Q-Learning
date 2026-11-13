from pathlib import Path

import hydra
from omegaconf import DictConfig

from src.utils.seed import set_seed
from src.env.breakout import make_env
from src.agents.registry import get_agent_class
from src.training.trainer import Trainer

BASE_DIR = Path(__file__).resolve().parent

@hydra.main(version_base='1.3', config_path='config', config_name='config')
def main(cfg: DictConfig) -> None:
    set_seed(cfg['seed'])

    env = make_env(
        render_mode='rgb_array',
        record_video=False,
    )
    agent = get_agent_class(cfg['agent']['name'])(cfg, env.action_space.n)
    trainer = Trainer(
        env,
        agent,
        cfg,
        verbose=False,
    )
    trainer.load(BASE_DIR / 'checkpoints' / cfg['eval']['run_id'] / cfg['eval']['checkpoint_name'])

    for _ in range(cfg['eval']['episode']):
        trainer.valid(
            render_mode='human',
        )

if __name__ == '__main__':
    main()
