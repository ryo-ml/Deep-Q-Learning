from pathlib import Path

from .utils.seed import set_seed
from .utils.config import load_config
from .env.breakout import make_env
from .agents.dqn_agent import DQNAgent
from .training.trainer import Trainer

BASE_DIR = Path(__file__).resolve().parent.parent
CFG_DIR = BASE_DIR / 'config'
CKPT_DIR = BASE_DIR / 'checkpoints'

def main() -> None:
    cfg = load_config(CFG_DIR / 'config_double_dqn.yaml')

    set_seed(cfg['seed'])

    env = make_env(
        render_mode='rgb_array',
        record_video=False,
    )
    agent = DQNAgent(cfg, env.action_space.n)
    trainer = Trainer(
        env,
        agent,
        cfg,
        verbose=False,
    )
    trainer.load(CKPT_DIR / cfg['eval']['checkpoint'])

    for _ in range(cfg['eval']['episode']):
        trainer.valid(
            render_mode='human',
        )

if __name__ == '__main__':
    main()
