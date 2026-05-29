import argparse
from pathlib import Path

from .utils.seed import set_seed
from .utils.config import load_config
from .env.breakout import make_env
from .agents.dqn_agent import DQNAgent
from .training.trainer import Trainer

BASE_DIR = Path(__file__).resolve().parent.parent

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', required=True, help='path to yaml configuration file')
    args = parser.parse_args()

    cfg = load_config(BASE_DIR / 'config' / args.config)

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
    trainer.load(BASE_DIR / 'checkpoints' / cfg['eval']['run_id'] / cfg['eval']['checkpoint_name'])

    for _ in range(cfg['eval']['episode']):
        trainer.valid(
            render_mode='human',
        )

if __name__ == '__main__':
    main()
