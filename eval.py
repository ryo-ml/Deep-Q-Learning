import re
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from src.utils.seed import set_seed
from src.env.breakout import make_env
from src.agents.registry import get_agent_class
from src.training.trainer import Trainer

def resolve_ckpt(path: Path) -> tuple[Path, Path]:
    """
    Returns (checkpoint file, run directory).
    `path` is either a checkpoint file or a run directory; for a run directory,
    the checkpoint with the largest step count is used.
    """
    if path.is_file():
        return path, path.parent.parent

    ckpt_dir = path / 'checkpoints'
    ckpts = list(ckpt_dir.glob('steps_*.pth'))
    if not ckpts:
        raise FileNotFoundError(f'No checkpoints found in {ckpt_dir}')
    # compare step counts as numbers (string order puts steps_9900000 after steps_50000000)
    latest = max(ckpts, key=lambda p: int(re.search(r'steps_(\d+)', p.stem).group(1)))
    return latest, path

def load_train_cfg(run_dir: Path, cfg: DictConfig) -> DictConfig:
    """
    Loads the config the run was trained with, so the agent matches the checkpoint.
    """
    saved = run_dir / '.hydra' / 'config.yaml'
    if not saved.exists():
        raise FileNotFoundError(f'{saved} not found: {run_dir} is not a run directory')

    train_cfg = OmegaConf.load(saved)
    train_cfg.device = cfg.device # device depends on the machine running the evaluation
    return train_cfg

@hydra.main(version_base='1.3', config_path='config', config_name='eval')
def main(cfg: DictConfig) -> None:
    ckpt, run_dir = resolve_ckpt(Path(cfg.eval.ckpt))
    train_cfg = load_train_cfg(run_dir, cfg)

    set_seed(train_cfg.seed)

    env = make_env(
        render_mode='rgb_array',
        record_video=False,
    )
    agent = get_agent_class(train_cfg.agent.name)(train_cfg, env.action_space.n)
    trainer = Trainer(
        env,
        agent,
        train_cfg,
        output_dir=run_dir,
        verbose=False,
    )
    trainer.load(ckpt)

    for _ in range(cfg.eval.episode):
        trainer.valid(
            render_mode='human',
        )

if __name__ == '__main__':
    main()
