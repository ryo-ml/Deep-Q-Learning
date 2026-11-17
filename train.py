from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from src.utils.tracking import init_wandb
from src.utils.seed import set_seed
from src.env.breakout import make_env
from src.agents.registry import get_agent_class
from src.training.trainer import Trainer

@hydra.main(version_base='1.3', config_path='config', config_name='config')
def main(cfg: DictConfig) -> None:
    output_dir = Path(HydraConfig.get().runtime.output_dir)

    set_seed(cfg.seed)
    env = make_env(
        cfg.env,
        seed=cfg.seed,
        render_mode='rgb_array',
        record_video=False,
    )
    agent = get_agent_class(cfg.agent.name)(cfg, env.action_space)
    trainer = Trainer(
        env,
        agent,
        cfg,
        output_dir=output_dir,
        verbose=True,
    )

    with init_wandb(cfg, output_dir):
        trainer.train()

if __name__ == '__main__':
    main()
