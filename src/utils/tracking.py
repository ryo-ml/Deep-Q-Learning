from pathlib import Path

from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf
import wandb

def init_wandb(cfg: DictConfig, output_dir: Path) -> wandb.Run:
    """
    Starts a wandb run. Call `run.finish()` when training ends.
    With `wandb.mode=disabled`, every wandb call becomes a no-op.
    """
    load_dotenv()

    cfg_dict = OmegaConf.to_container(
        cfg,
        resolve=True,
        throw_on_missing=True,
    )
    cfg_dict.pop('wandb')

    return wandb.init(
        entity=cfg.wandb.entity,
        project=cfg.wandb.project,
        group=cfg.agent.name,
        name=f'{cfg.agent.name}/{output_dir.name}',  # same as outputs/<agent>/<datetime>
        config=cfg_dict,
        dir=output_dir,
        mode=cfg.wandb.mode,
    )
