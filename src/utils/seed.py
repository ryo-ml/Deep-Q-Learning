from typing import Any
import random
import numpy as np
import torch

from .logger import get_logger

logger = get_logger(__name__)

def set_seed(cfg: dict[str, Any]) -> None:
    seed = cfg['seed']
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # MPS
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)

    # CUDA
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Ensure deterministic behavior (may reduce performance slightly)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    logger.info(f'Seed set to {seed}')
