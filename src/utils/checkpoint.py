import re
from pathlib import Path

def resolve_ckpt(path: Path) -> Path:
    """
    Returns the checkpoint file for `path`, which is either a checkpoint file or a run
    directory; for a run directory, the checkpoint with the largest step count is used.
    """
    if path.is_file():
        return path

    ckpt_dir = path / 'checkpoints'
    ckpts = list(ckpt_dir.glob('steps_*.pth'))
    if not ckpts:
        raise FileNotFoundError(f'No checkpoints found in {ckpt_dir}')
    # compare step counts as numbers (string order puts steps_9900000 after steps_50000000)
    return max(ckpts, key=lambda p: int(re.search(r'steps_(\d+)', p.stem).group(1)))
