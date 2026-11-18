"""
Plots the training curves saved in checkpoints (`Trainer.history`).

    uv run plot.py outputs/double_dqn/2026-11-13_15-00-00
    uv run plot.py outputs/dqn/<run> outputs/double_dqn/<run> --out figures/compare
"""
import argparse
from pathlib import Path

import torch

from src.utils.checkpoint import resolve_ckpt
from src.utils.visualize import plot_metric

# key -> plot_metric options
METRICS = {
    'episode_return': {'smooth': 100},
    'valid_score_mean': {'std_key': 'valid_score_std'},
    'q_value': {},
    'loss': {},
}

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('paths', nargs='+', type=Path, help='run directories or checkpoint files (the latest checkpoint of a run is used)')
    parser.add_argument('--out', type=Path, default=None, help='output directory (default: <run>/figures for one run; required for several)')
    args = parser.parse_args()

    if args.out is None and len(args.paths) > 1:
        parser.error('--out is required when plotting several runs')

    histories = {}
    for path in args.paths:
        ckpt = resolve_ckpt(path)
        run_dir = ckpt.parent.parent  # <run>/checkpoints/steps_<N>.pth
        histories[f'{run_dir.parent.name}/{run_dir.name}'] = torch.load(ckpt, map_location='cpu', weights_only=False)['history']

    out = args.out or run_dir / 'figures'
    out.mkdir(parents=True, exist_ok=True)
    for key, options in METRICS.items():
        plot_metric(histories, key, out / f'{key}.svg', **options)
    print(f'Figures saved under {out}')

if __name__ == '__main__':
    main()
