"""
Plots the training curves logged to <run>/metrics.jsonl.

    uv run plot.py outputs/double_dqn/2026-11-13_15-00-00
    uv run plot.py outputs/dqn/<run> outputs/double_dqn/<run> --out figures/compare
"""
import argparse
import json
from pathlib import Path

from src.utils.visualize import plot_metric

# key -> plot_metric options
METRICS = {
    'train/episode_return': {'smooth': 100},
    'valid/score_mean': {'std_key': 'valid/score_std'},
    'train/q_value': {},
    'train/loss': {},
    'train/epsilon': {},
}

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('paths', nargs='+', type=Path, help='run directories')
    parser.add_argument('--out', type=Path, default=None, help='output directory (default: <run>/figures for one run; required for several)')
    args = parser.parse_args()

    if args.out is None and len(args.paths) > 1:
        parser.error('--out is required when plotting several runs')

    runs = {}
    for run_dir in args.paths:
        with open(run_dir / 'metrics.jsonl') as f:
            runs[f'{run_dir.parent.name}/{run_dir.name}'] = [json.loads(line) for line in f]

    out = args.out or args.paths[0] / 'figures'
    out.mkdir(parents=True, exist_ok=True)
    for key, options in METRICS.items():
        plot_metric(runs, key, out / f'{key.replace("/", "_")}.svg', **options)
    print(f'Figures saved under {out}')

if __name__ == '__main__':
    main()
