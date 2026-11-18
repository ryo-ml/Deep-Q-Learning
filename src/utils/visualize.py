from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

def plot_metric(
    histories: dict[str, list[dict]],
    key: str,
    path: Path,
    smooth: int = 1,
    std_key: str | None = None,
) -> None:
    """
    Plots `key` against steps, one line per run.

    Args:
        histories: run label -> `Trainer.history`
        smooth: moving-average window; with smooth > 1, the raw values are drawn faintly behind
        std_key: draws mean ± std as a band (e.g. valid_score_std)
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    for label, history in histories.items():
        entries = [h for h in history if key in h]
        if not entries:
            continue
        steps = np.array([h['steps'] for h in entries])
        values = np.array([h[key] for h in entries])

        if smooth > 1 and len(values) >= smooth:
            line, = ax.plot(steps, values, alpha=0.2)
            smoothed = np.convolve(values, np.ones(smooth) / smooth, mode='valid')
            ax.plot(steps[smooth - 1:], smoothed, color=line.get_color(), label=label)
        else:
            line, = ax.plot(steps, values, label=label)

        if std_key is not None:
            std = np.array([h[std_key] for h in entries])
            ax.fill_between(steps, values - std, values + std, color=line.get_color(), alpha=0.2)

    ax.set_title(key if smooth <= 1 else f'{key} (moving average over {smooth})')
    ax.set_xlabel('steps')
    ax.set_ylabel(key)
    ax.grid(True)
    ax.legend()

    fig.savefig(path, dpi=200)
    plt.close(fig)
