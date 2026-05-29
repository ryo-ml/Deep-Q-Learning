from pathlib import Path
import matplotlib.pyplot as plt

from ..utils.logger import get_logger

logger = get_logger(__name__)

def plot_loss(steps: list, loss: list, path: Path) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.plot(steps, loss)
    ax.set_title('Loss')
    ax.set_xlabel('steps')
    ax.set_ylabel('loss')
    ax.grid(True)

    fig.savefig(path, dpi=400)

    plt.close(fig)
    logger.info(f'Loss plot saved to {path}')

def plot_q_value(steps: list, q_value: list, path: Path) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.plot(steps, q_value)
    ax.set_title('Q value')
    ax.set_xlabel('steps')
    ax.set_ylabel('Q')
    ax.grid(True)

    fig.savefig(path, dpi=400)

    plt.close(fig)
    logger.info(f'Q value plot saved to {path}')
