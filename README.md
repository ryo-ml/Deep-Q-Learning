# Atari DQN Implementation

A PyTorch implementation of deep reinforcement learning on [Atari Breakout](https://gymnasium.farama.org/v0.26.3/environments/atari/breakout/). The following algorithms are implemented:

- **DQNAgent** - Original DQN ([Mnih et al., 2013](https://arxiv.org/abs/1312.5602))
- **TargetDQNAgent** - DQN with Target Network ([Mnih et al., 2015](https://www.nature.com/articles/nature14236))
- **DoubleDQNAgent** - Double DQN ([van Hasselt et al., 2015](https://arxiv.org/abs/1509.06461))

## Requirements

- [uv](https://docs.astral.sh/uv/)
- Python 3.11 (installed automatically by uv)
- A [Weights & Biases](https://wandb.ai/) account (optional; see [Experiment tracking](#experiment-tracking))

## Setup

```bash
uv sync
```

Dependencies are defined in `pyproject.toml` and pinned in `uv.lock`.

## Project Structure

```
.
├── config/
│   ├── config.yaml               # base config for training (paper settings)
│   └── eval.yaml                 # config for evaluation
├── pyproject.toml
├── uv.lock
├── train.py                      # training entry point
├── eval.py                       # evaluation entry point
├── plot.py                       # training curves from checkpoints
├── scripts/
│   ├── play.py                   # play Breakout with the keyboard
│   └── random_action.py          # run a random policy
└── src/
    ├── agents/
    │   ├── dqn_agent.py          # DQNAgent -> TargetDQNAgent -> DoubleDQNAgent
    │   └── registry.py           # agent name -> class
    ├── models/
    │   └── dqn.py                # CNN
    ├── env/
    │   └── breakout.py           # env creation and preprocessing
    ├── buffer/
    │   └── replay_buffer.py
    ├── training/
    │   ├── trainer.py            # Trainer (warmup, training loop, validation, checkpoints)
    │   └── rollout.py            # plays one episode (validation and evaluation)
    └── utils/
        ├── checkpoint.py         # finds the latest checkpoint of a run
        ├── logger.py
        ├── seed.py
        ├── tracking.py           # wandb run setup
        └── visualize.py          # plotting functions used by plot.py
```

## Training

Configuration is managed with [Hydra](https://hydra.cc/).

```bash
# base config (Double DQN, paper settings)
uv run train.py

# choose an agent
uv run train.py agent.name=dqn

# override any value from the CLI
uv run train.py agent.name=target_dqn train.total_steps=2000000

# run several agents in one go (multirun)
uv run train.py -m agent.name=dqn,target_dqn,double_dqn
```

### Experiment tracking

Training metrics are logged to [Weights & Biases](https://wandb.ai/): `train/episode_return` (raw score, not clipped) at the end of each training episode, `train/loss`, `train/q_value` and `train/epsilon` every `train.log_interval` steps, and `valid/score_mean` / `valid/score_std` over `train.valid_episodes` episodes with the video of the first one every `train.valid_interval` steps. The full config (except the `wandb` section) is logged as the run's config.

Put your API key in a `.env` file at the repository root (gitignored):

```
WANDB_API_KEY=<your key>
```

```bash
# log to wandb (default: wandb.mode=online)
uv run train.py

# keep the run locally and upload it later with `wandb sync`
uv run train.py wandb.mode=offline

# turn wandb off (e.g. for quick tests); no API key is needed
uv run train.py wandb.mode=disabled
```

Runs are named `<agent>/<YYYY-MM-DD_HH-MM-SS>` (the same as the output directory) and grouped by agent in project `wandb.project`. `wandb.entity` is `null` by default, which uses your account's default entity (or `WANDB_ENTITY`).

## Evaluation

```bash
# a run directory: the checkpoint with the largest step count is used
uv run eval.py eval.ckpt=outputs/double_dqn/2026-11-13_15-00-00

# a specific checkpoint
uv run eval.py eval.ckpt=outputs/double_dqn/2026-11-13_15-00-00/checkpoints/steps_1000000.pth

# number of episodes to play (default: 10)
uv run eval.py eval.ckpt=<path> eval.episode=3

# show the game in a window (needs a display, e.g. on a Mac)
uv run eval.py eval.ckpt=<path> eval.render=true
```

Only the Q network (online network) weights are loaded from the checkpoint; the replay buffer, optimizer and target network are not created. Episodes are played with an epsilon-greedy policy (epsilon = 0.05) without a window by default, so evaluation also runs on a server; `eval.render=true` shows them on the screen. The score of each episode and their mean ± std are printed.

The other settings (`env.*`, `seed`, `device`) come from the current config, not from the run. If a run used different env settings, pass them on the CLI (e.g. `env.repeat_action_probability=0.25`).

## Plotting

The same metrics are also written to `metrics.jsonl` in the run directory (one JSON object per line, appended as training goes), so the training curves can be plotted without wandb:

```bash
# one run: figures are saved under <run>/figures
uv run plot.py outputs/double_dqn/2026-11-13_15-00-00

# several runs on the same axes (e.g. to compare agents)
uv run plot.py outputs/dqn/<run> outputs/double_dqn/<run> --out figures/compare
```

The plots use the same names as wandb: `train/episode_return` (moving average over 100 episodes), `valid/score_mean` (± std), `train/q_value`, `train/loss` and `train/epsilon`.

## Configuration

| Path | Content |
|------|---------|
| `config/config.yaml` | Base config for training. `agent.name` selects `dqn` / `target_dqn` / `double_dqn`. `loss` and `optimizer` are built with `hydra.utils.instantiate` (`_target_` is the class). Paper values are noted in comments |
| `config/eval.yaml` | Evaluation settings (`eval.ckpt`, `eval.episode`, `eval.render`) on top of the base config |
| `config/config.yaml` → `wandb` | wandb `project`, `entity` and `mode` (`online` / `offline` / `disabled`) |

To print the resolved config without running: `uv run train.py --cfg job`

As in the paper, losing a life is treated as the end of an episode for learning only: it is stored as `done` in the replay buffer, so the TD target is `r` there and the frame stack starts over, but the game is not reset. The env is reset only at game over, so `train/episode_return` and the validation / evaluation scores are the scores of whole games. This is not gymnasium's `AtariPreprocessing(terminal_on_life_loss=True)` (left `False` in `make_env`), which restarts the whole game after each lost life.

`seed` fixes Python, NumPy, PyTorch, the env's action space (all random actions are sampled from it) and the env itself (the number of no-op actions at each reset), so two runs with the same config give identical results on CPU. On GPU (CUDA / MPS) small differences can remain.

## Outputs

Each training run writes everything to its own directory:

```
outputs/<agent>/<YYYY-MM-DD_HH-MM-SS>/
├── .hydra/
│   ├── config.yaml      # resolved config used for the run
│   └── overrides.yaml   # CLI overrides
├── train.log
├── metrics.jsonl        # logged metrics (the same as wandb, except videos)
├── checkpoints/         # steps_<N>.pth (model weights, optimizer state)
├── figures/             # training curves (created by plot.py)
├── videos/              # first validation episode at each validation
└── wandb/               # wandb's local files for the run (unless wandb.mode=disabled)
```

Multirun (`-m`) jobs go to `multirun/<YYYY-MM-DD_HH-MM-SS>/<job number>_<agent>/` with the same layout.

Evaluation does not create an output directory.

## References

- Mnih et al. (2013) *Playing Atari with Deep Reinforcement Learning*
- Mnih et al. (2015) *Human-level control through deep reinforcement learning*
- van Hasselt et al. (2015) *Deep Reinforcement Learning with Double Q-learning*
