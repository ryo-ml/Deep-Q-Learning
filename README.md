# Atari DQN Implementation

A PyTorch implementation of deep reinforcement learning on [Atari Breakout](https://gymnasium.farama.org/v0.26.3/environments/atari/breakout/). The following algorithms are implemented:

- **DQNAgent** - Original DQN ([Mnih et al., 2013](https://arxiv.org/abs/1312.5602))
- **TargetDQNAgent** - DQN with Target Network ([Mnih et al., 2015](https://www.nature.com/articles/nature14236))
- **DoubleDQNAgent** - Double DQN ([van Hasselt et al., 2015](https://arxiv.org/abs/1509.06461))

## Requirements

- [uv](https://docs.astral.sh/uv/)
- Python 3.11 (installed automatically by uv)

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
│   ├── eval.yaml                 # config for evaluation
│   └── experiment/               # presets reproducing past runs
├── pyproject.toml
├── uv.lock
├── train.py                      # training entry point
├── eval.py                       # evaluation entry point
├── scripts/
│   ├── play.py                   # play Breakout with the keyboard
│   └── random_action.py          # run a random policy
└── src/
    ├── agents/
    │   ├── agent.py              # BaseAgent (ABC)
    │   ├── dqn_agent.py          # DQNAgent
    │   ├── target_dqn_agent.py   # TargetDQNAgent
    │   ├── double_dqn_agent.py   # DoubleDQNAgent
    │   └── registry.py           # agent name -> class
    ├── models/
    │   └── dqn.py                # CNN
    ├── env/
    │   └── breakout.py
    ├── buffer/
    │   └── replay_buffer.py
    ├── training/
    │   └── trainer.py            # Trainer
    └── utils/
        ├── logger.py
        └── seed.py
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

# use a preset that reproduces a past run
uv run train.py +experiment=target_dqn

# run several agents in one go (multirun)
uv run train.py -m agent.name=dqn,target_dqn,double_dqn
```

## Evaluation

```bash
# a run directory: the checkpoint with the largest step count is used
uv run eval.py eval.ckpt=outputs/double_dqn/2026-11-13_15-00-00

# a specific checkpoint
uv run eval.py eval.ckpt=outputs/double_dqn/2026-11-13_15-00-00/checkpoints/steps_1000000.pth

# number of episodes to play (default: 10)
uv run eval.py eval.ckpt=<path> eval.episode=3
```

The agent and model settings are restored from the run's `.hydra/config.yaml`, so they always match the checkpoint. `device` is taken from the current config.

## Configuration

| Path | Content |
|------|---------|
| `config/config.yaml` | Base config for training. `agent.name` selects `dqn` / `target_dqn` / `double_dqn`. Differences from the original paper are noted in comments |
| `config/eval.yaml` | Evaluation settings (`eval.ckpt`, `eval.episode`) on top of the base config |
| `config/experiment/*.yaml` | Presets reproducing past runs (overrides on top of the base config) |

To print the resolved config without running: `uv run train.py --cfg job`

## Outputs

Each training run writes everything to its own directory:

```
outputs/<agent>/<YYYY-MM-DD_HH-MM-SS>/
├── .hydra/
│   ├── config.yaml      # resolved config used for the run
│   └── overrides.yaml   # CLI overrides
├── train.log
├── checkpoints/         # steps_<N>.pth (model weights, optimizer state, history)
├── figures/             # loss and q value plots
└── videos/              # recorded validation episodes
```

Multirun (`-m`) jobs go to `multirun/<YYYY-MM-DD_HH-MM-SS>/<job number>_<agent>/` with the same layout.

Evaluation does not create an output directory.

## References

- Mnih et al. (2013) *Playing Atari with Deep Reinforcement Learning*
- Mnih et al. (2015) *Human-level control through deep reinforcement learning*
- van Hasselt et al. (2015) *Deep Reinforcement Learning with Double Q-learning*
