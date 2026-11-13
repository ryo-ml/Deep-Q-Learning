# Atari DQN Implementation

A PyTorch implementation of deep reinforcement learning on [Atari Breakout](https://gymnasium.farama.org/v0.26.3/environments/atari/breakout/). The following algorithms are implemented:

- **DQNAgent** - Original DQN ([Mnih et al., 2013](https://arxiv.org/abs/1312.5602))
- **TargetDQNAgent** - DQN with Target Network ([Mnih et al., 2015](https://www.nature.com/articles/nature14236))
- **DoubleDQNAgent** - Double DQN ([van Hasselt et al., 2015](https://arxiv.org/abs/1509.06461))

## Requirements

- Python 3.11
- PyTorch
- Gymnasium
- ale-py

## Setup

```bash
conda env create -f environment.yaml
conda activate atari
```

## Project Structure

```
.
├── config/
│   ├── config.yaml               # base config (paper settings)
│   └── experiment/               # presets reproducing past runs
├── environment.yaml
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
python train.py

# choose an agent
python train.py agent.name=dqn

# override any value from the CLI
python train.py agent.name=target_dqn train.total_steps=2000000

# use a preset that reproduces a past run
python train.py +experiment=target_dqn

# run several agents in one go (multirun)
python train.py -m agent.name=dqn,target_dqn,double_dqn
```

## Evaluation

```bash
python eval.py agent.name=dqn eval.run_id=<run_id> eval.checkpoint_name=steps_10000000.pth
# or, with a preset
python eval.py +experiment=dqn
```

`agent.name` must match the agent used for training.

## Configuration

| Path | Content |
|------|---------|
| `config/config.yaml` | Base config. `agent.name` selects `dqn` / `target_dqn` / `double_dqn`. Differences from the original paper are noted in comments |
| `config/experiment/*.yaml` | Presets reproducing past runs (overrides on top of the base config) |

To print the resolved config without running: `python train.py --cfg job`

## Outputs

| Path | Content |
|------|---------|
| `checkpoints/` | Model weights and optimizer state |
| `videos/` | Recorded evaluation episodes |
| `figures/` | Loss and q value plot figures |
| `outputs/`, `multirun/` | Hydra run directories (resolved config, logs) |

## References

- Mnih et al. (2013) *Playing Atari with Deep Reinforcement Learning*
- Mnih et al. (2015) *Human-level control through deep reinforcement learning*
- van Hasselt et al. (2015) *Deep Reinforcement Learning with Double Q-learning*
