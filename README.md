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
│   ├── config_dqn.yaml
│   ├── config_target_dqn.yaml
│   └── config_double_dqn.yaml
├── environment.yaml
└── src/
    ├── eval.py
    ├── main.py
    ├── agents/
    │   ├── agent.py              # BaseAgent (ABC)
    │   ├── dqn_agent.py          # DQNAgent
    │   ├── target_dqn_agent.py   # TargetDQNAgent
    │   └── double_dqn_agent.py   # DoubleDQNAgent
    ├── models/
    │   └── dqn.py                # CNN
    ├── env/
    │   └── breakout.py
    ├── buffer/
    │   └── replay_buffer.py
    ├── training/
    │   └── trainer.py            # Trainer
    └── utils/
        ├── config.py
        ├── logger.py
        └── seed.py
```

## Training

```bash
python -m src.main
```

## Evaluation

```bash
python -m src.eval
```

The checkpoint to evaluate can be specified via `eval.checkpoint` in the config file.

## Configuration

Training parameters are configured via yaml files under `config/`. Differences from the original paper are noted in comments.

```yaml
seed: 42

env:
  name: ALE/Breakout-v5 # currently not used

agent:
  epsilon_start: 1.0
  epsilon_end: 0.1
  epsilon_annealing: 100000 # paper: 1000000
  gamma: 0.99

model:
  loss: mse

optimizer:
  type: rmsprop
  lr: 0.00025
  alpha: 0.95
  eps: 0.01

replay_buffer:
  capacity: 500000 # paper: 1000000
  batch_size: 32

train:
  total_steps: 2000000 # paper: 50000000
  n_fixed_states: 1000
  warmup_steps: 50000
  log_interval: 5000
  valid_interval: 10000 # paper: 1000000
  save_interval: 50000
  update_interval: 4
  target_update_interval: 5000 # paper: 10000

eval:
  episode: 10
  checkpoint: steps_2000000.pth

device: mps
```

## Outputs

| Path | Content |
|------|---------|
| `checkpoints/` | Model weights and optimizer state |
| `videos/` | Recorded evaluation episodes |

## References

- Mnih et al. (2013) *Playing Atari with Deep Reinforcement Learning*
- Mnih et al. (2015) *Human-level control through deep reinforcement learning*
- van Hasselt et al. (2015) *Deep Reinforcement Learning with Double Q-learning*
